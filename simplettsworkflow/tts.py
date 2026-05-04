from __future__ import annotations

import json
import os
from importlib.util import find_spec
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from .settings import MODEL_ID, OUTPUT_DIR, VOICE_DESIGN_MODEL_ID


MODE_CLONE = "clone"
MODE_VOICE_DESIGN = "voice_design"
MODE_VOICE_DESIGN_THEN_CLONE = "voice_design_then_clone"


@dataclass(frozen=True)
class GeneratedAudio:
    index: int
    text: str
    filename: str
    path: str
    url: str


@dataclass(frozen=True)
class GenerationResult:
    output_dir: str
    items: list[GeneratedAudio]


class QwenTTSService:
    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        model_id: str | None = None,
        voice_design_model_id: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.model_id = model_id or os.getenv("QWEN_TTS_MODEL", MODEL_ID)
        self.voice_design_model_id = voice_design_model_id or os.getenv(
            "QWEN_TTS_VOICE_DESIGN_MODEL",
            VOICE_DESIGN_MODEL_ID,
        )
        self._models: dict[str, Any] = {}

    def _load_model(self, model_id: str) -> Any:
        if model_id in self._models:
            return self._models[model_id]

        from qwen_tts import Qwen3TTSModel
        import torch

        kwargs: dict[str, Any] = {}
        requested_device = os.getenv("QWEN_TTS_DEVICE")
        use_cuda = requested_device == "cuda" or (
            requested_device is None and torch.cuda.is_available()
        )

        # Keep model loading lazy so importing the web app never downloads weights.
        if use_cuda:
            kwargs.update(
                {
                    "device_map": "cuda:0",
                    "dtype": torch.bfloat16,
                }
            )
            if (
                os.getenv("QWEN_TTS_FLASH_ATTENTION", "1") != "0"
                and find_spec("flash_attn") is not None
            ):
                kwargs["attn_implementation"] = "flash_attention_2"
        else:
            kwargs.update({"device_map": "cpu", "dtype": torch.float32})

        self._models[model_id] = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        return self._models[model_id]

    def _load_clone_model(self) -> Any:
        return self._load_model(self.model_id)

    def _load_voice_design_model(self) -> Any:
        return self._load_model(self.voice_design_model_id)

    def generate_voice_clone(
        self,
        *,
        ref_audio_path: Path,
        ref_text: str,
        texts: list[str],
        language: str,
        emotion_instruction: str | None = None,
    ) -> GenerationResult:
        clean_texts = [line.strip() for line in texts if line.strip()]
        if not clean_texts:
            raise ValueError("At least one target text line is required.")

        model = self._load_clone_model()
        output_run_dir = self._create_output_run_dir()

        # Reuse the clone prompt so one reference clip can generate many lines efficiently.
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text.strip(),
        )

        languages = self._languages_for_batch(language, len(clean_texts))

        target_text: str | list[str] = clean_texts[0] if len(clean_texts) == 1 else clean_texts
        wavs, sample_rate = model.generate_voice_clone(
            text=target_text,
            language=languages,
            voice_clone_prompt=voice_clone_prompt,
        )

        items: list[GeneratedAudio] = []
        for index, (original_text, wav) in enumerate(zip(clean_texts, wavs), start=1):
            filename = f"line_{index:03}.wav"
            audio_path = output_run_dir / filename
            sf.write(str(audio_path), wav, sample_rate)
            items.append(
                GeneratedAudio(
                    index=index,
                    text=original_text,
                    filename=filename,
                    path=str(audio_path),
                    url=f"/outputs/{output_run_dir.name}/{filename}",
                )
            )

        self._write_metadata(
            output_run_dir=output_run_dir,
            mode=MODE_CLONE,
            model_id=self.model_id,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            language=language,
            emotion_instruction=emotion_instruction,
            items=items,
        )
        return GenerationResult(output_dir=str(output_run_dir), items=items)

    def generate_voice_design(
        self,
        *,
        texts: list[str],
        language: str,
        emotion_instruction: str,
    ) -> GenerationResult:
        clean_texts = [line.strip() for line in texts if line.strip()]
        instruction = emotion_instruction.strip()
        if not clean_texts:
            raise ValueError("At least one target text line is required.")
        if not instruction:
            raise ValueError("Emotion/style instruction is required for voice design mode.")

        model = self._load_voice_design_model()
        output_run_dir = self._create_output_run_dir()
        languages = self._languages_for_batch(language, len(clean_texts))
        target_text: str | list[str] = clean_texts[0] if len(clean_texts) == 1 else clean_texts
        instruct: str | list[str] = instruction if len(clean_texts) == 1 else [instruction] * len(clean_texts)

        wavs, sample_rate = model.generate_voice_design(
            text=target_text,
            language=languages,
            instruct=instruct,
        )

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            mode=MODE_VOICE_DESIGN,
            model_id=self.voice_design_model_id,
            ref_audio_path=None,
            ref_text="",
            language=language,
            emotion_instruction=instruction,
            items=items,
        )
        return GenerationResult(output_dir=str(output_run_dir), items=items)

    def generate_voice_design_then_clone(
        self,
        *,
        texts: list[str],
        language: str,
        emotion_instruction: str,
        design_ref_text: str,
    ) -> GenerationResult:
        clean_texts = [line.strip() for line in texts if line.strip()]
        instruction = emotion_instruction.strip()
        clean_ref_text = design_ref_text.strip()
        if not clean_texts:
            raise ValueError("At least one target text line is required.")
        if not instruction:
            raise ValueError("Emotion/style instruction is required for voice design then clone mode.")
        if not clean_ref_text:
            raise ValueError("Design reference text is required for voice design then clone mode.")

        output_run_dir = self._create_output_run_dir()
        design_model = self._load_voice_design_model()
        clone_model = self._load_clone_model()

        # Generate one style reference clip with the real instruct API, then reuse it as clone prompt.
        ref_wavs, ref_sample_rate = design_model.generate_voice_design(
            text=clean_ref_text,
            language=language,
            instruct=instruction,
        )
        design_ref_audio_path = output_run_dir / "design_reference.wav"
        sf.write(str(design_ref_audio_path), ref_wavs[0], ref_sample_rate)

        voice_clone_prompt = clone_model.create_voice_clone_prompt(
            ref_audio=(ref_wavs[0], ref_sample_rate),
            ref_text=clean_ref_text,
        )
        languages = self._languages_for_batch(language, len(clean_texts))
        target_text: str | list[str] = clean_texts[0] if len(clean_texts) == 1 else clean_texts
        wavs, sample_rate = clone_model.generate_voice_clone(
            text=target_text,
            language=languages,
            voice_clone_prompt=voice_clone_prompt,
        )

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            mode=MODE_VOICE_DESIGN_THEN_CLONE,
            model_id=f"{self.voice_design_model_id} -> {self.model_id}",
            ref_audio_path=design_ref_audio_path,
            ref_text=clean_ref_text,
            language=language,
            emotion_instruction=instruction,
            items=items,
        )
        return GenerationResult(output_dir=str(output_run_dir), items=items)

    def _create_output_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_run_dir = self.output_dir / timestamp
        counter = 1
        while output_run_dir.exists():
            output_run_dir = self.output_dir / f"{timestamp}-{counter:02}"
            counter += 1

        # All generated audio and metadata stay under the project output folder.
        output_run_dir.mkdir(parents=True, exist_ok=False)
        return output_run_dir

    @staticmethod
    def _languages_for_batch(language: str, count: int) -> str | list[str]:
        normalized = language.strip() or "Auto"
        return normalized if count == 1 else [normalized] * count

    @staticmethod
    def _write_audio_items(
        output_run_dir: Path,
        clean_texts: list[str],
        wavs: list[Any],
        sample_rate: int,
    ) -> list[GeneratedAudio]:
        items: list[GeneratedAudio] = []
        for index, (original_text, wav) in enumerate(zip(clean_texts, wavs), start=1):
            filename = f"line_{index:03}.wav"
            audio_path = output_run_dir / filename
            sf.write(str(audio_path), wav, sample_rate)
            items.append(
                GeneratedAudio(
                    index=index,
                    text=original_text,
                    filename=filename,
                    path=str(audio_path),
                    url=f"/outputs/{output_run_dir.name}/{filename}",
                )
            )
        return items

    @staticmethod
    def _write_metadata(
        *,
        output_run_dir: Path,
        mode: str,
        model_id: str,
        ref_audio_path: Path | None,
        ref_text: str,
        language: str,
        emotion_instruction: str | None,
        items: list[GeneratedAudio],
    ) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "model": model_id,
            "reference_audio": str(ref_audio_path) if ref_audio_path else "",
            "reference_text": ref_text,
            "language": language,
            "emotion_instruction": emotion_instruction or "",
            "items": [item.__dict__ for item in items],
        }
        metadata_path = output_run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def split_text_lines(texts: str) -> list[str]:
    return [line.strip() for line in texts.splitlines() if line.strip()]
