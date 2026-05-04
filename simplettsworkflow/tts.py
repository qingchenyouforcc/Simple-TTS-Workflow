from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from .settings import MODEL_ID, OUTPUT_DIR


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
    def __init__(self, output_dir: Path = OUTPUT_DIR, model_id: str | None = None) -> None:
        self.output_dir = output_dir
        self.model_id = model_id or os.getenv("QWEN_TTS_MODEL", MODEL_ID)
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

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
            if os.getenv("QWEN_TTS_FLASH_ATTENTION", "1") != "0":
                kwargs["attn_implementation"] = "flash_attention_2"
        else:
            kwargs.update({"device_map": "cpu", "dtype": torch.float32})

        self._model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
        return self._model

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

        model = self._load_model()
        output_run_dir = self._create_output_run_dir()

        # Reuse the clone prompt so one reference clip can generate many lines efficiently.
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text.strip(),
        )

        prepared_texts = [
            self._apply_experimental_emotion(line, emotion_instruction)
            for line in clean_texts
        ]
        languages = self._languages_for_batch(language, len(prepared_texts))

        target_text: str | list[str] = prepared_texts[0] if len(prepared_texts) == 1 else prepared_texts
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
            model_id=self.model_id,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            language=language,
            emotion_instruction=emotion_instruction,
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
    def _apply_experimental_emotion(text: str, emotion_instruction: str | None) -> str:
        instruction = (emotion_instruction or "").strip()
        if not instruction:
            return text

        # Qwen3-TTS Base voice clone docs do not define an instruct argument.
        # This conservative prefix lets users try style cues without claiming API support.
        return f"{instruction}\n{text}"

    @staticmethod
    def _write_metadata(
        *,
        output_run_dir: Path,
        model_id: str,
        ref_audio_path: Path,
        ref_text: str,
        language: str,
        emotion_instruction: str | None,
        items: list[GeneratedAudio],
    ) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_id,
            "reference_audio": str(ref_audio_path),
            "reference_text": ref_text,
            "language": language,
            "emotion_instruction": emotion_instruction or "",
            "emotion_mode": "experimental_prompt_prefix",
            "items": [item.__dict__ for item in items],
        }
        metadata_path = output_run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def split_text_lines(texts: str) -> list[str]:
    return [line.strip() for line in texts.splitlines() if line.strip()]


def supports_instruct_parameter(model_method: Any) -> bool:
    return "instruct" in inspect.signature(model_method).parameters
