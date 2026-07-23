from __future__ import annotations

import json
import logging
import os
import time
from importlib.util import find_spec
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from .emotion import AssistedSceneInput, EmotionAnalysis, EmotionAnalyzer
from .model_download import resolve_huggingface_model
from .settings import MODEL_ID, OUTPUT_DIR, VOICE_DESIGN_MODEL_ID, VOXCPM_MODEL_ID


logger = logging.getLogger(__name__)
ENGINE_QWEN = "qwen3_tts"
ENGINE_VOXCPM = "voxcpm2"
MODE_VOX_CONTROLLABLE_CLONE = "vox_controllable_clone"
MODE_VOX_DESIGN = "vox_design"
MODE_VOX_HIFI_CLONE = "vox_hifi_clone"
MODE_SCENE_DUBBING = "scene_dubbing"
SCENE_DUBBING_MODE_AUTO = "auto"
SCENE_DUBBING_MODE_ASSISTED = "assisted"
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
    emotion_analyses: list[EmotionAnalysis] = field(default_factory=list)


class QwenTTSService:
    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        model_id: str | None = None,
        voice_design_model_id: str | None = None,
        voxcpm_model_id: str | None = None,
        emotion_analyzer: EmotionAnalyzer | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.model_id = model_id or os.getenv("QWEN_TTS_MODEL", MODEL_ID)
        self.voice_design_model_id = voice_design_model_id or os.getenv(
            "QWEN_TTS_VOICE_DESIGN_MODEL",
            VOICE_DESIGN_MODEL_ID,
        )
        self.voxcpm_model_id = voxcpm_model_id or os.getenv("VOXCPM_MODEL", VOXCPM_MODEL_ID)
        self._models: dict[str, Any] = {}
        self._voxcpm_model: Any | None = None
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()

    def _load_model(self, model_id: str) -> Any:
        if model_id in self._models:
            logger.info("Using cached Qwen model: model=%s", model_id)
            return self._models[model_id]

        # Download the complete repository first, then load from the local
        # snapshot. This avoids extra remote HEAD requests from Transformers
        # while it probes optional custom generation code.
        model_path = resolve_huggingface_model(model_id)

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

        load_started_at = time.perf_counter()
        logger.info(
            "Loading Qwen model: model=%s local_path=%s kwargs=%s",
            model_id,
            model_path,
            _safe_kwargs(kwargs),
        )
        self._models[model_id] = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
        logger.info(
            "Qwen model loaded: model=%s elapsed=%.2fs",
            model_id,
            time.perf_counter() - load_started_at,
        )
        return self._models[model_id]

    def _load_clone_model(self) -> Any:
        return self._load_model(self.model_id)

    def _load_voice_design_model(self) -> Any:
        return self._load_model(self.voice_design_model_id)

    def _load_voxcpm_model(self) -> Any:
        if self._voxcpm_model is not None:
            logger.info("Using cached VoxCPM2 model: model=%s", self.voxcpm_model_id)
            return self._voxcpm_model

        from voxcpm import VoxCPM

        load_denoiser = _env_bool("VOXCPM_LOAD_DENOISER", False)
        optimize = _env_bool("VOXCPM_OPTIMIZE", True)
        load_started_at = time.perf_counter()
        logger.info(
            "Loading VoxCPM2 model: model=%s load_denoiser=%s optimize=%s",
            self.voxcpm_model_id,
            load_denoiser,
            optimize,
        )
        if os.getenv("VOXCPM_DEVICE"):
            logger.warning("VOXCPM_DEVICE is set but voxcpm 2.0.2 does not accept a device argument; ignoring it.")
        # VoxCPM2 is loaded only for Vox modes so Qwen workflows stay lightweight.
        self._voxcpm_model = VoxCPM.from_pretrained(
            self.voxcpm_model_id,
            load_denoiser=load_denoiser,
            optimize=optimize,
        )
        logger.info(
            "VoxCPM2 model loaded: model=%s elapsed=%.2fs",
            self.voxcpm_model_id,
            time.perf_counter() - load_started_at,
        )
        return self._voxcpm_model

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
        logger.info(
            "Qwen clone generation started: lines=%s language=%s ref_audio=%s output_dir=%s",
            len(clean_texts),
            language,
            ref_audio_path,
            output_run_dir,
        )

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
        logger.info("Qwen clone model call completed: lines=%s sample_rate=%s", len(clean_texts), sample_rate)

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
            engine=ENGINE_QWEN,
            mode=MODE_CLONE,
            model_id=self.model_id,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            language=language,
            style_instruction=emotion_instruction,
            items=items,
        )
        logger.info("Qwen clone generation completed: outputs=%s output_dir=%s", len(items), output_run_dir)
        return GenerationResult(output_dir=str(output_run_dir), items=items)

    def generate_vox_controllable_clone(
        self,
        *,
        ref_audio_path: Path,
        texts: list[str],
        style_instruction: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
    ) -> GenerationResult:
        clean_texts = _clean_texts(texts)
        model = self._load_voxcpm_model()
        output_run_dir = self._create_output_run_dir()
        sample_rate = int(model.tts_model.sample_rate)
        logger.info(
            "Vox controllable clone started: lines=%s ref_audio=%s cfg=%s steps=%s normalize=%s denoise=%s output_dir=%s",
            len(clean_texts),
            ref_audio_path,
            cfg_value,
            inference_timesteps,
            normalize,
            denoise,
            output_run_dir,
        )
        wavs = []
        for index, text in enumerate(clean_texts, start=1):
            logger.info(
                "Vox controllable clone line started: line=%s/%s text=%s",
                index,
                len(clean_texts),
                _preview(text),
            )
            wav = model.generate(
                text=_apply_vox_control_prefix(text, style_instruction),
                reference_wav_path=str(ref_audio_path),
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
            wavs.append(wav)
            logger.info("Vox controllable clone line completed: line=%s/%s", index, len(clean_texts))

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            engine=ENGINE_VOXCPM,
            mode=MODE_VOX_CONTROLLABLE_CLONE,
            model_id=self.voxcpm_model_id,
            ref_audio_path=ref_audio_path,
            ref_text="",
            language="Auto",
            style_instruction=style_instruction,
            items=items,
            generation_params=_vox_generation_params(cfg_value, inference_timesteps, normalize, denoise),
        )
        logger.info("Vox controllable clone completed: outputs=%s output_dir=%s", len(items), output_run_dir)
        return GenerationResult(output_dir=str(output_run_dir), items=items)

    def generate_scene_dubbing(
        self,
        *,
        ref_audio_path: Path,
        texts: list[str],
        scene_dubbing_mode: str = SCENE_DUBBING_MODE_AUTO,
        assisted_inputs: list[AssistedSceneInput] | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
    ) -> GenerationResult:
        workflow_started_at = time.perf_counter()
        if scene_dubbing_mode == SCENE_DUBBING_MODE_AUTO:
            clean_texts = _clean_texts(texts)
            if assisted_inputs is not None:
                raise ValueError("Assisted scene inputs are only valid in assisted mode.")
        elif scene_dubbing_mode == SCENE_DUBBING_MODE_ASSISTED:
            if not assisted_inputs:
                raise ValueError("At least one assisted scene item is required.")
            clean_texts = [item.text for item in assisted_inputs]
        else:
            raise ValueError(f"Unsupported scene dubbing mode: {scene_dubbing_mode}")

        analysis_started_at = time.perf_counter()
        logger.info(
            "Scene dubbing emotion analysis started: mode=%s items=%s analyzer=%s",
            scene_dubbing_mode,
            len(clean_texts),
            self.emotion_analyzer.model_id,
        )
        # EmotionAnalyzer releases its llama.cpp GPU context before returning,
        # so VoxCPM2 never has to share persistent VRAM with the analysis model.
        if scene_dubbing_mode == SCENE_DUBBING_MODE_ASSISTED:
            analyses = self.emotion_analyzer.analyze_assisted(assisted_inputs)
        else:
            analyses = self.emotion_analyzer.analyze_lines(clean_texts)
        logger.info(
            "Scene dubbing emotion analysis completed: mode=%s items=%s elapsed=%.2fs; "
            "Qwen context released, handing off to VoxCPM2",
            scene_dubbing_mode,
            len(analyses),
            time.perf_counter() - analysis_started_at,
        )

        model = self._load_voxcpm_model()
        output_run_dir = self._create_output_run_dir()
        sample_rate = int(model.tts_model.sample_rate)
        logger.info(
            "Scene dubbing generation started: mode=%s items=%s ref_audio=%s cfg=%s steps=%s "
            "normalize=%s denoise=%s output_dir=%s",
            scene_dubbing_mode,
            len(clean_texts),
            ref_audio_path,
            cfg_value,
            inference_timesteps,
            normalize,
            denoise,
            output_run_dir,
        )

        wavs = []
        for analysis in analyses:
            line_started_at = time.perf_counter()
            logger.info(
                "Scene dubbing line started: line=%s/%s instruction=%s text=%s",
                analysis.index,
                len(analyses),
                _preview(analysis.instruction),
                _preview(analysis.text),
            )
            wav = model.generate(
                text=_apply_vox_control_prefix(analysis.text, analysis.instruction),
                reference_wav_path=str(ref_audio_path),
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
            wavs.append(wav)
            logger.info(
                "Scene dubbing line completed: line=%s/%s elapsed=%.2fs samples=%s",
                analysis.index,
                len(analyses),
                time.perf_counter() - line_started_at,
                len(wav),
            )

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            engine=ENGINE_VOXCPM,
            mode=MODE_SCENE_DUBBING,
            model_id=self.voxcpm_model_id,
            ref_audio_path=ref_audio_path,
            ref_text="",
            language="Auto",
            style_instruction=None,
            items=items,
            generation_params=_vox_generation_params(cfg_value, inference_timesteps, normalize, denoise),
            emotion_analyses=analyses,
            emotion_analysis_model=self.emotion_analyzer.model_id,
            scene_dubbing_mode=scene_dubbing_mode,
        )
        logger.info(
            "Scene dubbing completed: mode=%s outputs=%s output_dir=%s elapsed=%.2fs",
            scene_dubbing_mode,
            len(items),
            output_run_dir,
            time.perf_counter() - workflow_started_at,
        )
        return GenerationResult(
            output_dir=str(output_run_dir),
            items=items,
            emotion_analyses=analyses,
        )

    def generate_vox_design(
        self,
        *,
        texts: list[str],
        style_instruction: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
    ) -> GenerationResult:
        clean_texts = _clean_texts(texts)
        model = self._load_voxcpm_model()
        output_run_dir = self._create_output_run_dir()
        sample_rate = int(model.tts_model.sample_rate)
        logger.info(
            "Vox design started: lines=%s cfg=%s steps=%s normalize=%s denoise=%s output_dir=%s",
            len(clean_texts),
            cfg_value,
            inference_timesteps,
            normalize,
            denoise,
            output_run_dir,
        )
        wavs = []
        for index, text in enumerate(clean_texts, start=1):
            logger.info("Vox design line started: line=%s/%s text=%s", index, len(clean_texts), _preview(text))
            wav = model.generate(
                text=_apply_vox_control_prefix(text, style_instruction),
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
            wavs.append(wav)
            logger.info("Vox design line completed: line=%s/%s", index, len(clean_texts))

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            engine=ENGINE_VOXCPM,
            mode=MODE_VOX_DESIGN,
            model_id=self.voxcpm_model_id,
            ref_audio_path=None,
            ref_text="",
            language="Auto",
            style_instruction=style_instruction,
            items=items,
            generation_params=_vox_generation_params(cfg_value, inference_timesteps, normalize, denoise),
        )
        logger.info("Vox design completed: outputs=%s output_dir=%s", len(items), output_run_dir)
        return GenerationResult(output_dir=str(output_run_dir), items=items)

    def generate_vox_hifi_clone(
        self,
        *,
        ref_audio_path: Path,
        ref_text: str,
        texts: list[str],
        style_instruction: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = False,
    ) -> GenerationResult:
        clean_ref_text = ref_text.strip()
        if not clean_ref_text:
            raise ValueError("Reference text is required for Vox Hi-Fi clone mode.")

        clean_texts = _clean_texts(texts)
        model = self._load_voxcpm_model()
        output_run_dir = self._create_output_run_dir()
        sample_rate = int(model.tts_model.sample_rate)
        logger.info(
            "Vox Hi-Fi clone started: lines=%s ref_audio=%s cfg=%s steps=%s normalize=%s denoise=%s output_dir=%s",
            len(clean_texts),
            ref_audio_path,
            cfg_value,
            inference_timesteps,
            normalize,
            denoise,
            output_run_dir,
        )
        wavs = []
        for index, text in enumerate(clean_texts, start=1):
            logger.info("Vox Hi-Fi clone line started: line=%s/%s text=%s", index, len(clean_texts), _preview(text))
            wav = model.generate(
                text=text,
                prompt_wav_path=str(ref_audio_path),
                prompt_text=clean_ref_text,
                reference_wav_path=str(ref_audio_path),
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
            wavs.append(wav)
            logger.info("Vox Hi-Fi clone line completed: line=%s/%s", index, len(clean_texts))

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            engine=ENGINE_VOXCPM,
            mode=MODE_VOX_HIFI_CLONE,
            model_id=self.voxcpm_model_id,
            ref_audio_path=ref_audio_path,
            ref_text=clean_ref_text,
            language="Auto",
            style_instruction=style_instruction,
            items=items,
            generation_params=_vox_generation_params(cfg_value, inference_timesteps, normalize, denoise),
        )
        logger.info("Vox Hi-Fi clone completed: outputs=%s output_dir=%s", len(items), output_run_dir)
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
        logger.info(
            "Qwen voice design started: lines=%s language=%s output_dir=%s",
            len(clean_texts),
            language,
            output_run_dir,
        )
        languages = self._languages_for_batch(language, len(clean_texts))
        target_text: str | list[str] = clean_texts[0] if len(clean_texts) == 1 else clean_texts
        instruct: str | list[str] = instruction if len(clean_texts) == 1 else [instruction] * len(clean_texts)

        wavs, sample_rate = model.generate_voice_design(
            text=target_text,
            language=languages,
            instruct=instruct,
        )
        logger.info("Qwen voice design model call completed: lines=%s sample_rate=%s", len(clean_texts), sample_rate)

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            engine=ENGINE_QWEN,
            mode=MODE_VOICE_DESIGN,
            model_id=self.voice_design_model_id,
            ref_audio_path=None,
            ref_text="",
            language=language,
            style_instruction=instruction,
            items=items,
        )
        logger.info("Qwen voice design completed: outputs=%s output_dir=%s", len(items), output_run_dir)
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
        logger.info(
            "Qwen design-then-clone started: lines=%s language=%s output_dir=%s",
            len(clean_texts),
            language,
            output_run_dir,
        )

        # Generate one style reference clip with the real instruct API, then reuse it as clone prompt.
        ref_wavs, ref_sample_rate = design_model.generate_voice_design(
            text=clean_ref_text,
            language=language,
            instruct=instruction,
        )
        design_ref_audio_path = output_run_dir / "design_reference.wav"
        sf.write(str(design_ref_audio_path), ref_wavs[0], ref_sample_rate)
        logger.info("Qwen designed reference audio written: path=%s sample_rate=%s", design_ref_audio_path, ref_sample_rate)

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
        logger.info("Qwen design-then-clone model call completed: lines=%s sample_rate=%s", len(clean_texts), sample_rate)

        items = self._write_audio_items(output_run_dir, clean_texts, wavs, sample_rate)
        self._write_metadata(
            output_run_dir=output_run_dir,
            engine=ENGINE_QWEN,
            mode=MODE_VOICE_DESIGN_THEN_CLONE,
            model_id=f"{self.voice_design_model_id} -> {self.model_id}",
            ref_audio_path=design_ref_audio_path,
            ref_text=clean_ref_text,
            language=language,
            style_instruction=instruction,
            items=items,
        )
        logger.info("Qwen design-then-clone completed: outputs=%s output_dir=%s", len(items), output_run_dir)
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
        logger.info("Created output directory: %s", output_run_dir)
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
            logger.info(
                "Generated audio written: index=%s filename=%s sample_rate=%s path=%s",
                index,
                filename,
                sample_rate,
                audio_path,
            )
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
        engine: str,
        mode: str,
        model_id: str,
        ref_audio_path: Path | None,
        ref_text: str,
        language: str,
        style_instruction: str | None,
        items: list[GeneratedAudio],
        generation_params: dict[str, Any] | None = None,
        emotion_analyses: list[EmotionAnalysis] | None = None,
        emotion_analysis_model: str | None = None,
        scene_dubbing_mode: str | None = None,
    ) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "engine": engine,
            "mode": mode,
            "model": model_id,
            "reference_audio": str(ref_audio_path) if ref_audio_path else "",
            "reference_text": ref_text,
            "language": language,
            "style_instruction": style_instruction or "",
            "emotion_instruction": style_instruction or "",
            "items": [item.__dict__ for item in items],
        }
        if generation_params is not None:
            metadata.update(generation_params)
        if emotion_analyses is not None:
            metadata["emotion_analysis_model"] = emotion_analysis_model or ""
            metadata["emotion_analyses"] = [analysis.__dict__ for analysis in emotion_analyses]
        if scene_dubbing_mode is not None:
            metadata["scene_dubbing_mode"] = scene_dubbing_mode
        metadata_path = output_run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Generation metadata written: path=%s", metadata_path)


def split_text_lines(texts: str) -> list[str]:
    return [line.strip() for line in texts.splitlines() if line.strip()]


def _clean_texts(texts: list[str]) -> list[str]:
    clean_texts = [line.strip() for line in texts if line.strip()]
    if not clean_texts:
        raise ValueError("At least one target text line is required.")
    return clean_texts


def _apply_vox_control_prefix(text: str, style_instruction: str | None) -> str:
    instruction = (style_instruction or "").strip()
    if not instruction:
        return text
    return f"({instruction}){text}"


def _vox_generation_params(
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise: bool,
) -> dict[str, Any]:
    return {
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "normalize": normalize,
        "denoise": denoise,
    }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _preview(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in kwargs.items()}
