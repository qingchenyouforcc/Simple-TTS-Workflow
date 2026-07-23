from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .emotion import AssistedSceneInput, parse_assisted_scene_blocks
from .logging_config import configure_application_logging
from .settings import BASE_DIR, OUTPUT_DIR, ROLE_DIR, UPLOAD_DIR
from .tts import (
    MODE_CLONE,
    MODE_SCENE_DUBBING,
    MODE_VOX_CONTROLLABLE_CLONE,
    MODE_VOX_DESIGN,
    MODE_VOX_HIFI_CLONE,
    MODE_VOICE_DESIGN,
    MODE_VOICE_DESIGN_THEN_CLONE,
    QwenTTSService,
    SCENE_DUBBING_MODE_ASSISTED,
    SCENE_DUBBING_MODE_AUTO,
    split_text_lines,
)
from .voice_presets import VoicePreset, find_voice_preset, load_voice_presets


configure_application_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="Simple Qwen3-TTS Workflow")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
service = QwenTTSService()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ROLE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
logger.info(
    "Application initialized: base_dir=%s output_dir=%s role_dir=%s upload_dir=%s",
    BASE_DIR,
    OUTPUT_DIR,
    ROLE_DIR,
    UPLOAD_DIR,
)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    started_at = time.perf_counter()
    log = logger.debug if request.url.path.startswith(("/static/", "/outputs/")) else logger.info
    log("HTTP request started: method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "HTTP request failed: method=%s path=%s elapsed=%.2fs",
            request.method,
            request.url.path,
            time.perf_counter() - started_at,
        )
        raise
    log(
        "HTTP request completed: method=%s path=%s status=%s elapsed=%.2fs",
        request.method,
        request.url.path,
        response.status_code,
        time.perf_counter() - started_at,
    )
    return response


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/voice-presets")
async def voice_presets():
    presets = load_voice_presets(ROLE_DIR)
    logger.info(
        "Voice presets loaded: count=%s names=%s",
        len(presets),
        [preset.name for preset in presets],
    )
    return JSONResponse({"presets": [preset.to_response() for preset in presets]})


@app.post("/api/generate")
async def generate(
    ref_audio: UploadFile | None = File(None),
    mode: str = Form(MODE_VOX_CONTROLLABLE_CLONE),
    scene_dubbing_mode: str = Form(SCENE_DUBBING_MODE_AUTO),
    voice_preset: str = Form(""),
    ref_text: str = Form(""),
    texts: str = Form(""),
    language: str = Form("Auto"),
    emotion_instruction: str = Form(""),
    design_ref_text: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    normalize: bool = Form(False),
    denoise: bool = Form(False),
):
    request_started_at = time.perf_counter()
    assisted_inputs: list[AssistedSceneInput] | None = None
    if mode == MODE_SCENE_DUBBING:
        if scene_dubbing_mode not in {
            SCENE_DUBBING_MODE_AUTO,
            SCENE_DUBBING_MODE_ASSISTED,
        }:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported scene dubbing mode: {scene_dubbing_mode}",
            )
        if scene_dubbing_mode == SCENE_DUBBING_MODE_ASSISTED:
            try:
                assisted_inputs = parse_assisted_scene_blocks(texts)
            except ValueError as exc:
                logger.warning(
                    "Assisted scene input rejected before model or output setup: error=%s",
                    exc,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            lines = [item.text for item in assisted_inputs]
        else:
            lines = split_text_lines(texts)
    else:
        lines = split_text_lines(texts)
    if not lines:
        raise HTTPException(status_code=400, detail="At least one target text line is required.")

    logger.info(
        "TTS request received: mode=%s scene_dubbing_mode=%s items=%s language=%s cfg=%s "
        "steps=%s normalize=%s denoise=%s has_ref_audio=%s has_style=%s",
        mode,
        scene_dubbing_mode if mode == MODE_SCENE_DUBBING else "",
        len(lines),
        language,
        cfg_value,
        inference_timesteps,
        normalize,
        denoise,
        ref_audio is not None and bool(ref_audio.filename),
        bool(emotion_instruction.strip()),
    )

    try:
        preset = None
        if mode in {MODE_VOX_CONTROLLABLE_CLONE, MODE_SCENE_DUBBING, MODE_VOX_HIFI_CLONE, MODE_CLONE}:
            preset = _get_requested_preset(voice_preset)
        if mode == MODE_VOX_CONTROLLABLE_CLONE:
            ref_audio_path, _ = _resolve_reference_material(ref_audio, ref_text, preset, require_text=False)
            result = service.generate_vox_controllable_clone(
                ref_audio_path=ref_audio_path,
                texts=lines,
                style_instruction=emotion_instruction,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
        elif mode == MODE_SCENE_DUBBING:
            ref_audio_path, _ = _resolve_reference_material(ref_audio, ref_text, preset, require_text=False)
            result = service.generate_scene_dubbing(
                ref_audio_path=ref_audio_path,
                texts=lines,
                scene_dubbing_mode=scene_dubbing_mode,
                assisted_inputs=assisted_inputs,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
        elif mode == MODE_VOX_DESIGN:
            result = service.generate_vox_design(
                texts=lines,
                style_instruction=emotion_instruction,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
        elif mode == MODE_VOX_HIFI_CLONE:
            ref_audio_path, resolved_ref_text = _resolve_reference_material(ref_audio, ref_text, preset)
            result = service.generate_vox_hifi_clone(
                ref_audio_path=ref_audio_path,
                ref_text=resolved_ref_text,
                texts=lines,
                style_instruction=emotion_instruction,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
        elif mode == MODE_CLONE:
            ref_audio_path, resolved_ref_text = _resolve_reference_material(ref_audio, ref_text, preset)
            result = service.generate_voice_clone(
                ref_audio_path=ref_audio_path,
                ref_text=resolved_ref_text,
                texts=lines,
                language=language,
                emotion_instruction=emotion_instruction,
            )
        elif mode == MODE_VOICE_DESIGN:
            result = service.generate_voice_design(
                texts=lines,
                language=language,
                emotion_instruction=emotion_instruction,
            )
        elif mode == MODE_VOICE_DESIGN_THEN_CLONE:
            result = service.generate_voice_design_then_clone(
                texts=lines,
                language=language,
                emotion_instruction=emotion_instruction,
                design_ref_text=design_ref_text,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported generation mode: {mode}")
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "TTS request rejected: mode=%s elapsed=%.2fs error=%s",
            mode,
            time.perf_counter() - request_started_at,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "TTS generation failed: mode=%s elapsed=%.2fs",
            mode,
            time.perf_counter() - request_started_at,
        )
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    logger.info(
        "TTS request completed: mode=%s outputs=%s output_dir=%s elapsed=%.2fs",
        mode,
        len(result.items),
        result.output_dir,
        time.perf_counter() - request_started_at,
    )
    return JSONResponse(
        {
            "output_dir": result.output_dir,
            "items": [item.__dict__ for item in result.items],
            "emotion_analyses": [
                _serialize_emotion_analysis(analysis)
                for analysis in getattr(result, "emotion_analyses", [])
            ],
        }
    )


def _get_requested_preset(voice_preset: str) -> VoicePreset | None:
    preset_name = voice_preset.strip()
    if not preset_name:
        return None
    preset = find_voice_preset(ROLE_DIR, preset_name)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown voice preset: {preset_name}")
    return preset


def _resolve_reference_material(
    ref_audio: UploadFile | None,
    ref_text: str,
    preset: VoicePreset | None,
    *,
    require_text: bool = True,
) -> tuple[Path, str]:
    if preset is not None:
        return preset.audio_path, preset.ref_text

    if ref_audio is None or not ref_audio.filename:
        raise HTTPException(status_code=400, detail="Reference audio is required.")
    resolved_ref_text = ref_text.strip()
    if require_text and not resolved_ref_text:
        raise HTTPException(status_code=400, detail="Reference text is required.")
    return _save_upload(ref_audio), resolved_ref_text


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    run_dir = UPLOAD_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    destination = run_dir / f"reference{suffix}"
    with destination.open("wb") as file_obj:
        shutil.copyfileobj(upload.file, file_obj)
    logger.info("Saved reference upload: filename=%s path=%s", upload.filename, destination)
    return destination


def _serialize_emotion_analysis(analysis) -> dict:
    return {
        "index": analysis.index,
        "text": analysis.text,
        "instruction": analysis.instruction,
        "description": getattr(analysis, "description", None),
        "keywords": list(getattr(analysis, "keywords", ()) or ()),
    }
