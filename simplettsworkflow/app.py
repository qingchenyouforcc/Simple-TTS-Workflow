from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .settings import BASE_DIR, OUTPUT_DIR, UPLOAD_DIR
from .tts import QwenTTSService, split_text_lines


app = FastAPI(title="Simple Qwen3-TTS Workflow")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
service = QwenTTSService()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/generate")
async def generate(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    texts: str = Form(""),
    language: str = Form("Auto"),
    emotion_instruction: str = Form(""),
):
    if not ref_audio.filename:
        raise HTTPException(status_code=400, detail="Reference audio is required.")
    if not ref_text.strip():
        raise HTTPException(status_code=400, detail="Reference text is required.")

    lines = split_text_lines(texts)
    if not lines:
        raise HTTPException(status_code=400, detail="At least one target text line is required.")

    ref_audio_path = _save_upload(ref_audio)
    try:
        result = service.generate_voice_clone(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            texts=lines,
            language=language,
            emotion_instruction=emotion_instruction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc

    return JSONResponse(
        {
            "output_dir": result.output_dir,
            "items": [item.__dict__ for item in result.items],
        }
    )


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    run_dir = UPLOAD_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    destination = run_dir / f"reference{suffix}"
    with destination.open("wb") as file_obj:
        shutil.copyfileobj(upload.file, file_obj)
    return destination
