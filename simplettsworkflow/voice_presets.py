from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass(frozen=True)
class VoicePreset:
    name: str
    audio_path: Path
    ref_text: str

    @property
    def audio_filename(self) -> str:
        return self.audio_path.name

    def to_response(self) -> dict[str, str]:
        return {
            "name": self.name,
            "audio_filename": self.audio_filename,
            "ref_text": self.ref_text,
        }


def load_voice_presets(role_dir: Path) -> list[VoicePreset]:
    if not role_dir.exists():
        return []

    presets: list[VoicePreset] = []
    seen_names: set[str] = set()
    config_paths = sorted(path for path in role_dir.glob("*/*.json") if path.is_file())
    for config_path in config_paths:
        preset = parse_voice_preset(config_path, role_dir)
        if preset is None:
            continue
        if preset.name in seen_names:
            logger.warning("Skipping duplicate voice preset: name=%s file=%s", preset.name, config_path)
            continue
        seen_names.add(preset.name)
        presets.append(preset)
    return presets


def find_voice_preset(role_dir: Path, name: str) -> VoicePreset | None:
    clean_name = name.strip()
    if not clean_name:
        return None
    for preset in load_voice_presets(role_dir):
        if preset.name == clean_name:
            return preset
    return None


def parse_voice_preset(config_path: Path, role_dir: Path) -> VoicePreset | None:
    try:
        fields = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Skipping invalid voice preset config: file=%s error=%s", config_path, exc)
        return None
    if not isinstance(fields, dict):
        logger.warning("Skipping voice preset with non-object JSON: file=%s", config_path)
        return None

    name = str(fields.get("name", "")).strip()
    audio_value = str(fields.get("reference", "")).strip()
    ref_text = str(fields.get("reference_text", "")).strip()
    if not name or not audio_value or not ref_text:
        logger.warning("Skipping incomplete voice preset: file=%s", config_path)
        return None

    audio_path = (config_path.parent / audio_value).resolve()
    role_root = role_dir.resolve()
    if role_root != audio_path and role_root not in audio_path.parents:
        logger.warning("Skipping voice preset outside role dir: name=%s audio=%s", name, audio_value)
        return None
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning("Skipping voice preset with unsupported audio type: name=%s audio=%s", name, audio_path)
        return None
    if not audio_path.is_file():
        logger.warning("Skipping voice preset with missing audio: name=%s audio=%s", name, audio_path)
        return None

    return VoicePreset(name=name, audio_path=audio_path, ref_text=ref_text)
