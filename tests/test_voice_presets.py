from pathlib import Path

from simplettsworkflow.voice_presets import load_voice_presets


def test_load_voice_presets_parses_valid_configs(tmp_path: Path) -> None:
    preset_dir = tmp_path / "alice"
    preset_dir.mkdir()
    (preset_dir / "alice.mp3").write_bytes(b"audio")
    (preset_dir / "preset.json").write_text(
        '{"name": "Alice", "reference": "alice.mp3", "reference_text": "hello from alice"}',
        encoding="utf-8",
    )

    presets = load_voice_presets(tmp_path)

    assert len(presets) == 1
    assert presets[0].name == "Alice"
    assert presets[0].audio_path == (preset_dir / "alice.mp3").resolve()
    assert presets[0].ref_text == "hello from alice"


def test_load_voice_presets_skips_invalid_configs(tmp_path: Path) -> None:
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    (valid_dir / "valid.mp3").write_bytes(b"audio")
    (valid_dir / "preset.json").write_text(
        '{"name": "Valid", "reference": "valid.mp3", "reference_text": "valid words"}',
        encoding="utf-8",
    )
    missing_audio_dir = tmp_path / "missing_audio"
    missing_audio_dir.mkdir()
    (missing_audio_dir / "preset.json").write_text(
        '{"name": "Missing", "reference": "missing.wav", "reference_text": "words"}',
        encoding="utf-8",
    )
    missing_text_dir = tmp_path / "missing_text"
    missing_text_dir.mkdir()
    (missing_text_dir / "preset.json").write_text(
        '{"name": "NoText", "reference": "valid.mp3"}',
        encoding="utf-8",
    )

    presets = load_voice_presets(tmp_path)

    assert [preset.name for preset in presets] == ["Valid"]


def test_load_voice_presets_skips_duplicate_names(tmp_path: Path) -> None:
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "first.mp3").write_bytes(b"audio")
    (second_dir / "second.mp3").write_bytes(b"audio")
    (first_dir / "preset.json").write_text(
        '{"name": "Same", "reference": "first.mp3", "reference_text": "first"}',
        encoding="utf-8",
    )
    (second_dir / "preset.json").write_text(
        '{"name": "Same", "reference": "second.mp3", "reference_text": "second"}',
        encoding="utf-8",
    )

    presets = load_voice_presets(tmp_path)

    assert len(presets) == 1
    assert presets[0].audio_filename == "first.mp3"


def test_load_voice_presets_ignores_root_configs(tmp_path: Path) -> None:
    (tmp_path / "alice.mp3").write_bytes(b"audio")
    (tmp_path / "alice.json").write_text(
        '{"name": "Alice", "reference": "alice.mp3", "reference_text": "hello"}',
        encoding="utf-8",
    )

    assert load_voice_presets(tmp_path) == []


def test_load_voice_presets_ignores_legacy_txt_configs(tmp_path: Path) -> None:
    preset_dir = tmp_path / "alice"
    preset_dir.mkdir()
    (preset_dir / "alice.mp3").write_bytes(b"audio")
    (preset_dir / "preset.txt").write_text(
        "预设名称：Alice\n预设参考音频：alice.mp3\n预设参考文本：hello\n",
        encoding="utf-8",
    )

    assert load_voice_presets(tmp_path) == []
