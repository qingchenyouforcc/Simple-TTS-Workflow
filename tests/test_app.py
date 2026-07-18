from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from simplettsworkflow.app import app


client = TestClient(app)


def test_index_renders_form() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Simple TTS Workflow" in response.text
    assert "vox_controllable_clone" in response.text
    assert "voice_preset" in response.text
    assert 'id="mode-select"' in response.text
    assert 'id="target-texts"' in response.text
    assert 'id="result-list"' in response.text


def test_vox_controllable_clone_requires_reference_audio() -> None:
    response = client.post(
        "/api/generate",
        data={"mode": "vox_controllable_clone", "texts": "hello"},
    )
    assert response.status_code == 400


def test_clone_generate_requires_target_text() -> None:
    response = client.post(
        "/api/generate",
        data={"mode": "clone", "ref_text": "hello", "texts": "", "language": "Auto"},
        files={"ref_audio": ("ref.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 400


def test_clone_generate_requires_reference_audio() -> None:
    response = client.post(
        "/api/generate",
        data={"mode": "clone", "ref_text": "hello", "texts": "target", "language": "Auto"},
    )
    assert response.status_code == 400


def test_vox_hifi_clone_requires_reference_text() -> None:
    response = client.post(
        "/api/generate",
        data={"mode": "vox_hifi_clone", "texts": "target"},
        files={"ref_audio": ("ref.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 400


def test_clone_generate_uses_service_without_prefixing_emotion(monkeypatch, tmp_path: Path) -> None:
    class FakeResult:
        output_dir = str(tmp_path)
        items = [
            SimpleNamespace(
                index=1,
                text="hello",
                filename="line_001.wav",
                path=str(tmp_path / "line_001.wav"),
                url="/outputs/run/line_001.wav",
            )
        ]

    def fake_generate_voice_clone(**kwargs):
        assert kwargs["texts"] == ["hello", "second"]
        assert kwargs["emotion_instruction"] == "happy"
        return FakeResult()

    monkeypatch.setattr("simplettsworkflow.app.service.generate_voice_clone", fake_generate_voice_clone)
    response = client.post(
        "/api/generate",
        data={
            "mode": "clone",
            "ref_text": "reference",
            "texts": "hello\n\nsecond",
            "language": "English",
            "emotion_instruction": "happy",
        },
        files={"ref_audio": ("ref.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["filename"] == "line_001.wav"


def test_voice_design_requires_instruction() -> None:
    response = client.post(
        "/api/generate",
        data={"mode": "voice_design", "texts": "hello", "language": "English"},
    )
    assert response.status_code == 400


def test_voice_design_uses_service(monkeypatch, tmp_path: Path) -> None:
    class FakeResult:
        output_dir = str(tmp_path)
        items = [
            SimpleNamespace(
                index=1,
                text="hello",
                filename="line_001.wav",
                path=str(tmp_path / "line_001.wav"),
                url="/outputs/run/line_001.wav",
            )
        ]

    def fake_generate_voice_design(**kwargs):
        assert kwargs["texts"] == ["hello"]
        assert kwargs["emotion_instruction"] == "sad"
        return FakeResult()

    monkeypatch.setattr("simplettsworkflow.app.service.generate_voice_design", fake_generate_voice_design)
    response = client.post(
        "/api/generate",
        data={
            "mode": "voice_design",
            "texts": "hello",
            "language": "English",
            "emotion_instruction": "sad",
        },
    )

    assert response.status_code == 200


def test_vox_controllable_clone_uses_service(monkeypatch, tmp_path: Path) -> None:
    class FakeResult:
        output_dir = str(tmp_path)
        items = [
            SimpleNamespace(
                index=1,
                text="hello",
                filename="line_001.wav",
                path=str(tmp_path / "line_001.wav"),
                url="/outputs/run/line_001.wav",
            )
        ]

    def fake_generate_vox_controllable_clone(**kwargs):
        assert kwargs["texts"] == ["hello"]
        assert kwargs["style_instruction"] == "sad"
        assert kwargs["cfg_value"] == 1.7
        assert kwargs["inference_timesteps"] == 12
        assert kwargs["normalize"] is True
        assert kwargs["denoise"] is False
        return FakeResult()

    monkeypatch.setattr(
        "simplettsworkflow.app.service.generate_vox_controllable_clone",
        fake_generate_vox_controllable_clone,
    )
    response = client.post(
        "/api/generate",
        data={
            "mode": "vox_controllable_clone",
            "texts": "hello",
            "emotion_instruction": "sad",
            "cfg_value": "1.7",
            "inference_timesteps": "12",
            "normalize": "true",
        },
        files={"ref_audio": ("ref.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200


def test_voice_presets_endpoint_returns_available_presets(monkeypatch, tmp_path: Path) -> None:
    preset_dir = tmp_path / "alice"
    preset_dir.mkdir()
    preset_audio = preset_dir / "alice.mp3"
    preset_audio.write_bytes(b"audio")
    (preset_dir / "preset.json").write_text(
        '{"name": "Alice", "reference": "alice.mp3", "reference_text": "hello from alice"}',
        encoding="utf-8",
    )
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "preset.json").write_text(
        '{"name": "Broken", "reference": "missing.wav", "reference_text": "missing audio"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("simplettsworkflow.app.ROLE_DIR", tmp_path)

    response = client.get("/api/voice-presets")

    assert response.status_code == 200
    assert response.json() == {
        "presets": [
            {
                "name": "Alice",
                "audio_filename": "alice.mp3",
                "ref_text": "hello from alice",
            }
        ]
    }


def test_vox_controllable_clone_uses_voice_preset_without_upload(monkeypatch, tmp_path: Path) -> None:
    preset_dir = tmp_path / "alice"
    preset_dir.mkdir()
    preset_audio = preset_dir / "alice.mp3"
    preset_audio.write_bytes(b"audio")
    (preset_dir / "preset.json").write_text(
        '{"name": "Alice", "reference": "alice.mp3", "reference_text": "hello from alice"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("simplettsworkflow.app.ROLE_DIR", tmp_path)

    class FakeResult:
        output_dir = str(tmp_path)
        items = [
            SimpleNamespace(
                index=1,
                text="hello",
                filename="line_001.wav",
                path=str(tmp_path / "line_001.wav"),
                url="/outputs/run/line_001.wav",
            )
        ]

    def fake_generate_vox_controllable_clone(**kwargs):
        assert kwargs["ref_audio_path"] == preset_audio.resolve()
        return FakeResult()

    monkeypatch.setattr(
        "simplettsworkflow.app.service.generate_vox_controllable_clone",
        fake_generate_vox_controllable_clone,
    )
    response = client.post(
        "/api/generate",
        data={"mode": "vox_controllable_clone", "voice_preset": "Alice", "texts": "hello"},
    )

    assert response.status_code == 200


def test_vox_hifi_clone_uses_voice_preset_text_without_upload(monkeypatch, tmp_path: Path) -> None:
    preset_dir = tmp_path / "alice"
    preset_dir.mkdir()
    preset_audio = preset_dir / "alice.mp3"
    preset_audio.write_bytes(b"audio")
    (preset_dir / "preset.json").write_text(
        '{"name": "Alice", "reference": "alice.mp3", "reference_text": "hello from alice"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("simplettsworkflow.app.ROLE_DIR", tmp_path)

    class FakeResult:
        output_dir = str(tmp_path)
        items = [
            SimpleNamespace(
                index=1,
                text="target",
                filename="line_001.wav",
                path=str(tmp_path / "line_001.wav"),
                url="/outputs/run/line_001.wav",
            )
        ]

    def fake_generate_vox_hifi_clone(**kwargs):
        assert kwargs["ref_audio_path"] == preset_audio.resolve()
        assert kwargs["ref_text"] == "hello from alice"
        return FakeResult()

    monkeypatch.setattr("simplettsworkflow.app.service.generate_vox_hifi_clone", fake_generate_vox_hifi_clone)
    response = client.post(
        "/api/generate",
        data={"mode": "vox_hifi_clone", "voice_preset": "Alice", "texts": "target"},
    )

    assert response.status_code == 200


def test_qwen_clone_uses_voice_preset_text_without_upload(monkeypatch, tmp_path: Path) -> None:
    preset_dir = tmp_path / "alice"
    preset_dir.mkdir()
    preset_audio = preset_dir / "alice.mp3"
    preset_audio.write_bytes(b"audio")
    (preset_dir / "preset.json").write_text(
        '{"name": "Alice", "reference": "alice.mp3", "reference_text": "hello from alice"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("simplettsworkflow.app.ROLE_DIR", tmp_path)

    class FakeResult:
        output_dir = str(tmp_path)
        items = [
            SimpleNamespace(
                index=1,
                text="target",
                filename="line_001.wav",
                path=str(tmp_path / "line_001.wav"),
                url="/outputs/run/line_001.wav",
            )
        ]

    def fake_generate_voice_clone(**kwargs):
        assert kwargs["ref_audio_path"] == preset_audio.resolve()
        assert kwargs["ref_text"] == "hello from alice"
        return FakeResult()

    monkeypatch.setattr("simplettsworkflow.app.service.generate_voice_clone", fake_generate_voice_clone)
    response = client.post(
        "/api/generate",
        data={"mode": "clone", "voice_preset": "Alice", "texts": "target", "language": "Auto"},
    )

    assert response.status_code == 200


def test_unknown_mode_returns_400() -> None:
    response = client.post(
        "/api/generate",
        data={"mode": "not_real", "texts": "hello"},
    )
    assert response.status_code == 400
