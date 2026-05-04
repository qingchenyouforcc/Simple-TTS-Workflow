from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from simplettsworkflow.app import app


client = TestClient(app)


def test_index_renders_form() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Simple Qwen3-TTS Workflow" in response.text
    assert "vox_controllable_clone" in response.text


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


def test_unknown_mode_returns_400() -> None:
    response = client.post(
        "/api/generate",
        data={"mode": "not_real", "texts": "hello"},
    )
    assert response.status_code == 400
