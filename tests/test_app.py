from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from simplettsworkflow.app import app


client = TestClient(app)


def test_index_renders_form() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Simple Qwen3-TTS Workflow" in response.text


def test_generate_requires_target_text() -> None:
    response = client.post(
        "/api/generate",
        data={"ref_text": "hello", "texts": "", "language": "Auto"},
        files={"ref_audio": ("ref.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 400


def test_generate_uses_service(monkeypatch, tmp_path: Path) -> None:
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
            "ref_text": "reference",
            "texts": "hello\n\nsecond",
            "language": "English",
            "emotion_instruction": "happy",
        },
        files={"ref_audio": ("ref.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["filename"] == "line_001.wav"
