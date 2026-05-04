from pathlib import Path

import pytest

from simplettsworkflow.tts import QwenTTSService, split_text_lines


class FakeModel:
    def __init__(self) -> None:
        self.prompt_args = None
        self.generate_args = None

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_args = kwargs
        return {"prompt": "cached"}

    def generate_voice_clone(self, **kwargs):
        self.generate_args = kwargs
        return [[0.0, 0.1], [0.2, 0.3]], 24000

    def generate_voice_design(self, **kwargs):
        self.generate_args = kwargs
        return [[0.4, 0.5], [0.6, 0.7]], 24000


def test_split_text_lines_ignores_blank_lines() -> None:
    assert split_text_lines(" first\n\n second \n") == ["first", "second"]


def test_generate_requires_at_least_one_text(tmp_path: Path) -> None:
    service = QwenTTSService(output_dir=tmp_path)
    with pytest.raises(ValueError):
        service.generate_voice_clone(
            ref_audio_path=tmp_path / "ref.wav",
            ref_text="hello",
            texts=[],
            language="Auto",
        )


def test_generate_voice_clone_writes_output_and_metadata(monkeypatch, tmp_path: Path) -> None:
    fake_model = FakeModel()
    writes = []
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(service, "_load_clone_model", lambda: fake_model)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: writes.append((path, wav, sr)))

    result = service.generate_voice_clone(
        ref_audio_path=tmp_path / "ref.wav",
        ref_text="reference words",
        texts=["hello", "world"],
        language="English",
        emotion_instruction="Speak happily.",
    )

    assert len(result.items) == 2
    assert writes[0][0].endswith("line_001.wav")
    assert writes[1][2] == 24000
    assert fake_model.prompt_args == {
        "ref_audio": str(tmp_path / "ref.wav"),
        "ref_text": "reference words",
    }
    assert fake_model.generate_args["language"] == ["English", "English"]
    assert fake_model.generate_args["text"] == ["hello", "world"]
    assert (Path(result.output_dir) / "metadata.json").exists()


def test_generate_voice_design_uses_instruct(monkeypatch, tmp_path: Path) -> None:
    fake_model = FakeModel()
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(service, "_load_voice_design_model", lambda: fake_model)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: None)

    result = service.generate_voice_design(
        texts=["hello", "world"],
        language="English",
        emotion_instruction="Speak sadly.",
    )

    assert len(result.items) == 2
    assert fake_model.generate_args["text"] == ["hello", "world"]
    assert fake_model.generate_args["instruct"] == ["Speak sadly.", "Speak sadly."]


def test_generate_voice_design_then_clone_reuses_designed_reference(monkeypatch, tmp_path: Path) -> None:
    design_model = FakeModel()
    clone_model = FakeModel()
    writes = []
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(service, "_load_voice_design_model", lambda: design_model)
    monkeypatch.setattr(service, "_load_clone_model", lambda: clone_model)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: writes.append((path, wav, sr)))

    result = service.generate_voice_design_then_clone(
        texts=["target one", "target two"],
        language="English",
        emotion_instruction="Speak softly.",
        design_ref_text="reference style text",
    )

    assert len(result.items) == 2
    assert design_model.generate_args == {
        "text": "reference style text",
        "language": "English",
        "instruct": "Speak softly.",
    }
    assert clone_model.prompt_args["ref_text"] == "reference style text"
    assert clone_model.generate_args["text"] == ["target one", "target two"]
    assert writes[0][0].endswith("design_reference.wav")
