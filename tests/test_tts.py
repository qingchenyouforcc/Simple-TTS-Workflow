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


class FakeVoxModel:
    def __init__(self) -> None:
        self.generate_calls = []
        self.tts_model = type("FakeTTSModel", (), {"sample_rate": 48000})()

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [0.8, 0.9]


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


def test_vox_controllable_clone_wraps_style_and_reference(monkeypatch, tmp_path: Path) -> None:
    fake_model = FakeVoxModel()
    writes = []
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(service, "_load_voxcpm_model", lambda: fake_model)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: writes.append((path, wav, sr)))

    result = service.generate_vox_controllable_clone(
        ref_audio_path=tmp_path / "voice.wav",
        texts=["hello", "second"],
        style_instruction="sad and slow",
        cfg_value=1.8,
        inference_timesteps=12,
        normalize=True,
        denoise=False,
    )

    assert len(result.items) == 2
    assert fake_model.generate_calls[0]["text"] == "(sad and slow)hello"
    assert fake_model.generate_calls[0]["reference_wav_path"] == str(tmp_path / "voice.wav")
    assert fake_model.generate_calls[0]["cfg_value"] == 1.8
    assert fake_model.generate_calls[0]["normalize"] is True
    assert writes[0][2] == 48000


def test_vox_controllable_clone_without_style_does_not_add_empty_prefix(monkeypatch, tmp_path: Path) -> None:
    fake_model = FakeVoxModel()
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(service, "_load_voxcpm_model", lambda: fake_model)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: None)

    service.generate_vox_controllable_clone(
        ref_audio_path=tmp_path / "voice.wav",
        texts=["hello"],
        style_instruction="",
    )

    assert fake_model.generate_calls[0]["text"] == "hello"


def test_vox_design_does_not_pass_reference_audio(monkeypatch, tmp_path: Path) -> None:
    fake_model = FakeVoxModel()
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(service, "_load_voxcpm_model", lambda: fake_model)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: None)

    service.generate_vox_design(texts=["hello"], style_instruction="warm voice")

    assert fake_model.generate_calls[0]["text"] == "(warm voice)hello"
    assert "reference_wav_path" not in fake_model.generate_calls[0]


def test_vox_hifi_clone_uses_prompt_text_without_style_prefix(monkeypatch, tmp_path: Path) -> None:
    fake_model = FakeVoxModel()
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(service, "_load_voxcpm_model", lambda: fake_model)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: None)

    service.generate_vox_hifi_clone(
        ref_audio_path=tmp_path / "voice.wav",
        ref_text="exact transcript",
        texts=["target text"],
        style_instruction="ignored style",
    )

    call = fake_model.generate_calls[0]
    assert call["text"] == "target text"
    assert call["prompt_wav_path"] == str(tmp_path / "voice.wav")
    assert call["prompt_text"] == "exact transcript"
    assert call["reference_wav_path"] == str(tmp_path / "voice.wav")


def test_load_voxcpm_model_does_not_pass_device(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return FakeVoxModel()

    monkeypatch.setenv("VOXCPM_DEVICE", "cuda:0")
    monkeypatch.setattr("voxcpm.VoxCPM", FakeVoxCPM)
    service = QwenTTSService(output_dir=tmp_path, voxcpm_model_id="local-vox")

    model = service._load_voxcpm_model()

    assert isinstance(model, FakeVoxModel)
    assert calls[0][0] == ("local-vox",)
    assert "device" not in calls[0][1]


def test_load_qwen_model_uses_downloaded_snapshot(monkeypatch, tmp_path: Path) -> None:
    calls = []
    fake_model = object()

    class FakeQwen3TTSModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return fake_model

    monkeypatch.setenv("QWEN_TTS_DEVICE", "cpu")
    monkeypatch.setattr(
        "simplettsworkflow.tts.resolve_huggingface_model",
        lambda model_id: str(tmp_path / "cached-snapshot"),
    )
    monkeypatch.setattr("qwen_tts.Qwen3TTSModel", FakeQwen3TTSModel)
    service = QwenTTSService(output_dir=tmp_path, model_id="owner/model")

    model = service._load_clone_model()

    assert model is fake_model
    assert calls[0][0] == (str(tmp_path / "cached-snapshot"),)
    assert calls[0][1]["device_map"] == "cpu"
