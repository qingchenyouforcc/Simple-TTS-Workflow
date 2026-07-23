import json
from pathlib import Path

import pytest

from simplettsworkflow.tts import (
    QwenTTSService,
    SCENE_DUBBING_MODE_ASSISTED,
    split_text_lines,
)
from simplettsworkflow.emotion import AssistedSceneInput, EmotionAnalysis


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


def test_scene_dubbing_analyzes_before_loading_vox_and_writes_metadata(monkeypatch, tmp_path: Path) -> None:
    events = []

    class FakeAnalyzer:
        model_id = "local/emotion-Q4_K_M.gguf"

        def analyze_lines(self, texts):
            events.append(("analyze", list(texts)))
            return [
                EmotionAnalysis(1, texts[0], "开心明亮，语速较快，音调稍高，音量适中，节奏轻快"),
                EmotionAnalysis(2, texts[1], "悲伤克制，语速缓慢，音调低沉，音量较轻，停顿明显"),
            ]

    fake_model = FakeVoxModel()
    service = QwenTTSService(output_dir=tmp_path, emotion_analyzer=FakeAnalyzer())

    def load_vox():
        events.append(("load_vox", None))
        return fake_model

    monkeypatch.setattr(service, "_load_voxcpm_model", load_vox)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: None)

    result = service.generate_scene_dubbing(
        ref_audio_path=tmp_path / "voice.wav",
        texts=["今天真是太好了！", "可是你再也不会回来了。"],
        cfg_value=1.8,
        inference_timesteps=12,
        normalize=True,
        denoise=False,
    )

    assert [event[0] for event in events] == ["analyze", "load_vox"]
    assert fake_model.generate_calls[0]["text"].startswith("(开心明亮")
    assert fake_model.generate_calls[0]["text"].endswith("今天真是太好了！")
    assert fake_model.generate_calls[1]["text"].startswith("(悲伤克制")
    assert fake_model.generate_calls[0]["reference_wav_path"] == str(tmp_path / "voice.wav")
    assert fake_model.generate_calls[0]["cfg_value"] == 1.8
    assert len(result.emotion_analyses) == 2

    metadata = json.loads((Path(result.output_dir) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "scene_dubbing"
    assert metadata["scene_dubbing_mode"] == "auto"
    assert metadata["emotion_analysis_model"] == "local/emotion-Q4_K_M.gguf"
    assert metadata["emotion_analyses"][0]["description"] is None
    assert metadata["emotion_analyses"][0]["keywords"] == []
    assert metadata["emotion_analyses"][1]["instruction"].startswith("悲伤克制")


def test_scene_dubbing_analysis_failure_does_not_load_vox(monkeypatch, tmp_path: Path) -> None:
    class FailingAnalyzer:
        model_id = "local/emotion.gguf"

        def analyze_lines(self, texts):
            raise ValueError("第 1 行情绪分析失败")

    service = QwenTTSService(output_dir=tmp_path, emotion_analyzer=FailingAnalyzer())
    monkeypatch.setattr(
        service,
        "_load_voxcpm_model",
        lambda: pytest.fail("VoxCPM2 must not load after analysis failure"),
    )

    with pytest.raises(ValueError, match="第 1 行情绪分析失败"):
        service.generate_scene_dubbing(
            ref_audio_path=tmp_path / "voice.wav",
            texts=["hello"],
        )

    assert list(tmp_path.iterdir()) == []


def test_assisted_scene_dubbing_uses_guidance_and_writes_mode_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events = []
    assisted_inputs = [
        AssistedSceneInput(
            index=1,
            text="Alright, I should probably fix the filter first…",
            description="体现疲惫与无奈，但保持稳定，不破音。",
            keywords=("疲惫", "无奈", "认命"),
        )
    ]

    class FakeAnalyzer:
        model_id = "local/emotion.gguf"

        def analyze_assisted(self, items):
            events.append(("analyze_assisted", list(items)))
            item = items[0]
            return [
                EmotionAnalysis(
                    index=item.index,
                    text=item.text,
                    instruction=(
                        "情绪：疲惫无奈；强度：中等；语速：稍慢；音高：偏低；"
                        "音量：适中；节奏与停顿：句尾稍作停顿。"
                    ),
                    description=item.description,
                    keywords=item.keywords,
                )
            ]

    fake_model = FakeVoxModel()
    service = QwenTTSService(output_dir=tmp_path, emotion_analyzer=FakeAnalyzer())

    def load_vox():
        events.append(("load_vox", None))
        return fake_model

    monkeypatch.setattr(service, "_load_voxcpm_model", load_vox)
    monkeypatch.setattr("simplettsworkflow.tts.sf.write", lambda path, wav, sr: None)

    result = service.generate_scene_dubbing(
        ref_audio_path=tmp_path / "voice.wav",
        texts=[assisted_inputs[0].text],
        scene_dubbing_mode=SCENE_DUBBING_MODE_ASSISTED,
        assisted_inputs=assisted_inputs,
    )

    assert [event[0] for event in events] == ["analyze_assisted", "load_vox"]
    assert fake_model.generate_calls[0]["text"].endswith(assisted_inputs[0].text)
    assert "体现疲惫与无奈" not in fake_model.generate_calls[0]["text"]
    assert result.items[0].text == assisted_inputs[0].text

    metadata = json.loads((Path(result.output_dir) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["scene_dubbing_mode"] == "assisted"
    assert metadata["emotion_analyses"][0]["description"] == assisted_inputs[0].description
    assert metadata["emotion_analyses"][0]["keywords"] == ["疲惫", "无奈", "认命"]


def test_assisted_scene_dubbing_requires_parsed_items_before_loading_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = QwenTTSService(output_dir=tmp_path)
    monkeypatch.setattr(
        service,
        "_load_voxcpm_model",
        lambda: pytest.fail("VoxCPM2 must not load without assisted items"),
    )

    with pytest.raises(ValueError, match="At least one assisted scene item"):
        service.generate_scene_dubbing(
            ref_audio_path=tmp_path / "voice.wav",
            texts=[],
            scene_dubbing_mode=SCENE_DUBBING_MODE_ASSISTED,
            assisted_inputs=[],
        )

    assert list(tmp_path.iterdir()) == []
