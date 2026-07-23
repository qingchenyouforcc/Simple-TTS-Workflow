import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simplettsworkflow.emotion import (
    AssistedSceneInput,
    EmotionAnalyzer,
    RESPONSE_FORMAT,
    parse_assisted_scene_blocks,
)


class FakeLlamaModel:
    def __init__(self, responses=None, token_count: int = 3) -> None:
        self.responses = list(responses or [])
        self.token_count = token_count
        self.calls = []
        self.closed = False

    def tokenize(self, text, add_bos=False):
        return list(range(self.token_count))

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"choices": [{"message": {"content": response}}]}

    def close(self):
        self.closed = True


def test_load_model_uses_local_path_and_gpu_layers(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "emotion.gguf"
    model_path.write_bytes(b"gguf")
    calls = []

    class FakeLlama(FakeLlamaModel):
        def __init__(self, **kwargs):
            calls.append(kwargs)
            super().__init__()

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    analyzer = EmotionAnalyzer(model_path=str(model_path), n_ctx=2048, n_gpu_layers=-1, main_gpu=1)

    analyzer._load_model()

    assert calls == [
        {
            "model_path": str(model_path.resolve()),
            "n_ctx": 2048,
            "n_gpu_layers": -1,
            "main_gpu": 1,
            "verbose": False,
        }
    ]


def test_analyze_lines_returns_distinct_instructions_and_releases_model() -> None:
    model = FakeLlamaModel(
        responses=[
            '{"instruction":"情绪：开心明亮；强度：较强；语速：较快；音高：稍高；音量：适中；节奏与停顿：轻快流畅。"}',
            '{"instruction":"情绪：悲伤克制；强度：中等；语速：缓慢；音高：低沉；音量：较轻；节奏与停顿：舒缓且停顿明显。"}',
        ]
    )
    analyzer = EmotionAnalyzer(model_path="unused.gguf")
    analyzer._model = model

    analyses = analyzer.analyze_lines(["太好了！", "我还是没能等到你。"])

    assert [analysis.index for analysis in analyses] == [1, 2]
    assert analyses[0].instruction.startswith("情绪：开心明亮")
    assert analyses[1].instruction.startswith("情绪：悲伤克制")
    assert analyses[0].description is None
    assert analyses[0].keywords == ()
    assert model.calls[0]["response_format"] == RESPONSE_FORMAT
    assert model.calls[0]["temperature"] == 0.2
    assert model.closed is True
    assert analyzer._model is None


def test_invalid_response_is_retried_once() -> None:
    model = FakeLlamaModel(
        responses=[
            "not json",
            '{"instruction":"情绪：紧张急促；强度：较强；语速：较快；音高：偏高；音量：稍强；节奏与停顿：急促且短暂停顿。"}',
        ]
    )
    analyzer = EmotionAnalyzer(model_path="unused.gguf")
    analyzer._model = model

    analyses = analyzer.analyze_lines(["快躲开！"])

    assert len(model.calls) == 2
    assert analyses[0].instruction.startswith("情绪：紧张急促")


def test_empty_delivery_field_is_retried() -> None:
    model = FakeLlamaModel(
        responses=[
            '{"instruction":"情绪：平静；强度：中；语速：中；音高：中；音量：中；节奏与停顿：。"}',
            '{"instruction":"情绪：平静安慰；强度：中等；语速：适中；音高：平稳；音量：适中；节奏与停顿：舒缓且自然停顿。"}',
        ]
    )
    analyzer = EmotionAnalyzer(model_path="unused.gguf")
    analyzer._model = model

    analyses = analyzer.analyze_lines(["别怕，我就在这里。"])

    assert len(model.calls) == 2
    assert analyses[0].instruction.endswith("舒缓且自然停顿。")


def test_second_failure_reports_line_and_releases_model() -> None:
    model = FakeLlamaModel(responses=["bad", "still bad"])
    analyzer = EmotionAnalyzer(model_path="unused.gguf")
    analyzer._model = model

    with pytest.raises(ValueError, match="第 1 行情绪分析失败"):
        analyzer.analyze_lines(["无法分析"])

    assert model.closed is True


def test_overlong_line_is_rejected_before_generation() -> None:
    model = FakeLlamaModel(responses=[], token_count=600)
    analyzer = EmotionAnalyzer(model_path="unused.gguf", n_ctx=1024)
    analyzer._model = model

    with pytest.raises(ValueError, match="第 1 行文本过长"):
        analyzer.analyze_lines(["很长的文本"])

    assert model.calls == []
    assert model.closed is True


def test_parse_assisted_scene_blocks_supports_batches_and_keyword_variants() -> None:
    items = parse_assisted_scene_blocks(
        "Alright, I should probably fix the filter first…\n"
        "体现疲惫与无奈，但保持稳定，不破音。\n"
        "keyword：疲惫，无奈，认命\n\n\n"
        "We actually made it!\n"
        "兴奋而惊喜，语速稍快。\n"
        "KEYWORD: 兴奋, 惊喜、释然"
    )

    assert [item.index for item in items] == [1, 2]
    assert items[0].text == "Alright, I should probably fix the filter first…"
    assert items[0].description == "体现疲惫与无奈，但保持稳定，不破音。"
    assert items[0].keywords == ("疲惫", "无奈", "认命")
    assert items[1].keywords == ("兴奋", "惊喜", "释然")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("原句\n描述", "第 1 个辅助条目格式错误"),
        ("原句\n描述\n关键词：疲惫", "第三行必须使用 keyword"),
        ("原句\n描述\nkeyword：", "keyword 内容不能为空"),
        ("原句\n描述\nkeyword：疲惫\n多余一行", "当前为 4 行"),
    ],
)
def test_parse_assisted_scene_blocks_reports_malformed_item(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_assisted_scene_blocks(text)


def test_analyze_assisted_includes_description_and_keywords_in_prompt() -> None:
    model = FakeLlamaModel(
        responses=[
            '{"instruction":"情绪：疲惫无奈；强度：中等；语速：稍慢；音高：偏低；音量：适中；节奏与停顿：句尾稍作停顿。"}',
        ]
    )
    analyzer = EmotionAnalyzer(model_path="unused.gguf")
    analyzer._model = model
    item = AssistedSceneInput(
        index=1,
        text="Alright, I should probably fix the filter first…",
        description="体现疲惫与无奈，但保持稳定，不破音。",
        keywords=("疲惫", "无奈", "认命"),
    )

    analyses = analyzer.analyze_assisted([item])

    user_prompt = model.calls[0]["messages"][1]["content"]
    assert "原句：Alright, I should probably fix the filter first…" in user_prompt
    assert "情感描述：体现疲惫与无奈，但保持稳定，不破音。" in user_prompt
    assert "关键词：疲惫，无奈，认命" in user_prompt
    assert analyses[0].text == item.text
    assert analyses[0].description == item.description
    assert analyses[0].keywords == item.keywords
    assert model.closed is True


def test_assisted_failure_reports_item_number() -> None:
    model = FakeLlamaModel(responses=["bad", "still bad"])
    analyzer = EmotionAnalyzer(model_path="unused.gguf")
    analyzer._model = model

    with pytest.raises(ValueError, match="第 1 个辅助条目情绪分析失败"):
        analyzer.analyze_assisted(
            [AssistedSceneInput(1, "原句", "描述", ("关键词",))]
        )
