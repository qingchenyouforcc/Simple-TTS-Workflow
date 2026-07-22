from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_download import resolve_huggingface_file
from .settings import EMOTION_MODEL_FILE, EMOTION_MODEL_REPO


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一名专业配音导演。请分析用户提供的一行待配音文本，并给出一条可直接交给语音合成模型的中文表达指令。
instruction 必须严格使用这一格式：情绪：…；强度：…；语速：…；音高：…；音量：…；节奏与停顿：…。
合格示例：情绪：温柔关切；强度：中等；语速：稍慢；音高：平稳；音量：轻柔；节奏与停顿：舒缓并自然停顿。
不要指定性别、年龄、口音或音色身份，不要复述、改写或引用正文，不要添加括号。只按要求输出 JSON。"""

RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "minLength": 2,
                "maxLength": 160,
            }
        },
        "required": ["instruction"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class EmotionAnalysis:
    index: int
    text: str
    instruction: str


class EmotionAnalyzer:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        model_repo: str | None = None,
        model_file: str | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        main_gpu: int | None = None,
    ) -> None:
        self.model_path = model_path if model_path is not None else os.getenv("QWEN_EMOTION_MODEL_PATH", "")
        self.model_repo = model_repo or os.getenv("QWEN_EMOTION_MODEL_REPO", EMOTION_MODEL_REPO)
        self.model_file = model_file or os.getenv("QWEN_EMOTION_MODEL_FILE", EMOTION_MODEL_FILE)
        self.n_ctx = n_ctx if n_ctx is not None else _int_env("QWEN_EMOTION_N_CTX", 4096, minimum=512)
        self.n_gpu_layers = (
            n_gpu_layers
            if n_gpu_layers is not None
            else _int_env("QWEN_EMOTION_N_GPU_LAYERS", -1, minimum=-1)
        )
        self.main_gpu = main_gpu if main_gpu is not None else _int_env("QWEN_EMOTION_MAIN_GPU", 1, minimum=0)
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        if self.model_path.strip():
            return str(Path(self.model_path).expanduser())
        return f"{self.model_repo}/{self.model_file}"

    def analyze_lines(self, texts: list[str]) -> list[EmotionAnalysis]:
        if not texts:
            raise ValueError("At least one target text line is required.")

        started_at = time.perf_counter()
        logger.info(
            "Emotion analysis batch started: lines=%s model=%s n_ctx=%s gpu_layers=%s main_gpu=%s",
            len(texts),
            self.model_id,
            self.n_ctx,
            self.n_gpu_layers,
            self.main_gpu,
        )
        model = self._load_model()
        analyses: list[EmotionAnalysis] = []
        try:
            for index, text in enumerate(texts, start=1):
                analyses.append(
                    EmotionAnalysis(
                        index=index,
                        text=text,
                        instruction=self._analyze_line(model, text, index),
                    )
                )
            logger.info(
                "Emotion analysis batch completed: lines=%s elapsed=%.2fs",
                len(analyses),
                time.perf_counter() - started_at,
            )
            return analyses
        finally:
            self.release_model()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        configured_path = self.model_path.strip()
        if configured_path:
            local_path = Path(configured_path).expanduser()
            if not local_path.is_file():
                raise ValueError(f"QWEN_EMOTION_MODEL_PATH does not point to a GGUF file: {local_path}")
            resolved_path = str(local_path.resolve())
        else:
            resolved_path = resolve_huggingface_file(self.model_repo, self.model_file)

        from llama_cpp import Llama

        load_started_at = time.perf_counter()
        logger.info(
            "Loading Qwen emotion analyzer: model=%s path=%s n_ctx=%s n_gpu_layers=%s main_gpu=%s",
            self.model_id,
            resolved_path,
            self.n_ctx,
            self.n_gpu_layers,
            self.main_gpu,
        )
        self._model = Llama(
            model_path=resolved_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self.main_gpu,
            verbose=False,
        )
        logger.info(
            "Qwen emotion analyzer loaded: model=%s elapsed=%.2fs",
            self.model_id,
            time.perf_counter() - load_started_at,
        )
        return self._model

    def _analyze_line(self, model: Any, text: str, line_index: int) -> str:
        token_count = len(model.tokenize(text.encode("utf-8"), add_bos=False))
        logger.info(
            "Emotion analysis line prepared: line=%s chars=%s tokens=%s text=%s",
            line_index,
            len(text),
            token_count,
            _preview(text),
        )
        if token_count > self.n_ctx - 512:
            raise ValueError(
                f"第 {line_index} 行文本过长，情绪分析需要少于 {self.n_ctx - 512} 个模型 token。"
            )

        last_error: Exception | None = None
        for attempt in range(1, 3):
            attempt_started_at = time.perf_counter()
            try:
                logger.info(
                    "Emotion analysis inference started: line=%s attempt=%s/2",
                    line_index,
                    attempt,
                )
                user_content = f"待分析正文如下，仅用于判断表演方式：\n{text}\n请输出配音指令。"
                if attempt > 1:
                    user_content = (
                        "上一次输出不合格。只描述表演方式，绝对不要复述正文，并完整填写规定的六个字段。\n"
                        f"正文：{text}"
                    )
                response = model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=RESPONSE_FORMAT,
                    temperature=0.2,
                    max_tokens=256,
                )
                content = response["choices"][0]["message"]["content"]
                payload = json.loads(content)
                instruction = " ".join(str(payload["instruction"]).split()).strip()
                if not 2 <= len(instruction) <= 160:
                    raise ValueError("instruction length is outside the allowed range")
                _validate_instruction(instruction, text)
                instruction = instruction.replace("(", "（").replace(")", "）")
                logger.info(
                    "Emotion analysis inference completed: line=%s attempt=%s/2 elapsed=%.2fs instruction=%s",
                    line_index,
                    attempt,
                    time.perf_counter() - attempt_started_at,
                    instruction,
                )
                return instruction
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Emotion analysis attempt failed: line=%s attempt=%s/2 elapsed=%.2fs error=%r",
                    line_index,
                    attempt,
                    time.perf_counter() - attempt_started_at,
                    exc,
                )

        raise ValueError(f"第 {line_index} 行情绪分析失败：模型未返回有效指令。") from last_error

    def release_model(self) -> None:
        release_started_at = time.perf_counter()
        model, self._model = self._model, None
        if model is None:
            return
        try:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        finally:
            del model
            gc.collect()
            logger.info(
                "Qwen emotion analyzer released before VoxCPM2 generation: elapsed=%.2fs",
                time.perf_counter() - release_started_at,
            )


def _int_env(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _validate_instruction(instruction: str, source_text: str) -> None:
    compact_instruction = "".join(instruction.split()).strip("。！？!?，,；;：:")
    compact_source = "".join(source_text.split()).strip("。！？!?，,；;：:")
    if compact_instruction == compact_source or (
        len(compact_source) >= 6 and compact_source in compact_instruction
    ):
        raise ValueError("instruction copied the source text")

    required_markers = ("情绪：", "强度：", "语速：", "音高：", "音量：", "节奏与停顿：")
    if not all(marker in instruction for marker in required_markers):
        raise ValueError("instruction did not include all required delivery fields")

    for index, marker in enumerate(required_markers):
        value_start = instruction.index(marker) + len(marker)
        if index + 1 < len(required_markers):
            value_end = instruction.index(required_markers[index + 1], value_start)
        else:
            value_end = len(instruction)
        value = instruction[value_start:value_end].strip(" 。！？!?，,；;：:")
        if not value:
            raise ValueError(f"instruction field {marker} was empty")


def _preview(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."
