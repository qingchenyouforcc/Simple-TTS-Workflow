from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_FALLBACK_ENDPOINT = "https://huggingface.co"
DEFAULT_ETAG_TIMEOUT = 60
DEFAULT_DOWNLOAD_TIMEOUT = 300
DEFAULT_MAX_WORKERS = 2
DEFAULT_RETRIES = 3


@dataclass(frozen=True)
class HuggingFaceDownloadSettings:
    endpoint: str
    fallback_endpoint: str | None
    etag_timeout: int
    download_timeout: int
    max_workers: int
    retries: int
    offline: bool


def configure_huggingface_downloads() -> HuggingFaceDownloadSettings:
    """Configure Hub networking before huggingface_hub is imported."""
    endpoint = os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    configured_fallback = os.getenv("SIMPLETTS_HF_FALLBACK_ENDPOINT")
    if configured_fallback is None:
        fallback_endpoint = (
            DEFAULT_HF_FALLBACK_ENDPOINT if endpoint == DEFAULT_HF_ENDPOINT else None
        )
    else:
        fallback_endpoint = configured_fallback.rstrip("/") or None
    if fallback_endpoint == endpoint:
        fallback_endpoint = None

    etag_timeout = _positive_int_env("HF_HUB_ETAG_TIMEOUT", DEFAULT_ETAG_TIMEOUT)
    download_timeout = _positive_int_env("HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_DOWNLOAD_TIMEOUT)
    max_workers = _positive_int_env("SIMPLETTS_HF_MAX_WORKERS", DEFAULT_MAX_WORKERS)
    retries = _positive_int_env("SIMPLETTS_HF_RETRIES", DEFAULT_RETRIES)
    offline = _env_bool("HF_HUB_OFFLINE", False)

    # huggingface_hub reads timeout variables at import time.
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(etag_timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    return HuggingFaceDownloadSettings(
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
        etag_timeout=etag_timeout,
        download_timeout=download_timeout,
        max_workers=max_workers,
        retries=retries,
        offline=offline,
    )


def resolve_huggingface_model(model_id_or_path: str) -> str:
    """Return a local model path, downloading a Hub repository if necessary."""
    local_path = Path(model_id_or_path).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    settings = configure_huggingface_downloads()
    # Import after configuring the environment so timeouts take effect.
    from huggingface_hub import snapshot_download

    logger.info(
        "Downloading model snapshot: model=%s endpoint=%s fallback_endpoint=%s "
        "etag_timeout=%ss download_timeout=%ss max_workers=%s retries=%s offline=%s",
        model_id_or_path,
        settings.endpoint,
        settings.fallback_endpoint,
        settings.etag_timeout,
        settings.download_timeout,
        settings.max_workers,
        settings.retries,
        settings.offline,
    )

    endpoints = [settings.endpoint]
    if settings.fallback_endpoint is not None and not settings.offline:
        endpoints.append(settings.fallback_endpoint)

    last_error: Exception | None = None
    for endpoint_index, endpoint in enumerate(endpoints):
        for attempt in range(1, settings.retries + 1):
            try:
                snapshot_path = snapshot_download(
                    repo_id=model_id_or_path,
                    endpoint=endpoint,
                    etag_timeout=settings.etag_timeout,
                    max_workers=settings.max_workers,
                    local_files_only=settings.offline,
                )
                logger.info(
                    "Model snapshot ready: model=%s endpoint=%s path=%s",
                    model_id_or_path,
                    endpoint,
                    snapshot_path,
                )
                return snapshot_path
            except Exception as exc:
                last_error = exc
                if settings.offline:
                    break
                if attempt < settings.retries:
                    delay = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "Model snapshot download attempt failed; retrying: "
                        "model=%s endpoint=%s attempt=%s/%s delay=%ss error=%r",
                        model_id_or_path,
                        endpoint,
                        attempt,
                        settings.retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)

        if endpoint_index + 1 < len(endpoints):
            logger.warning(
                "Model endpoint exhausted; switching to fallback: "
                "model=%s from_endpoint=%s to_endpoint=%s",
                model_id_or_path,
                endpoint,
                endpoints[endpoint_index + 1],
            )

    attempted_endpoints = ", ".join(endpoints)
    raise RuntimeError(
        f"模型下载失败（已尝试端点：{attempted_endpoints}；"
        f"每个端点最多 {settings.retries} 次）。请检查网络，或设置 HF_ENDPOINT "
        "指定下载源；已有完整缓存时可设置 HF_HUB_OFFLINE=1。"
    ) from last_error


def resolve_huggingface_file(repo_id: str, filename: str) -> str:
    """Return one cached Hub file, downloading only that file when needed."""
    settings = configure_huggingface_downloads()
    from huggingface_hub import hf_hub_download

    logger.info(
        "Downloading model file: repo=%s filename=%s endpoint=%s fallback_endpoint=%s "
        "etag_timeout=%ss download_timeout=%ss retries=%s offline=%s",
        repo_id,
        filename,
        settings.endpoint,
        settings.fallback_endpoint,
        settings.etag_timeout,
        settings.download_timeout,
        settings.retries,
        settings.offline,
    )

    endpoints = [settings.endpoint]
    if settings.fallback_endpoint is not None and not settings.offline:
        endpoints.append(settings.fallback_endpoint)

    last_error: Exception | None = None
    for endpoint_index, endpoint in enumerate(endpoints):
        for attempt in range(1, settings.retries + 1):
            try:
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    endpoint=endpoint,
                    etag_timeout=settings.etag_timeout,
                    local_files_only=settings.offline,
                )
                logger.info(
                    "Model file ready: repo=%s filename=%s endpoint=%s path=%s",
                    repo_id,
                    filename,
                    endpoint,
                    file_path,
                )
                return file_path
            except Exception as exc:
                last_error = exc
                if settings.offline:
                    break
                if attempt < settings.retries:
                    delay = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "Model file download attempt failed; retrying: repo=%s filename=%s "
                        "endpoint=%s attempt=%s/%s delay=%ss error=%r",
                        repo_id,
                        filename,
                        endpoint,
                        attempt,
                        settings.retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)

        if endpoint_index + 1 < len(endpoints):
            logger.warning(
                "Model file endpoint exhausted; switching to fallback: repo=%s filename=%s "
                "from_endpoint=%s to_endpoint=%s",
                repo_id,
                filename,
                endpoint,
                endpoints[endpoint_index + 1],
            )

    attempted_endpoints = ", ".join(endpoints)
    raise RuntimeError(
        f"模型文件下载失败（仓库：{repo_id}；文件：{filename}；"
        f"已尝试端点：{attempted_endpoints}；每个端点最多 {settings.retries} 次）。"
        "请检查网络，或通过 QWEN_EMOTION_MODEL_PATH 指定本地 GGUF 文件。"
    ) from last_error


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}
