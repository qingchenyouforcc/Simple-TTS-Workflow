from pathlib import Path

import pytest

from simplettsworkflow.model_download import (
    DEFAULT_HF_ENDPOINT,
    configure_huggingface_downloads,
    resolve_huggingface_file,
    resolve_huggingface_model,
)


def test_download_settings_use_resilient_defaults(monkeypatch) -> None:
    for name in (
        "HF_ENDPOINT",
        "SIMPLETTS_HF_FALLBACK_ENDPOINT",
        "HF_HUB_ETAG_TIMEOUT",
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "SIMPLETTS_HF_MAX_WORKERS",
        "SIMPLETTS_HF_RETRIES",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = configure_huggingface_downloads()

    assert settings.endpoint == DEFAULT_HF_ENDPOINT
    assert settings.fallback_endpoint == "https://huggingface.co"
    assert settings.etag_timeout == 60
    assert settings.download_timeout == 300
    assert settings.max_workers == 2
    assert settings.retries == 3
    assert settings.offline is False


def test_existing_local_model_does_not_download(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: pytest.fail("local model should not be downloaded"),
    )

    assert resolve_huggingface_model(str(model_dir)) == str(model_dir.resolve())


def test_remote_model_is_downloaded_with_configured_endpoint(monkeypatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    calls = []
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test/")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "90")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    monkeypatch.setenv("SIMPLETTS_HF_MAX_WORKERS", "2")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(snapshot_dir),
    )

    result = resolve_huggingface_model("owner/model")

    assert result == str(snapshot_dir)
    assert calls == [
        {
            "repo_id": "owner/model",
            "endpoint": "https://hub.example.test",
            "etag_timeout": 90,
            "max_workers": 2,
            "local_files_only": False,
        }
    ]


def test_single_model_file_download_only_requests_target(monkeypatch, tmp_path: Path) -> None:
    calls = []
    cached_file = tmp_path / "model.gguf"
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: calls.append(kwargs) or str(cached_file),
    )

    result = resolve_huggingface_file("owner/gguf", "model-Q4_K_M.gguf")

    assert result == str(cached_file)
    assert calls == [
        {
            "repo_id": "owner/gguf",
            "filename": "model-Q4_K_M.gguf",
            "endpoint": "https://hub.example.test",
            "etag_timeout": 60,
            "local_files_only": False,
        }
    ]


def test_offline_mode_only_uses_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs) or str(tmp_path / "cached"),
    )

    resolve_huggingface_model("owner/model")

    assert calls[0]["local_files_only"] is True


def test_transient_metadata_failure_retries_and_reuses_cache(monkeypatch, tmp_path: Path) -> None:
    calls = []
    sleeps = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise OSError("temporary mirror metadata failure")
        return str(tmp_path / "cached")

    monkeypatch.setenv("SIMPLETTS_HF_RETRIES", "3")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("simplettsworkflow.model_download.time.sleep", sleeps.append)

    result = resolve_huggingface_model("owner/model")

    assert result == str(tmp_path / "cached")
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_mirror_failure_falls_back_to_official_endpoint(monkeypatch, tmp_path: Path) -> None:
    calls = []
    sleeps = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs["endpoint"] == "https://hf-mirror.com":
            raise OSError("mirror unavailable")
        return str(tmp_path / "official-cache")

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("SIMPLETTS_HF_RETRIES", "2")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("simplettsworkflow.model_download.time.sleep", sleeps.append)

    result = resolve_huggingface_model("owner/model")

    assert result == str(tmp_path / "official-cache")
    assert [call["endpoint"] for call in calls] == [
        "https://hf-mirror.com",
        "https://hf-mirror.com",
        "https://huggingface.co",
    ]
    assert sleeps == [1]


def test_custom_endpoint_does_not_implicitly_fallback(monkeypatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
    monkeypatch.delenv("SIMPLETTS_HF_FALLBACK_ENDPOINT", raising=False)

    settings = configure_huggingface_downloads()

    assert settings.fallback_endpoint is None


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_timeout_is_rejected(monkeypatch, value: str) -> None:
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", value)

    with pytest.raises(ValueError, match="HF_HUB_ETAG_TIMEOUT"):
        configure_huggingface_downloads()
