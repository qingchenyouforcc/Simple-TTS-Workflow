"""Restrict candidate cleanup to audio issued by this server instance."""

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from threading import Lock


class CandidateStore:
    def __init__(self, root: Path, secret_path: Path | None = None):
        self.root = root.resolve()
        self._secret = secrets.token_bytes(32)
        if secret_path is not None:
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with secret_path.open("xb") as stream:
                    stream.write(self._secret)
            except FileExistsError:
                self._secret = secret_path.read_bytes()
            if len(self._secret) != 32:
                raise ValueError("候选版本签名密钥损坏，请恢复本地状态文件。")
        self._lock = Lock()

    def token(self, path: str) -> str:
        relative = Path(path).resolve().relative_to(self.root).as_posix()
        signature = hmac.new(self._secret, relative.encode(), hashlib.sha256).hexdigest()
        return signature + ":" + relative

    def resolve(self, token: str) -> Path:
        signature, separator, relative = token.partition(":")
        expected = hmac.new(self._secret, relative.encode(), hashlib.sha256).hexdigest()
        if not separator or not hmac.compare_digest(signature, expected):
            raise ValueError("候选版本已失效，请重新生成后再筛选。")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path.suffix != ".wav":
            raise ValueError("无效的候选音频路径。")
        return path

    def retain(self, groups: list[dict]) -> int:
        with self._lock:
            return self._retain(groups)

    def _retain(self, groups: list[dict]) -> int:
        keep = set()
        discard = set()
        for group in groups:
            keep.add(self.resolve(group["selected"]))
            discard.update(self.resolve(token) for token in group["discard"])
        if keep & discard:
            raise ValueError("选中的版本不能同时被删除。")
        if any(not path.is_file() for path in keep):
            raise ValueError("选中的音频文件不存在，未清理任何候选版本。")

        # Validate every path and metadata document before deleting any audio.
        updates = {}
        for directory in {path.parent for path in discard}:
            metadata_path = directory / "metadata.json"
            if metadata_path.is_symlink():
                raise ValueError("无效的音频元数据路径。")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                removed_names = {path.name for path in discard if path.parent == directory}
                removed_indices = {item["index"] for item in metadata["items"] if item["filename"] in removed_names}
                metadata["items"] = [item for item in metadata["items"] if item["filename"] not in removed_names]
                if "emotion_analyses" in metadata:
                    metadata["emotion_analyses"] = [item for item in metadata["emotion_analyses"] if item["index"] not in removed_indices]
                updates[metadata_path] = metadata
        for path in discard:
            path.unlink(missing_ok=True)
        for path, metadata in updates.items():
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        return len(discard)
