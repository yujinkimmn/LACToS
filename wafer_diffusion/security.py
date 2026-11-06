"""Fail-closed helpers. The Python audit hook is defense in depth, not an OS sandbox."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "d1ebf01dd268629bed18f3e3e9ae9c22c666f117"
UPSTREAM_HASHES = {
    "ood/scoring.py": "c962c7078091eb569a912bcdf4f1e6648823a674c92a9af24c8ef241c4f85f34",
    "ood/superpixels.py": "bd48e73d2d5e053f7768041f08617654725f4b09ab565696f559dcf5ec1b1adf",
}
_guard_enabled = False


def enable_offline_guard():
    global _guard_enabled
    if _guard_enabled:
        return
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["WANDB_MODE"] = "disabled"

    def guard(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
            raise PermissionError("Wafer runtime blocks network operations; prepare dependencies separately.")
        if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"}:
            raise PermissionError("Wafer runtime blocks external process execution.")
    sys.addaudithook(guard)
    _guard_enabled = True


def forbid_hosted_data():
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        raise PermissionError("Wafer data commands are disabled on hosted CI. CI runs synthetic tests only.")


def local_path(value: str | Path) -> Path:
    value = str(value)
    if "://" in value or value.startswith(("\\\\", "//")):
        raise ValueError("Use a local filesystem path; URLs and UNC paths are not accepted.")
    return Path(value).expanduser().resolve()


def private_output(value: str | Path) -> Path:
    path = local_path(value)
    if path == PROJECT_ROOT or PROJECT_ROOT in path.parents or any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Data, checkpoints and results must be outside the code repository.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False) as f:
        temp = Path(f.name)
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verified_upstream(root: str | Path) -> dict:
    root = local_path(root)
    report = {}
    for rel, expected in UPSTREAM_HASHES.items():
        file = root / rel
        if file.is_symlink() or root not in file.resolve().parents:
            raise ValueError("Upstream module path escapes the selected directory.")
        data = file.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream source hash mismatch: {rel}. Expected reviewed commit {UPSTREAM_COMMIT}.")
        report[rel] = actual
    return report


def load_reviewed_module(root: str | Path, rel: str):
    """Execute only the exact reviewed bytes; no package __init__ or arbitrary module names."""
    if rel not in UPSTREAM_HASHES:
        raise ValueError("Module is not in the source allowlist.")
    file = local_path(root) / rel
    verified_upstream(root)
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_HASHES[rel]:
        raise ValueError("Upstream changed between verification and reading.")
    module = types.ModuleType("_reviewed_" + file.stem)
    module.__file__ = str(file)
    # Intentional execution of SHA-256 allowlisted source; never a user-supplied expression.
    exec(compile(data, str(file), "exec"), module.__dict__)
    return module
