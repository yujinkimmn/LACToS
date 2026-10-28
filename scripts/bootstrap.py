"""Fetch the exact official implementation without redistributing it."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def bootstrap():
    lock = json.loads((ROOT / "upstream.json").read_text())
    target = ROOT / lock["directory"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", lock["repository"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", lock["commit"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != lock["commit"]:
        raise RuntimeError(f"Wrong upstream commit: {actual}; expected {lock['commit']}")
    if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Upstream checkout must be unmodified")
    print(f"Verified official LACToS: {actual}")
    return target


if __name__ == "__main__":
    bootstrap()
