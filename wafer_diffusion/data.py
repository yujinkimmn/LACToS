"""Explicit local image roots, group holdout, decoded-image duplicate detection."""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps

from .security import local_path, write_json

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDS = ["id", "path", "group", "split", "label", "pixel_sha256"]
Image.MAX_IMAGE_PIXELS = 50_000_000


def read_rgb(path: str | Path) -> np.ndarray:
    path = local_path(path)
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError("Supported images: PNG, JPEG, BMP, TIFF. Convert DAT explicitly before use.")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Image file exceeds the 256 MiB limit.")
    # Match file content against an explicit decoder allowlist, not just its extension.
    # In particular, a renamed EPS must never reach Pillow's Ghostscript subprocess path.
    with Image.open(path, formats=["PNG", "JPEG", "BMP", "TIFF"]) as image:
        if image.width * image.height > 50_000_000:
            raise ValueError("Image exceeds the pixel limit.")
        if image.mode.startswith(("I", "F")):
            raise ValueError("High-bit-depth images need an explicit, fixed conversion to 8-bit RGB.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are not accepted.")
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8).copy()


def pixel_digest(image: np.ndarray) -> str:
    h = hashlib.sha256(str(image.shape).encode())
    h.update(image.tobytes())
    return h.hexdigest()


def prepare_image(path: str | Path, size: int, crop_border: int = 0) -> np.ndarray:
    image = read_rgb(path)
    if crop_border < 0 or 2 * crop_border >= min(image.shape[:2]):
        raise ValueError("Invalid crop_border.")
    if crop_border:
        image = image[crop_border:-crop_border, crop_border:-crop_border]
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)).copy()


def scan_roots(roots, pattern="*"):
    found = {}
    for value in roots:
        root = local_path(value)
        if root.is_file():
            candidates = [(root.parent, root)]
        elif root.is_dir():
            candidates = []
            for folder, dirs, files in os.walk(root, followlinks=False):
                for d in dirs:
                    if (Path(folder) / d).is_symlink():
                        raise ValueError("Symlink directories are rejected; supply their resolved roots explicitly.")
                candidates.extend((root, Path(folder) / f) for f in sorted(files))
        else:
            raise FileNotFoundError("An explicitly selected image root does not exist.")
        for base, path in candidates:
            if path.suffix.lower() not in EXTENSIONS or not fnmatch.fnmatch(path.name, pattern):
                continue
            if path.is_symlink():
                raise ValueError("Symlink images are rejected; provide resolved image files explicitly.")
            found[str(path.resolve())] = (base, path.resolve())
    if not found:
        raise ValueError("No images matched the selected roots and filename pattern.")
    return [found[k] for k in sorted(found)]


def make_manifest(normal_roots, anomaly_roots, output: Path, group_regex=None,
                  group_parent=False, pattern="*", val_fraction=.2, test_fraction=.2, seed=42):
    if (group_regex is None) == (not group_parent):
        raise ValueError("Choose exactly one: --group-regex or --group-parent.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Holdout fractions must be positive and sum to less than 1.")
    regex = re.compile(group_regex) if group_regex else None
    if regex and "group" not in regex.groupindex:
        raise ValueError("--group-regex must contain (?P<group>...). Include lot identity when wafer IDs repeat.")
    rows, seen = [], {}
    for label, roots in [(0, normal_roots), (1, anomaly_roots)]:
        if not roots:
            continue
        for root, path in scan_roots(roots, pattern):
            relative = path.relative_to(root).as_posix()
            if regex:
                match = regex.search(relative)
                if match is None:
                    raise ValueError("A selected image did not match --group-regex; no fallback split is allowed.")
                group = match.group("group")
            else:
                group = path.parent.name
            if not group:
                raise ValueError("Group identity cannot be empty.")
            digest = pixel_digest(read_rgb(path))
            if digest in seen:
                old = seen[digest]
                if old["group"] != group or int(old["label"]) != label:
                    raise ValueError("Duplicate decoded image spans different groups or labels; curate it first.")
                continue
            row = dict(id=digest[:20], path=str(path), group=group, split="", label=label, pixel_sha256=digest)
            seen[digest] = row
            rows.append(row)
    groups = sorted({r["group"] for r in rows if r["label"] == 0})
    if len(groups) < 3:
        raise ValueError("At least 3 normal wafer/lot groups are required for train/val/test holdout.")
    np.random.default_rng(seed).shuffle(groups)
    nv = max(1, round(len(groups) * val_fraction))
    nt = max(1, round(len(groups) * test_fraction))
    if nv + nt >= len(groups):
        raise ValueError("Too few groups for these holdout fractions.")
    assignment = {g: "val" if i < nv else "test" if i < nv + nt else "train" for i, g in enumerate(groups)}
    for row in rows:
        row["split"] = assignment.get(row["group"], "test")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Manifest already exists. Use a new work directory to preserve the split.")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit_manifest(output)
    write_json(output.with_suffix(".summary.json"), {
        "seed": seed, "group_regex": group_regex, "group_parent": group_parent,
        "unused_train_anomalies": sum(r["split"] == "train" and r["label"] == 1 for r in rows),
        "counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "groups": {s: sorted({r["group"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")},
    })
    return rows


def audit_manifest(path: str | Path, verify_images=True):
    with local_path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Manifest is missing required columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest.")
    groups, digests, ids = {}, {}, set()
    for row in rows:
        if row["split"] not in {"train", "val", "test"} or str(row["label"]) not in {"0", "1"}:
            raise ValueError("Invalid split or image label.")
        if not re.fullmatch(r"[a-f0-9]{20}", row["id"]) or row["id"] in ids:
            raise ValueError("Invalid or duplicate image ID.")
        ids.add(row["id"])
        if not row["group"]:
            raise ValueError("Images must have explicit groups.")
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("A wafer/lot group crosses dataset splits.")
        groups[row["group"]] = row["split"]
        if not re.fullmatch(r"[a-f0-9]{64}", row["pixel_sha256"]):
            raise ValueError("Invalid image fingerprint.")
        digest = pixel_digest(read_rgb(row["path"])) if verify_images else row["pixel_sha256"]
        if digest != row["pixel_sha256"] or digest[:20] != row["id"]:
            raise ValueError("An image changed after manifest creation.")
        if digest in digests:
            raise ValueError("Duplicate image in manifest.")
        digests[digest] = row["split"]
    for split in ("train", "val", "test"):
        if not any(r["split"] == split and int(r["label"]) == 0 for r in rows):
            raise ValueError(f"No normal images in {split}.")
    return rows


def manifest_fingerprint(rows):
    content = sorted((r["pixel_sha256"], r["group"], r["split"], int(r["label"])) for r in rows)
    return hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()
