"""Tensor-only NPZ checkpoints. No torch.load, pickle, external weights or dynamic classes."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

MAX_BYTES = 2 * 1024**3


def save_checkpoint(path, model, ema, optimizer, metadata, rng):
    path = Path(path)
    arrays = {"metadata": np.asarray(json.dumps(metadata, allow_nan=False))}
    for namespace, state in [("model", model.state_dict()), ("ema", ema.state_dict())]:
        for name, tensor in state.items():
            arrays[f"{namespace}/{name}"] = tensor.detach().cpu().numpy()
    opt = optimizer.state_dict()
    arrays["optimizer_groups"] = np.asarray(json.dumps(opt["param_groups"]))
    for index, state in opt["state"].items():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("Unsupported optimizer state type.")
            arrays[f"optimizer/{index}/{key}"] = value.detach().cpu().numpy()
    arrays["rng"] = rng.get_state().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
        np.savez(f, **arrays)
    temp.replace(path)


def read_checkpoint(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Checkpoint exceeds the 2 GiB limit.")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 20000 or len({i.filename for i in entries}) != len(entries):
            raise ValueError("Invalid checkpoint archive members.")
        if sum(i.file_size for i in entries) > MAX_BYTES:
            raise ValueError("Expanded checkpoint exceeds the size limit.")
        if any(i.compress_type != zipfile.ZIP_STORED for i in entries):
            raise ValueError("Only uncompressed tensor archives produced by this pipeline are accepted.")
        # Validate NPY headers before numpy can allocate memory from a forged shape.
        for entry in entries:
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported tensor header version.")
                if dtype.hasobject:
                    raise ValueError("Object arrays are forbidden in checkpoints.")
                size = math.prod(shape) * dtype.itemsize
                if len(shape) > 8 or size > MAX_BYTES or size != entry.file_size - stream.tell():
                    raise ValueError("Invalid tensor shape/size in checkpoint header.")
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("Missing checkpoint metadata.")
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("format") != "wafer-ddpm-npz-v1":
            raise ValueError("Unsupported checkpoint format; pickle/PT files are not accepted.")
        arrays = {}
        for key in archive.files:
            value = archive[key]
            if key not in {"metadata", "optimizer_groups"} and value.dtype.kind not in "biuf":
                raise ValueError("Non-numeric tensor in checkpoint.")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError("Non-finite checkpoint tensor.")
            arrays[key] = value
    return metadata, arrays


def load_weights(model, arrays, namespace="ema"):
    expected = model.state_dict()
    actual = {k[len(namespace) + 1:]: v for k, v in arrays.items() if k.startswith(namespace + "/")}
    if expected.keys() != actual.keys():
        raise ValueError("Checkpoint architecture does not match.")
    state = {}
    for key, reference in expected.items():
        value = actual[key]
        if tuple(value.shape) != tuple(reference.shape) or value.dtype != reference.detach().cpu().numpy().dtype:
            raise ValueError("Checkpoint tensor shape or dtype mismatch.")
        state[key] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)


def load_optimizer(optimizer, arrays):
    groups = json.loads(str(arrays["optimizer_groups"]))
    state = {}
    for key, value in arrays.items():
        if key.startswith("optimizer/"):
            _, index, name = key.split("/")
            if name not in {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Unsupported optimizer tensor.")
            state.setdefault(int(index), {})[name] = torch.from_numpy(value.copy())
    optimizer.load_state_dict({"state": state, "param_groups": groups})
