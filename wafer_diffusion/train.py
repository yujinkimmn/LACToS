"""Normal-only training with EMA, deterministic minibatches, held-out denoising validation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import load_optimizer, load_weights, read_checkpoint, save_checkpoint
from .data import audit_manifest, manifest_fingerprint, prepare_image
from .diffusion import Diffusion
from .model import build_model
from .security import private_output, write_json

DEFAULT_CONFIG = {
    "image_size": 256, "crop_border": 0, "base_channels": 32,
    "channel_mult": [1, 2, 4, 4], "attention": True,
    "diffusion_steps": 1000, "schedule": "linear", "batch_size": 8,
    "learning_rate": 0.0002, "ema_decay": .999, "max_steps": 50000,
    "validate_every": 500, "val_images": 64, "workers": 4, "seed": 42,
}


def validate_config(config):
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Unknown or missing training config keys.")
    integer_bounds = {"image_size": (16, 512), "crop_border": (0, 10000), "base_channels": (8, 64),
                      "diffusion_steps": (8, 4000), "batch_size": (1, 128), "max_steps": (1, 10_000_000),
                      "validate_every": (1, 100000), "val_images": (1, 10000), "workers": (1, 32), "seed": (0, 2**31-1)}
    for key, (lo, hi) in integer_bounds.items():
        if type(config[key]) is not int or not lo <= config[key] <= hi:
            raise ValueError(f"Invalid training setting: {key}.")
    mult = config["channel_mult"]
    if not isinstance(mult, list) or not 2 <= len(mult) <= 5 or any(type(m) is not int or not 1 <= m <= 4 for m in mult):
        raise ValueError("Invalid channel multipliers.")
    if config["base_channels"] % 2 or config["image_size"] % (2 ** (len(mult)-1)):
        raise ValueError("Use even base_channels and image_size divisible by the U-Net downsampling factor.")
    if type(config["attention"]) is not bool:
        raise ValueError("attention must be boolean.")
    if config["attention"] and config["image_size"] // 2**(len(mult)-1) > 32:
        raise ValueError("Attention bottleneck is limited to 32x32 to bound memory.")
    if config["schedule"] not in {"linear", "cosine"}:
        raise ValueError("Invalid diffusion schedule.")
    for key, lo, hi in [("learning_rate", 0, .01), ("ema_decay", 0, 1)]:
        if type(config[key]) not in {int, float} or not lo < config[key] < hi:
            raise ValueError(f"Invalid {key}.")
    return config


def get_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - set(config):
            raise ValueError("Config must be a JSON object containing known keys only.")
        config.update(data)
    return validate_config(config)


def select_device(device):
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda or auto.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; install the approved matching GPU wheel.")
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)


def image_tensor(row, config):
    data = prepare_image(row["path"], config["image_size"], config["crop_border"])
    return torch.from_numpy(data).permute(2, 0, 1).float() / 127.5 - 1


def validation_loss(model, rows, config, diffusion, device):
    rng = torch.Generator().manual_seed(config["seed"] + 99)
    losses = []
    model.eval()
    with torch.no_grad():
        for row in rows[:config["val_images"]]:
            clean = image_tensor(row, config).unsqueeze(0).to(device)
            t = torch.randint(diffusion.steps, (1,), generator=rng).to(device)
            noise = torch.randn(clean.shape, generator=rng).to(device)
            losses.append(float(F.mse_loss(model(diffusion.q_sample(clean, t, noise), t), noise)))
    return float(np.mean(losses))


def train(manifest, output, config, device="auto", resume=None):
    validate_config(config)
    rows = audit_manifest(manifest)
    fingerprint = manifest_fingerprint(rows)
    train_rows = [r for r in rows if r["split"] == "train" and int(r["label"]) == 0]
    val_rows = [r for r in rows if r["split"] == "val" and int(r["label"]) == 0]
    output = private_output(output)
    if (output / "last.npz").exists() and not resume:
        raise FileExistsError("Training output exists; use --resume or a new output directory.")
    device = select_device(device)
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=.01)
    diffusion = Diffusion(config["diffusion_steps"], config["schedule"], device)
    rng = torch.Generator().manual_seed(config["seed"])
    start, best = 0, None
    if resume:
        meta, arrays = read_checkpoint(resume)
        if meta["manifest_sha256"] != fingerprint:
            raise ValueError("Resume dataset/split differs from the checkpoint.")
        allowed_changes = {"max_steps", "validate_every", "workers", "val_images"}
        if any(config[k] != meta["config"][k] for k in config if k not in allowed_changes):
            raise ValueError("Resume config differs from the checkpoint.")
        load_weights(model, arrays, "model")
        load_weights(ema, arrays, "ema")
        load_optimizer(optimizer, arrays)
        rng.set_state(torch.from_numpy(arrays["rng"].copy()))
        start, best = meta["step"], meta["best_val_loss"]
    if start >= config["max_steps"]:
        raise ValueError("max_steps must exceed the checkpoint's completed step.")
    write_json(output / "config.json", config)
    write_json(output / "environment.json", {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)})
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for step in range(start + 1, config["max_steps"] + 1):
            indices = torch.randint(len(train_rows), (config["batch_size"],), generator=rng).tolist()
            clean = torch.stack(list(pool.map(lambda i: image_tensor(train_rows[i], config), indices))).to(device)
            time_index = torch.randint(diffusion.steps, (len(indices),), generator=rng).to(device)
            noise = torch.randn(clean.shape, generator=rng).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(diffusion.q_sample(clean, time_index, noise), time_index), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - config["ema_decay"])
            if step == start + 1 or step % 100 == 0:
                elapsed = time.monotonic() - t0
                eta = elapsed / (step - start) * (config["max_steps"] - step)
                print(f"train {step}/{config['max_steps']} loss={loss.item():.6f} eta_s={eta:.0f}", flush=True)
            if step % config["validate_every"] == 0 or step == config["max_steps"]:
                val = validation_loss(ema, val_rows, config, diffusion, device)
                improved = best is None or val < best
                if improved:
                    best = val
                meta = {"format": "wafer-ddpm-npz-v1", "config": config, "step": step,
                        "manifest_sha256": fingerprint, "best_val_loss": best,
                        "train_image_hashes": [r["pixel_sha256"] for r in train_rows],
                        "train_groups": sorted({r["group"] for r in train_rows})}
                save_checkpoint(output / "last.npz", model, ema, optimizer, meta, rng)
                if improved:
                    save_checkpoint(output / "best.npz", model, ema, optimizer, meta, rng)
                with (output / "loss.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "train_loss": loss.item(), "val_loss": val}) + "\n")
                print(f"validation step={step} denoising_mse={val:.6f}", flush=True)
    return output / "last.npz"
