"""Synthetic-only real training/reconstruction/scoring checks, never a wafer quality benchmark."""
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from .checkpoint import read_checkpoint
from .data import make_manifest
from .inference import Inference, calibrate, evaluate, infer
from .scoring import DEFAULT_SCORING
from .security import write_json
from .train import DEFAULT_CONFIG, train


def make_synthetic(root):
    rng = np.random.default_rng(1234)
    for group in range(8):
        folder = root / "normal" / f"synthetic_lot_{group:02d}"
        folder.mkdir(parents=True)
        for index in range(2):
            image = np.clip(100 + group * 3 + rng.normal(0, 2, (16, 16, 3)), 0, 255).astype(np.uint8)
            image[::4, :, :] += 15
            Image.fromarray(image).save(folder / f"{index}_preproc.png")
    folder = root / "anomaly" / "synthetic_heldout_anomaly"
    folder.mkdir(parents=True)
    for index in range(2):
        image = np.clip(110 + rng.normal(0, 2, (16, 16, 3)), 0, 255).astype(np.uint8)
        image[4:8, 4+index:8+index] = [240, 30, 30]
        Image.fromarray(image).save(folder / f"{index}_preproc.png")


def run_smoke(upstream, output):
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="wafer-synthetic-") as scratch:
        scratch = Path(scratch)
        make_synthetic(scratch)
        manifest = scratch / "manifest.csv"
        rows = make_manifest([scratch / "normal"], [scratch / "anomaly"], manifest, group_parent=True)
        config = {**DEFAULT_CONFIG, "image_size": 16, "base_channels": 8, "channel_mult": [1, 2],
                  "attention": False, "diffusion_steps": 8, "schedule": "cosine",
                  "batch_size": 2, "max_steps": 3, "validate_every": 3, "workers": 1, "val_images": 2}
        checkpoint = train(manifest, scratch / "model", config, "cpu")
        before, arrays = read_checkpoint(checkpoint)
        first_weights = arrays["model/input.weight"].copy()
        config["max_steps"] = 4
        checkpoint = train(manifest, scratch / "model", config, "cpu", checkpoint)
        after, arrays = read_checkpoint(checkpoint)
        if after["step"] != 4 or np.array_equal(first_weights, arrays["model/input.weight"]):
            raise AssertionError("Resume did not update model weights.")
        scoring = {**DEFAULT_SCORING, "draws": 2, "superpixel_size": 4, "bins": 4, "hist_sigma": 0.0, "spatial_sigma": .5}
        engine = Inference(checkpoint, upstream, scoring, "cpu")
        test_image = next(r["path"] for r in rows if r["split"] == "test")
        first, repeated = engine.run(test_image), engine.run(test_image)
        if not np.array_equal(first["reconstructions"], repeated["reconstructions"]):
            raise AssertionError("Identical-seed sampling is not reproducible on CPU.")
        if np.array_equal(first["reconstructions"][0], first["reconstructions"][1]):
            raise AssertionError("Independent reconstruction draws are identical.")
        calibrate(engine, manifest, scratch / "calibration", target_fpr=.1)
        metrics = evaluate(engine, manifest, scratch / "calibration/calibration.json", scratch / "test")
        infer(engine, [test_image], output / "synthetic_example", save_recon=True)
        report = {"scope": "synthetic functional smoke test, not wafer/paper performance",
                  "train_steps": after["step"], "resume_updates_weights": True,
                  "same_seed_reproducible": True, "distinct_draws": True,
                  "official_scoring_finite": bool(np.isfinite(first["score_map"]).all()),
                  "heldout_test_images": metrics["test_count"], "upstream_hashes": engine.scorer.hashes,
                  "torch": torch.__version__, "numpy": np.__version__}
        write_json(output / "smoke_report.json", report)
        print("PASS: synthetic training, resume, DPS, official scoring, calibration, evaluation", flush=True)
        return report
