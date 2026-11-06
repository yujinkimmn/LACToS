"""DPS reconstructions, official scores, validation-only calibration, untouched test metrics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .checkpoint import load_weights, read_checkpoint
from .data import audit_manifest, manifest_fingerprint, pixel_digest, prepare_image, read_rgb, scan_roots
from .diffusion import Diffusion
from .model import build_model
from .scoring import OfficialScorer, image_score, validate_scoring
from .security import private_output, sha256_file, write_json
from .train import select_device, validate_config


def settings_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prepare_run_output(output, engine):
    output = private_output(output)
    metadata_path = output / "inference.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        current = engine.provenance()
        if previous["checkpoint_sha256"] != current["checkpoint_sha256"] or previous["settings_sha256"] != current["settings_sha256"]:
            raise ValueError("Output belongs to another model/settings; use a new directory.")
    elif any(output.iterdir()):
        raise FileExistsError("Output directory is not empty and has no matching inference provenance.")
    write_json(metadata_path, engine.provenance())
    return output


class Inference:
    def __init__(self, checkpoint, upstream, config, device="auto"):
        self.settings = validate_scoring(config)
        self.metadata, arrays = read_checkpoint(checkpoint)
        self.config = validate_config(self.metadata["config"])
        self.checkpoint_hash = sha256_file(Path(checkpoint))
        self.device = select_device(device)
        self.model = build_model(self.config).to(self.device).eval().requires_grad_(False)
        load_weights(self.model, arrays, "ema")
        del arrays
        self.diffusion = Diffusion(self.config["diffusion_steps"], self.config["schedule"], self.device)
        self.scorer = OfficialScorer(upstream, self.settings, self.device)
        self.training_hashes = set(self.metadata["train_image_hashes"])

    def run(self, path):
        original_hash = pixel_digest(read_rgb(path))
        if original_hash in self.training_hashes:
            raise ValueError("Inference input duplicates a training image; use held-out images.")
        original = prepare_image(path, self.config["image_size"], self.config["crop_border"])
        clean = (torch.from_numpy(original).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(self.device)
        seed = (self.settings["seed"] + int(original_hash[:12], 16)) % (2**63 - 1)
        t0 = time.monotonic()
        recons = self.diffusion.reconstruct(
            self.model, clean, self.settings["draws"], self.settings["sigma"], self.settings["dps_scale"], seed,
            progress=lambda n, total: print(f"reconstruction {original_hash[:10]} {n}/{total}", flush=True))
        recons = ((recons.permute(0, 2, 3, 1).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        score_map = self.scorer(original, recons)
        return {"id": original_hash[:20], "pixel_sha256": original_hash, "original": original,
                "reconstructions": recons, "score_map": score_map,
                "image_score": image_score(score_map, self.settings["top_fraction"]),
                "seconds": time.monotonic() - t0}

    def provenance(self):
        return {"checkpoint_sha256": self.checkpoint_hash, "settings": self.settings,
                "settings_sha256": settings_digest(self.settings), "train_step": self.metadata["step"],
                "preprocess": {k: self.config[k] for k in ("image_size", "crop_border")},
                "model": "independent-wafer-unet-ddpm", "sampler": "full-chain-dps",
                "scoring": "official-lactos-intensity-only-signed", "upstream_hashes": self.scorer.hashes}


def save_result(result, directory, save_recon=False, threshold=None):
    directory.mkdir(parents=True, exist_ok=True)
    score = result["score_map"]
    np.save(directory / "score_map.npy", score, allow_pickle=False)
    np.save(directory / "recon_mean.npy", result["reconstructions"].mean(axis=0).astype(np.float32), allow_pickle=False)
    Image.fromarray(result["original"]).save(directory / "input.png")
    Image.fromarray(result["reconstructions"].mean(axis=0).round().astype(np.uint8)).save(directory / "recon_mean.png")
    lo, hi = np.percentile(score, [1, 99])
    color = np.clip((score - lo) / max(float(hi - lo), 1e-8), 0, 1)
    heat = np.stack([color, .3 * (1 - color), 1 - color], axis=-1) * 255
    Image.fromarray(heat.astype(np.uint8)).save(directory / "heatmap.png")
    Image.fromarray((.55 * result["original"] + .45 * heat).astype(np.uint8)).save(directory / "overlay.png")
    if save_recon:
        for i, image in enumerate(result["reconstructions"]):
            Image.fromarray(image).save(directory / f"recon_{i:03d}.png")
    summary = {k: result[k] for k in ("id", "pixel_sha256", "image_score", "seconds")}
    summary["display_range"] = [float(lo), float(hi)]
    summary["display_note"] = "Heatmap colors are normalized for display only; decisions use raw signed scores."
    if threshold is not None:
        summary.update(threshold=threshold, predicted_anomaly=result["image_score"] > threshold)
    write_json(directory / "result.json", summary)
    return summary


def calibration_threshold(calibration, engine):
    data = json.loads(Path(calibration).read_text(encoding="utf-8"))
    if data.get("checkpoint_sha256") != engine.checkpoint_hash or data.get("settings_sha256") != settings_digest(engine.settings):
        raise ValueError("Calibration belongs to a different checkpoint or inference configuration. Recalibrate.")
    threshold = data.get("threshold")
    if type(threshold) not in {float, int} or not np.isfinite(threshold):
        raise ValueError("Invalid calibration threshold.")
    return data


def checked_holdout(manifest, engine):
    rows = audit_manifest(manifest)
    if manifest_fingerprint(rows) != engine.metadata["manifest_sha256"]:
        raise ValueError("Manifest differs from training; create a separate experiment for a changed dataset.")
    return rows


def calibrate(engine, manifest, output, target_fpr=.01):
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    rows = checked_holdout(manifest, engine)
    rows = [r for r in rows if r["split"] == "val" and int(r["label"]) == 0]
    output = private_output(output)
    values = np.array([engine.run(r["path"])["image_score"] for r in rows])
    threshold = float(np.quantile(values, 1-target_fpr, method="higher"))
    data = {**engine.provenance(), "threshold": threshold, "target_fpr": target_fpr,
            "manifest_sha256": engine.metadata["manifest_sha256"], "normal_val_count": len(rows),
            "calibration_image_hashes": [r["pixel_sha256"] for r in rows],
            "observed_val_fpr": float((values > threshold).mean()),
            "normal_val_scores": values.tolist(),
            "method": "empirical higher quantile of normal validation image scores",
            "limitation": "Not a guaranteed future FPR; small validation sets cannot resolve low false-alarm rates."}
    write_json(output / "calibration.json", data)
    return data


def infer(engine, inputs, output, pattern="*", calibration=None, save_recon=False):
    output = prepare_run_output(output, engine)
    threshold = calibration_threshold(calibration, engine)["threshold"] if calibration else None
    summaries = []
    for _, path in scan_roots(inputs, pattern):
        result = engine.run(path)
        summary = save_result(result, output / result["id"], save_recon, threshold)
        summary["path"] = str(path)
        summaries.append(summary)
    write_json(output / "summary.json", summaries)
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["id", "path", "image_score", "seconds"] + (["threshold", "predicted_anomaly"] if calibration else [])
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def binary_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    predictions = scores > threshold
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    auroc = ap = None
    if np.any(labels == 0) and np.any(labels == 1):
        # Group equal scores so ties receive half credit in AUROC and grouped AP.
        order = np.argsort(-scores, kind="stable")
        y, s = labels[order], scores[order]
        ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
        tps = np.cumsum(y)[ends]
        fps = 1 + ends - tps
        tpr, fpr = np.r_[0, tps/y.sum()], np.r_[0, fps/(len(y)-y.sum())]
        auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        ap = float(np.sum(np.diff(tpr) * (tps / (tps + fps))))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "normal_false_alarm_rate": fp/(fp+tn) if fp+tn else None,
            "anomaly_miss_rate": fn/(fn+tp) if fn+tp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "precision": tp/(tp+fp) if tp+fp else None,
            "image_auroc": auroc, "image_ap": ap}


def evaluate(engine, manifest, calibration, output, save_recon=False):
    rows = checked_holdout(manifest, engine)
    calibration_data = calibration_threshold(calibration, engine)
    if calibration_data["manifest_sha256"] != engine.metadata["manifest_sha256"]:
        raise ValueError("Calibration split differs from training.")
    rows = [r for r in rows if r["split"] == "test"]
    output = prepare_run_output(output, engine)
    labels, scores, entries = [], [], []
    for row in rows:
        result = engine.run(row["path"])
        if result["pixel_sha256"] in calibration_data["calibration_image_hashes"]:
            raise ValueError("Test image was used for calibration.")
        save_result(result, output / result["id"], save_recon, calibration_data["threshold"])
        labels.append(int(row["label"]))
        scores.append(result["image_score"])
        entries.append({"id": row["id"], "group": row["group"], "label": int(row["label"]), "score": result["image_score"]})
    metrics = {**engine.provenance(), **binary_metrics(labels, scores, calibration_data["threshold"]),
               "threshold": calibration_data["threshold"], "test_count": len(rows), "images": entries,
               "metric_scope": "image-level; no pixel AUROC/AP/AUPRO without genuine annotation masks"}
    write_json(output / "metrics.json", metrics)
    return metrics
