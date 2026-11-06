"""Call only two hash-verified official modules; bypass evaluate.py/ResNet/checkpoint imports."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from .security import load_reviewed_module, verified_upstream

DEFAULT_SCORING = {"draws": 5, "sigma": .2, "dps_scale": .4, "seed": 42,
                   "superpixel_size": 20, "bins": 16, "hist_sigma": 1.0,
                   "spatial_sigma": 2.0, "top_fraction": .01}


def validate_scoring(config):
    if set(config) != set(DEFAULT_SCORING):
        raise ValueError("Unknown or missing inference settings.")
    limits = {"draws": (1, 64), "sigma": (1e-8, 5), "dps_scale": (0, 5),
              "seed": (0, 2**31-1), "superpixel_size": (4, 1024), "bins": (2, 32),
              "hist_sigma": (0, 3), "spatial_sigma": (0, 20), "top_fraction": (1e-8, 1)}
    for key, (lo, hi) in limits.items():
        value = config[key]
        if type(value) not in {int, float} or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"Invalid inference setting: {key}.")
    for key in ("draws", "seed", "superpixel_size", "bins"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer.")
    return config


class OfficialScorer:
    def __init__(self, upstream, config, device="cpu"):
        self.hashes = verified_upstream(upstream)
        self.superpixels = load_reviewed_module(upstream, "ood/superpixels.py")
        self.scoring = load_reviewed_module(upstream, "ood/scoring.py")
        self.config = validate_scoring(config)
        self.device = torch.device(device)

    def __call__(self, original, reconstructions):
        config = self.config
        if original.dtype != np.uint8 or reconstructions.dtype != np.uint8:
            raise ValueError("Scoring requires uint8 RGB inputs.")
        h, w, channels = original.shape
        if channels != 3 or reconstructions.shape != (config["draws"], h, w, 3):
            raise ValueError("Reconstruction shape/draw-count mismatch.")
        labels, _, parents = self.superpixels.recursive_subdivide(
            original, np.zeros((h, w), np.int32), target_size=config["superpixel_size"],
            compactness=12.0, alpha_grad=10.0)
        counts = np.bincount(labels.ravel())
        if int(counts.max()) * len(reconstructions) > 32767:
            raise ValueError("Official int16 histogram would overflow: reduce draws or superpixel_size.")
        if len(counts) * config["bins"]**3 > self.scoring.MAX_HIST_ELEMENTS:
            raise ValueError("Official dense histogram limit exceeded; reduce bins to avoid a silent scoring change.")
        result, _, _ = self.scoring.compute_delta_map_gpu(
            labels_fine=labels, images_recon_all=reconstructions,
            pca_feats_recon=np.zeros((len(reconstructions), h, w, 1), np.float32),
            label_image=original, label_pca_map=np.zeros((h, w, 1), np.float32),
            parent_map=parents, bins_rgb=config["bins"], bins_pca=2,
            smooth_sigma=config["hist_sigma"], min_pixels=1,
            device=self.device, gray_scale=False, use_pca=False)
        if not np.isfinite(result).all():
            raise FloatingPointError("Official scorer returned a non-finite map.")
        # Preserve the official signed NLL - entropy; no abs or per-image score normalization.
        return gaussian_filter(result, config["spatial_sigma"]).astype(np.float32)


def image_score(score_map, fraction):
    flat = score_map.ravel()
    count = max(1, int(np.ceil(len(flat) * fraction)))
    return float(np.partition(flat, len(flat) - count)[-count:].mean())
