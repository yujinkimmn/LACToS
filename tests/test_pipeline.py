import csv
import json
import math
import io
import shutil
import socket
import subprocess
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from wafer_diffusion.checkpoint import read_checkpoint
from wafer_diffusion.data import audit_manifest, make_manifest, prepare_image, read_rgb
from wafer_diffusion.diffusion import Diffusion
from wafer_diffusion.inference import binary_metrics, calibration_threshold
from wafer_diffusion.model import build_model
from wafer_diffusion.scoring import DEFAULT_SCORING, OfficialScorer
from wafer_diffusion.security import forbid_hosted_data, load_reviewed_module, private_output, verified_upstream
from wafer_diffusion.smoke import make_synthetic, run_smoke
from wafer_diffusion.train import DEFAULT_CONFIG, train


def test_end_to_end(tmp_path, upstream):
    report = run_smoke(upstream, tmp_path / "results")
    assert report["train_steps"] == 4
    assert report["official_scoring_finite"]
    assert report["same_seed_reproducible"]


def test_group_holdout_and_image_mutation(tmp_path):
    make_synthetic(tmp_path)
    manifest = tmp_path / "manifest.csv"
    rows = make_manifest([tmp_path / "normal"], [], manifest, group_parent=True)
    groups = {s: {r["group"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(rows[0]["path"])
    with pytest.raises(ValueError, match="changed"):
        audit_manifest(manifest)


def test_cross_group_duplicate_rejected(tmp_path):
    make_synthetic(tmp_path)
    image = tmp_path / "normal/synthetic_lot_00/0_preproc.png"
    shutil.copyfile(image, tmp_path / "normal/synthetic_lot_01/copied.png")
    with pytest.raises(ValueError, match="Duplicate decoded"):
        make_manifest([tmp_path / "normal"], [], tmp_path / "m.csv", group_parent=True)


def test_regex_needs_lot_and_wafer_capture(tmp_path):
    make_synthetic(tmp_path)
    with pytest.raises(ValueError, match="named|contain"):
        make_manifest([tmp_path / "normal"], [], tmp_path / "m.csv", group_regex="synthetic_lot")
    rows = make_manifest([tmp_path / "normal"], [], tmp_path / "m.csv", group_regex=r"(?P<group>synthetic_lot_\d+)")
    assert len(rows) == 16


def test_no_pickle_checkpoint(tmp_path):
    file = tmp_path / "bad.npz"
    np.savez(file, metadata=np.asarray(json.dumps({"format": "wafer-ddpm-npz-v1"})), evil=np.array({"x": 1}, dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        read_checkpoint(file)


def test_compressed_archive_rejected(tmp_path):
    file = tmp_path / "bad.npz"
    np.savez_compressed(file, metadata=np.asarray(json.dumps({"format": "wafer-ddpm-npz-v1"})), weights=np.zeros(1000))
    with pytest.raises(ValueError, match="uncompressed"):
        read_checkpoint(file)


def test_checkpoint_nonfinite_rejected(tmp_path):
    file = tmp_path / "bad.npz"
    np.savez(file, metadata=np.asarray(json.dumps({"format": "wafer-ddpm-npz-v1"})), weights=np.array([np.nan]))
    with pytest.raises(ValueError, match="Non-finite"):
        read_checkpoint(file)


def test_forged_tensor_shape_rejected_before_allocation(tmp_path):
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {"descr": "<f4", "fortran_order": False, "shape": (10**12,)})
    path = tmp_path / "huge_header.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("weights.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="shape/size"):
        read_checkpoint(path)


def test_default_resolution_forward_and_gradient():
    model = build_model(DEFAULT_CONFIG)
    image = torch.randn(1, 3, 256, 256)
    output = model(image, torch.tensor([500]))
    assert output.shape == image.shape
    output.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_network_or_hosted_wafer_data(monkeypatch):
    with pytest.raises(PermissionError, match="network"):
        socket.getaddrinfo("example.com", 443)
    with socket.socket() as client:
        with pytest.raises(PermissionError, match="network"):
            client.connect(("127.0.0.1", 1))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(PermissionError, match="hosted CI"):
        forbid_hosted_data()


def test_external_process_blocked():
    with pytest.raises(PermissionError, match="external process"):
        subprocess.run(["echo", "not-executed"], check=True)


def test_disguised_unsupported_format_rejected(tmp_path):
    path = tmp_path / "looks_like_image.png"
    path.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n")
    with pytest.raises(OSError):
        read_rgb(path)


def test_output_rejects_repo_and_urls():
    with pytest.raises(ValueError, match="outside"):
        private_output(".")
    with pytest.raises(ValueError, match="local filesystem"):
        private_output("https://example.com/data")


def test_modified_upstream_rejected_before_execution(tmp_path, upstream):
    for rel in ("ood/scoring.py", "ood/superpixels.py"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        shutil.copyfile(upstream / rel, tmp_path / rel)
    file = tmp_path / "ood/scoring.py"
    file.write_bytes(file.read_bytes() + b"\nraise RuntimeError('unreviewed code')\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_reviewed_module(tmp_path, "ood/scoring.py")


def test_official_intensity_score_matches_analytic_entropy(upstream):
    official = load_reviewed_module(upstream, "ood/scoring.py")
    original = np.zeros((4, 4, 3), np.uint8)
    original[2:] = 255
    recons = np.zeros((4, 4, 4, 3), np.uint8)
    recons[3] = 255
    result, _, _ = official.compute_delta_map_gpu(
        labels_fine=np.zeros((4, 4), np.int32), images_recon_all=recons,
        pca_feats_recon=np.zeros((4, 4, 4, 1), np.float32), label_image=original,
        label_pca_map=np.zeros((4, 4, 1), np.float32), bins_rgb=2, bins_pca=2,
        smooth_sigma=0, min_pixels=1, device=torch.device("cpu"), use_pca=False)
    entropy = -.75*math.log2(.75) - .25*math.log2(.25)
    nll = -.5*math.log2(.75) - .5*math.log2(.25)
    np.testing.assert_allclose(result, nll-entropy, atol=1e-5)


def test_feature_download_path_not_imported(upstream):
    import sys
    OfficialScorer(upstream, DEFAULT_SCORING)
    assert "torchvision" not in sys.modules
    assert "ood.embeddings" not in sys.modules


def test_diffusion_forward_equation():
    diffusion = Diffusion(8, "cosine")
    t = torch.tensor([0, 7])
    x = torch.zeros(2, 3, 4, 4)
    noisy = diffusion.q_sample(x, t, torch.ones_like(x))
    torch.testing.assert_close(noisy[:, 0, 0, 0], (1-diffusion.abar[t]).sqrt())
    assert diffusion.variance[0] == 0


def test_dps_guidance_changes_toward_measurement():
    diffusion = Diffusion(32, "cosine")
    class GaussianPrior(torch.nn.Module):
        def forward(self, image, t):
            return image * (1-diffusion.abar[t]).sqrt().view(-1, 1, 1, 1)
    clean = torch.full((1, 3, 8, 8), .5)
    model = GaussianPrior()
    unconditional = diffusion.reconstruct(model, clean, 2, .01, 0, 123)
    conditional = diffusion.reconstruct(model, clean, 2, .01, .5, 123)
    assert ((conditional-clean)**2).mean() < ((unconditional-clean)**2).mean()


def test_attention_backprop_for_dps():
    config = {**DEFAULT_CONFIG, "image_size": 16, "base_channels": 8, "channel_mult": [1, 2], "attention": True}
    model = build_model(config).requires_grad_(False)
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    gradient = torch.autograd.grad(model(x, torch.tensor([2])).square().mean(), x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_metrics_ties_and_missing_anomaly_labels():
    metrics = binary_metrics([0, 1, 0, 1], [.5, .5, .5, .5], .6)
    assert metrics["image_auroc"] == .5
    assert metrics["image_ap"] == .5
    assert metrics["anomaly_miss_rate"] == 1
    assert binary_metrics([0, 0], [0, 1], .5)["image_auroc"] is None


def test_no_silent_16bit_conversion(tmp_path):
    file = tmp_path / "sensor.tif"
    Image.fromarray(np.ones((8, 8), np.uint16) * 1000).save(file)
    with pytest.raises(ValueError, match="High-bit-depth"):
        read_rgb(file)


def test_anomalies_in_train_groups_are_never_used_for_training(tmp_path):
    make_synthetic(tmp_path)
    # Normal and anomalous images from the same wafer must stay in the same split.
    for group in range(8):
        folder = tmp_path / "anomaly" / f"synthetic_lot_{group:02d}"
        folder.mkdir()
        Image.fromarray(np.full((16, 16, 3), 200+group, np.uint8)).save(folder / "bad.png")
    manifest = tmp_path / "manifest.csv"
    rows = make_manifest([tmp_path / "normal"], [tmp_path / "anomaly"], manifest, group_parent=True)
    assert any(r["split"] == "train" and r["label"] == 1 for r in rows)
    config = {**DEFAULT_CONFIG, "image_size": 16, "base_channels": 8, "channel_mult": [1, 2],
              "attention": False, "max_steps": 1, "validate_every": 1, "val_images": 1, "batch_size": 2}
    checkpoint = train(manifest, tmp_path / "model", config, "cpu")
    meta, _ = read_checkpoint(checkpoint)
    expected = {r["pixel_sha256"] for r in rows if r["split"] == "train" and r["label"] == 0}
    assert set(meta["train_image_hashes"]) == expected


def test_calibration_cannot_silently_move_to_other_model(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"checkpoint_sha256": "old-model", "threshold": 0.5}))
    engine = SimpleNamespace(checkpoint_hash="new-model", settings=DEFAULT_SCORING)
    with pytest.raises(ValueError, match="different checkpoint"):
        calibration_threshold(path, engine)


def test_resume_matches_uninterrupted_cpu_training(tmp_path):
    make_synthetic(tmp_path)
    manifest = tmp_path / "manifest.csv"
    make_manifest([tmp_path / "normal"], [], manifest, group_parent=True)
    config = {**DEFAULT_CONFIG, "image_size": 16, "base_channels": 8, "channel_mult": [1, 2],
              "attention": False, "max_steps": 1, "validate_every": 1, "val_images": 1, "batch_size": 2}
    resume = train(manifest, tmp_path / "resumed", config, "cpu")
    config["max_steps"] = 2
    train(manifest, tmp_path / "resumed", config, "cpu", resume)
    uninterrupted = train(manifest, tmp_path / "uninterrupted", config, "cpu")
    _, a = read_checkpoint(resume)
    _, b = read_checkpoint(uninterrupted)
    for key in a:
        if key.startswith(("model/", "ema/", "optimizer/")):
            np.testing.assert_array_equal(a[key], b[key])
