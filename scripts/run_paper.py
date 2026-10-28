"""Recompute paper metrics and/or run the official scorer, with strict result checks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from bootstrap import ROOT, bootstrap


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def call(args, cwd, env, log):
    print("RUN:", " ".join(map(str, args)), flush=True)
    with log.open("w") as f:
        process = subprocess.Popen(list(map(str, args)), cwd=cwd, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end="", flush=True)
        process.wait()
    require(process.returncode == 0, f"Command failed ({process.returncode}): {log}")


def verify_summaries(out, sets, expected):
    counts = {}
    for category, sub in sets:
        path = out / "json" / sub / f"{category}_{sub}.json"
        data = json.loads(path.read_text())
        require(not data["failed"], f"Official evaluator caught sample errors: {data['failed']}")
        require(len(data["samples"]) == expected, f"Expected {expected} samples: {path}")
        require(all(s["n_reconstructions"] == 20 for s in data["samples"].values()), "Expected 20 draws per image")
        counts[sub] = len(data["samples"])
    return counts


def run(stage, assets, output, device="cpu", threads=4):
    require(os.environ.get("GITHUB_ACTIONS") == "true", "Paper dataset execution is restricted to GitHub Actions")
    started = time.perf_counter()
    upstream = bootstrap()
    assets, out = assets.resolve(), output.resolve()
    require(assets.is_dir(), f"Missing assets: {assets}")
    require(not out.exists(), f"Output exists: {out}; select a fresh directory")
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    link = upstream / "assets"
    if link.exists() or link.is_symlink():
        require(link.resolve() == assets, f"Existing assets link points elsewhere: {link}")
    else:
        link.symlink_to(assets, target_is_directory=True)
    env = dict(os.environ, ASSETS=str(assets), OUT=str(out), ABL_SEED="0",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    python = sys.executable
    result = {"stage": stage, "device": device, "threads": threads,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "upstream_commit": json.loads((ROOT / "upstream.json").read_text())["commit"],
              "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in
                           ["torch", "torchvision", "numpy", "scipy", "scikit-image", "scikit-learn", "Pillow"]},
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "diffusion_sampling_rerun": False, "training_rerun": False}
    if stage == "reference":
        call([python, "scripts/make_table1.py", "--maps", "reference", "--exact"], upstream, env, out / "logs/table1.log")
        table = json.loads((out / "table1.json").read_text())
        require(sum(len(methods) * 3 for methods in table["table"].values()) == 63, "Incomplete Table 1")
        result.update(status="PASS", compared_cells=63, scope="Metrics recomputed from author-provided score maps; no fresh inference")
    elif stage == "xray":
        env["DDAD_FE_WEIGHTS"] = str(assets / "checkpoints/lactos/backbone_xray_ddad_lastlayer_feat4.pth")
        tag = "sp10_m20_hs0_rgbonly_b16"
        for sub in ["xrsc", "xrgd"]:
            call([python, "tools/seeded_run.py", "embedder/evaluate_ddad_fe.py",
                  "--config", "configs/lactos_xray.yaml", "--device", device,
                  "--subcategory", sub, "--results_dir", assets / f"recons/results_patches_ddad_native_xray_{sub}",
                  "--gt_root", assets / "data/xray/ground_truth", "--image_dir", assets / f"data/xray/test/{sub}",
                  "--figures_dir", out / "figures", "--dump_maps", out / f"lactos_maps/xray/{tag}/{sub}",
                  "--output_dir", out / f"json/{sub}"], upstream, env, out / f"logs/{sub}.log")
        result["samples"] = verify_summaries(out, [("xray", "xrsc"), ("xray", "xrgd")], 72)
        # --maps mine forbids filling fresh-run gaps with the authors' maps.
        call([python, "scripts/make_table1.py", "--maps", "mine", "--only", "lactos"], upstream, env, out / "logs/table1.log")
        table = json.loads((out / "table1.json").read_text())
        require(set(table["table"]["xray"]) == {"lactos"}, "Missing fresh DVXray cell")
        result.update(status="PASS", metrics=dict(zip(["AUROC", "AP", "AUPRO"], table["table"]["xray"]["lactos"])),
                      scope="All 144 DVXray images freshly scored from the authors' 20 reconstructions per image, original paper settings")
    elif stage == "ffhq":
        import yaml
        cfg = yaml.safe_load((upstream / "configs/lactos_ffhq_faces.yaml").read_text())
        cfg["embeddings"]["batch_size"] = 1  # Direct encoding, bounded peak RAM on an 8 GB CPU host.
        config = out / "ffhq_cpu.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        for variant in ["full", "rgbonly"]:
            args = [python, "tools/seeded_run.py", "evaluate.py", "--config", config,
                    "--device", device, "--max_samples", "2", "--figures_dir", out / "figures",
                    "--dump_maps", out / variant, "--output_dir", out / f"json/{variant}"]
            if variant == "rgbonly":
                args.append("--no_pca")
            call(args, upstream, env, out / f"logs/{variant}.log")
            summary = json.loads((out / f"json/{variant}/faces_random.json").read_text())
            require(not summary["failed"] and len(summary["samples"]) == 2, "FFHQ sample scoring failed")
        sys.path.insert(0, str(upstream))
        from full_metric_table import full_metrics
        import numpy as np
        from scipy.ndimage import gaussian_filter
        fresh, refs, gt, valid = [], [], [], []
        refbase = assets / "reference_results/lactos_maps/ffhq"
        files = sorted((out / "full").glob("*.npz"))
        require(len(files) == 2, "Expected two FFHQ maps")
        for f in files:
            with np.load(f) as full, np.load(out / "rgbonly" / f.name) as rgb:
                d = np.nan_to_num(full["delta_map"].astype(np.float64))
                r = np.nan_to_num(rgb["delta_map"].astype(np.float64))
                fresh.append(gaussian_filter((r + .25 * (d-r)).astype(np.float32), 5, mode="nearest"))
                gt.append(full["gt_mask"] > 0); valid.append(np.ones(d.shape, bool))
            with np.load(refbase / "sp20_m20_hs1_b16x32/faces" / f.name) as full, np.load(refbase / "sp20_m20_hs1_rgbonly_b16/faces" / f.name) as rgb:
                d = np.nan_to_num(full["delta_map"].astype(np.float64))
                r = np.nan_to_num(rgb["delta_map"].astype(np.float64))
                refs.append(gaussian_filter((r + .25 * (d-r)).astype(np.float32), 5, mode="nearest"))
        fm, rm = full_metrics(fresh, gt, valid), full_metrics(refs, gt, valid)
        keys = ["pooled_auroc", "pooled_aupr", "aupro"]
        deltas = {k: fm[k]-rm[k] for k in keys}
        require(all(abs(v) <= .005 for v in deltas.values()), f"FFHQ metric discrepancy exceeds .005: {deltas}")
        result.update(status="PASS", samples=[f.stem for f in files], metrics={k: fm[k] for k in keys},
                      reference_subset_metrics={k: rm[k] for k in keys}, metric_deltas=deltas,
                      scope="Two FFHQ faces: fresh full and RGB-only scores combined with paper lambda=.25, sigma=5",
                      deviations=["CPU instead of GPU", "embedding batch 1 instead of 4; direct AE encoding preserved"],
                      caveat="Two-image installation check, not the full 100-image FFHQ benchmark")
    result["runtime_seconds"] = time.perf_counter() - started
    write_json(out / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["reference", "xray", "ffhq"])
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    run(a.stage, a.assets, a.output, a.device, a.threads)
