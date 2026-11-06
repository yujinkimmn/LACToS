"""CLI. All user-data commands are local-only and disabled on GitHub Actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import audit_manifest, make_manifest
from .inference import Inference, calibrate, evaluate, infer
from .scoring import DEFAULT_SCORING, validate_scoring
from .security import forbid_hosted_data, private_output, verified_upstream
from .train import get_config, train


def inference_config(path):
    config = dict(DEFAULT_SCORING)
    if path:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict) or set(values) - set(config):
            raise ValueError("Unknown inference JSON keys.")
        config.update(values)
    return validate_scoring(config)


def dataset_args(parser):
    parser.add_argument("--normal-root", nargs="+", required=True)
    parser.add_argument("--anomaly-root", nargs="+", default=[])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--group-regex", help="Regex on root-relative path, including named (?P<group>...) capture.")
    group.add_argument("--group-parent", action="store_true", help="Immediate parent folder is a globally unique wafer/lot ID.")
    parser.add_argument("--include", default="*", help="Filename glob, e.g. '*preproc*'.")
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)


def inference_args(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True, help="Local reviewed LACToS directory; never downloaded by this command.")
    parser.add_argument("--inference-config")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", required=True)


def prepare(args, path):
    return make_manifest(args.normal_root, args.anomaly_root, path, args.group_regex,
                         args.group_parent, args.include, args.val_fraction, args.test_fraction, args.seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline wafer DDPM training → DPS reconstructions → official LACToS intensity scoring")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Build a grouped train/val/test manifest from normal image roots.")
    dataset_args(p)
    p.add_argument("--work-dir", required=True)
    p = sub.add_parser("audit-data")
    p.add_argument("--manifest", required=True)
    p = sub.add_parser("doctor", help="Check dependencies, device and reviewed upstream files without loading a dataset.")
    p.add_argument("--upstream", required=True)
    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-config")
    p.add_argument("--resume")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p = sub.add_parser("infer", help="One image or multiple local roots; generates reconstructions automatically.")
    inference_args(p)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--include", default="*")
    p.add_argument("--calibration")
    p.add_argument("--save-recon", action="store_true")
    p = sub.add_parser("calibrate")
    inference_args(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--target-fpr", type=float, default=.01)
    p = sub.add_parser("evaluate")
    inference_args(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--save-recon", action="store_true")
    p = sub.add_parser("run", help="Prepare → train → calibrate → evaluate in one local command.")
    dataset_args(p)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--train-config")
    p.add_argument("--inference-config")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--target-fpr", type=float, default=.01)
    p.add_argument("--save-recon", action="store_true")
    p = sub.add_parser("smoke", help="Synthetic-only end-to-end verification; never accepts dataset paths.")
    p.add_argument("--upstream", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command not in {"doctor", "smoke"}:
            forbid_hosted_data()
        if args.command == "doctor":
            import torch
            hashes = verified_upstream(args.upstream)
            print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                              "upstream_hashes": hashes, "network_guard": "enabled by python -m entry point"}, indent=2))
        elif args.command == "prepare":
            rows = prepare(args, private_output(args.work_dir) / "manifest.csv")
            print(f"Prepared {len(rows)} images; grouped split and decoded-image duplicates verified.")
        elif args.command == "audit-data":
            print(f"Verified {len(audit_manifest(args.manifest))} images.")
        elif args.command == "train":
            train(args.manifest, args.output, get_config(args.train_config), args.device, args.resume)
        elif args.command in {"infer", "calibrate", "evaluate"}:
            engine = Inference(args.checkpoint, args.upstream, inference_config(args.inference_config), args.device)
            if args.command == "infer":
                infer(engine, args.input, args.output, args.include, args.calibration, args.save_recon)
            elif args.command == "calibrate":
                calibrate(engine, args.manifest, args.output, args.target_fpr)
            else:
                evaluate(engine, args.manifest, args.calibration, args.output, args.save_recon)
        elif args.command == "run":
            verified_upstream(args.upstream)
            work = private_output(args.work_dir)
            manifest = work / "manifest.csv"
            prepare(args, manifest)
            checkpoint = train(manifest, work / "model", get_config(args.train_config), args.device)
            engine = Inference(checkpoint, args.upstream, inference_config(args.inference_config), args.device)
            calibrate(engine, manifest, work / "calibration", args.target_fpr)
            evaluate(engine, manifest, work / "calibration" / "calibration.json", work / "test", args.save_recon)
        elif args.command == "smoke":
            from .smoke import run_smoke
            run_smoke(args.upstream, private_output(args.output))
    except (ValueError, RuntimeError, OSError, FloatingPointError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit("Use python -m wafer_diffusion so the offline guard starts before imports.")
