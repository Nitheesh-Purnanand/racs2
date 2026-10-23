"""
Reusable evaluation driver: system x class x SNR.

This is the single measurement path used by Phases 15, 16, 21, 22 and 26. It
loads a role-tagged split, calls a runner for every (image, SNR) case, computes
MSE/PSNR/SSIM/LPIPS on unit-range tensors, and writes a frozen report artifact.

Usage:
    python scripts/eval_report.py --system dummy --split evaluation \
        --runner scripts/dummy_runner.py:run_case \
        --out experiments/results/phase6_dummy_eval.json

The runner contract is documented in `scripts/dummy_runner.py`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image

import common.paths  # noqa: F401  (caches on D:)
from common.paths import PROJECT_ROOT
from evaluation.metrics import all_metrics
from evaluation.report import build_report, write_report
from evaluation.splits import Split, load_tokens, resolve_path, sha256_file


def log(message):
    print(message, flush=True)


def pil_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_runner(spec):
    if ":" not in spec:
        raise SystemExit("--runner must look like 'module:function' or 'path.py:function'")
    target, function_name = spec.rsplit(":", 1)
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = Path(target)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise SystemExit(f"runner file not found: {path}")
        module_name = "racs2_runner_" + sha256_file(path)[:8]
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if not hasattr(module, function_name):
        raise SystemExit(f"runner {spec} has no attribute {function_name!r}")
    return getattr(module, function_name)


def load_config(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with io.open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a system on a role-tagged split.")
    parser.add_argument("--system", required=True)
    parser.add_argument("--split", required=True, choices=["calibration", "evaluation"])
    parser.add_argument("--runner", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--snr", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="limit images per class (debug only)")
    parser.add_argument("--approximate", action="store_true",
                        help="required when --split calibration")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.split == "calibration" and not args.approximate:
        raise SystemExit(
            "refusing to report calibration data as authoritative; pass --approximate"
        )

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    snr_values = args.snr or [int(value) for value in eval_cfg.get("snr_db", [5, 10, 15, 20])]
    lpips_net = eval_cfg.get("lpips_net", "alex")

    split = Split.load(role=args.split,
                       allow_calibration=(args.split == "calibration"))
    tokens, tokens_version = load_tokens(split=split)
    runner = load_runner(args.runner)

    results = []
    started = time.time()
    total_cases = 0
    for cls in eval_cfg.get("classes", ["common", "facial", "style"]):
        records = split.records(cls)
        if args.limit:
            records = records[:args.limit]
        log(f"[eval] {args.split}/{cls}: {len(records)} images x {len(snr_values)} SNRs")
        for record in records:
            record = dict(record)
            record.update(tokens[record["image_id"]])
            for snr_db in snr_values:
                case_started = time.time()
                output = runner(record, snr_db, args.split)
                secret = output["secret"]
                recovered = output["recovered"]
                if secret.size != recovered.size:
                    raise RuntimeError(
                        f"{record['image_id']}@{snr_db}: size mismatch "
                        f"{secret.size} vs {recovered.size}"
                    )
                metrics = all_metrics(pil_to_tensor(secret), pil_to_tensor(recovered),
                                      with_lpips=True, lpips_net=lpips_net)
                case = {
                    "image_id": record["image_id"],
                    "class": cls,
                    "snr_db": int(snr_db),
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                }
                for name in ("reference_latency_s", "total_latency_s", "peak_vram_gb"):
                    if name in output:
                        case[name] = output[name]
                if "total_latency_s" not in case:
                    case["total_latency_s"] = time.time() - case_started
                results.append(case)
                total_cases += 1
                if total_cases % 10 == 0:
                    log(f"[eval] {total_cases} cases")

    report = build_report(
        system=args.system,
        role=args.split,
        split=split,
        results=results,
        approximate=(args.split == "calibration"),
        percentiles=tuple(eval_cfg.get("report_percentiles", [5, 25, 50, 75, 95])),
        meta={
            "lpips_net": lpips_net,
            "runner": args.runner,
            "tokens_version": tokens_version,
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "seconds_total": round(time.time() - started, 3),
        },
        config={"evaluation": eval_cfg, "system": args.system,
                "snr_db": snr_values, "split": args.split},
    )
    out_path = write_report(report, args.out)

    log("")
    log(f"=== {args.system} on {args.split} (approximate={report['approximate']}) ===")
    header = f"{'class':8s} {'SNR':>4s} {'n':>4s} {'PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s}"
    log(header)
    for cell in report["cells"]:
        metrics = cell["metrics"]
        psnr = metrics.get("psnr", {}).get("mean")
        ssim = metrics.get("ssim", {}).get("mean")
        lpips = metrics.get("lpips", {}).get("mean")
        log(f"{cell['class']:8s} {cell['snr_db']:4d} {cell['n']:4d} "
            f"{(psnr if psnr is not None else float('nan')):8.3f} "
            f"{(ssim if ssim is not None else float('nan')):7.4f} "
            f"{(lpips if lpips is not None else float('nan')):7.4f}")
    log(f"cases: {total_cases}")
    log(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
