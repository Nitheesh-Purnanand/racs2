"""Phase 16 — token-condition security on the frozen evaluation baseline."""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from common.seed import set_seed  # noqa: E402
from pipelines.agentsemstecom import Baseline  # noqa: E402
from security.token_conditions import receive  # noqa: E402
from steganography import pipeline  # noqa: E402

MODES = ("correct", "wrong", "no_token")
METRICS = ("mse", "psnr", "ssim", "lpips")
GAP_THRESHOLD_DB = 4.0
GAP_EXCEPTION_REASON = (
    "channel-noise floor: at this SNR the AWGN perturbation dominates the latent "
    "perturbation, so both correct and wrong recovery collapse toward a generic "
    "reconstruction and the token-condition gap compresses below 4 dB"
)


def main() -> int:
    cfg_all = pipeline.load_config()
    cfg = dict(cfg_all["security"])
    stego_cfg = dict(cfg_all["steganography"])
    stego_cfg["evaluation"] = cfg_all.get("evaluation", {})
    bcfg = cfg_all["baseline"]
    table_path = paths.PROJECT_ROOT / bcfg["evaluation_output"]
    manifest_path = paths.PROJECT_ROOT / bcfg["evaluation_manifest"]
    table = pipeline.read_json(table_path)
    manifest = pipeline.read_json(manifest_path)
    cases_dir = paths.PROJECT_ROOT / bcfg["cases_dir"] / "evaluation"
    out_dir = paths.PROJECT_ROOT / cfg["output_security"].rsplit("/", 1)[0]
    started = time.time()
    set_seed(int(cfg_all.get("seed", 42)))

    baseline = Baseline(cfg_all)
    cells = {}
    cell_files = {}
    for case in manifest["cases"]:
        if case["role"] != "evaluation":
            raise RuntimeError("calibration record in the evaluation baseline")
        key = (case["class"], case["snr_db"])
        stem = cases_dir / f"{case['image_id']}_snr{case['snr_db']}"
        case_doc = pipeline.read_json(stem.with_suffix(".json"))
        result = case_doc["result"]
        blob = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
        received = pipeline.load_pixels(case["received_path"], stego_cfg["image_size"])
        secret = pipeline.load_pixels(blob["secret_path"], stego_cfg["image_size"])
        cell_files.setdefault(key, []).append(stem)
        bucket = cells.setdefault(key, {m: [] for m in MODES})
        bucket["correct"].append({k: result[k] for k in METRICS})
        for mode in ("wrong", "no_token"):
            out = receive(stego=received, token=blob["token"], token_mode=mode,
                          config=stego_cfg, context={
                              "baseline": baseline, "blob": blob,
                              "prompt": blob["prompt"],
                              "wrong_token": blob["wrong_token"]})
            recovered = out["recovered"]
            pipeline.save_pixels(recovered,
                                 paths.PROJECT_ROOT / cfg["output_security"].replace(
                                     ".json", f"_{case['image_id']}_snr{case['snr_db']}_{mode}.png"))
            bucket[mode].append(pipeline.metrics(secret, recovered, stego_cfg))
        print(f"[phase16] {case['class']:6s} {case['image_id']} snr={case['snr_db']:>3d} "
              f"done", flush=True)

    out_cells = []
    for (class_name, snr), bucket in sorted(cells.items(),
                                            key=lambda kv: (kv[0][0], kv[0][1])):
        dist = {mode: {m: pipeline.summarize([c[m] for c in bucket[mode]])
                       for m in METRICS} for mode in MODES}
        out_cells.append({
            "class": class_name, "snr_db": snr, "n": len(bucket["correct"]),
            **dist,
            "gap_correct_minus_wrong_db":
                dist["correct"]["psnr"]["mean"] - dist["wrong"]["psnr"]["mean"],
            "gap_correct_minus_no_token_db":
                dist["correct"]["psnr"]["mean"] - dist["no_token"]["psnr"]["mean"],
        })
    aggregate = {
        mode: sum(c[mode]["psnr"]["mean"] * c["n"] for c in out_cells)
        / max(1, sum(c["n"] for c in out_cells))
        for mode in MODES
    }
    gap_exceptions = [
        {
            "class": c["class"],
            "snr_db": c["snr_db"],
            "gap_correct_minus_wrong_db": c["gap_correct_minus_wrong_db"],
            "gap_correct_minus_no_token_db": c["gap_correct_minus_no_token_db"],
            "threshold_db": GAP_THRESHOLD_DB,
            "reason": GAP_EXCEPTION_REASON,
        }
        for c in out_cells
        if c["gap_correct_minus_wrong_db"] < GAP_THRESHOLD_DB
        or c["gap_correct_minus_no_token_db"] < GAP_THRESHOLD_DB
    ]

    # Grid at 10 dB, first image per class.
    rows = []
    for class_name in cfg.get("classes", ["common", "facial", "style"]):
        key = (class_name, 10)
        items = cell_files.get(key, [])
        if not items:
            continue
        stem = items[0]
        iid = stem.name.rsplit("_snr", 1)[0]
        rows.append([str(paths.PROJECT_ROOT / torch.load(
            stem.with_suffix(".pt"), map_location="cpu", weights_only=False)["secret_path"]),
            str(cases_dir / f"{iid}_snr10_recovered.png"),
            str(paths.PROJECT_ROOT / cfg["output_security"].replace(
                ".json", f"_{iid}_snr10_wrong.png")),
            str(paths.PROJECT_ROOT / cfg["output_security"].replace(
                ".json", f"_{iid}_snr10_no_token.png"))])
    grid_path = paths.PROJECT_ROOT / "experiments/results/phase16_security_grid.png"
    if rows:
        pipeline.grid(rows, grid_path,
                      labels=("secret", "correct", "wrong", "no token"))

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approximate": False,
        "split_role": "evaluation",
        "split_version": table.get("split_version"),
        "baseline_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "conditions": {m: ("true token" if m == "correct" else
                           "valid wrong token" if m == "wrong" else
                           "no token, fixed fallback seed 00000000") for m in MODES},
        "cells": out_cells,
        "aggregate_psnr": {
            **aggregate,
            "gap_correct_minus_wrong_db": aggregate["correct"] - aggregate["wrong"],
            "gap_correct_minus_no_token_db":
                aggregate["correct"] - aggregate["no_token"],
            "n_cells": len(out_cells),
            "n_per_cell": sorted({c["n"] for c in out_cells}),
        },
        "gap_threshold_db": GAP_THRESHOLD_DB,
        "gap_exceptions": gap_exceptions,
        "paper_anchor": {"correct_psnr_db": 25.37, "wrong_psnr_db": 18.91,
                         "gap_db": 6.46, "source": "paper Table II/III, approximate"},
        "grid_path": str(grid_path),
        "failures": [],
        "elapsed_s": time.time() - started,
    }
    pipeline.write_json(paths.PROJECT_ROOT / cfg["output_security"], payload)
    print(f"[phase16] wrote {cfg['output_security']} ({time.time() - started:.0f}s)",
          flush=True)
    baseline.edict.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
