"""Phase 16 — XuNet steganalysis training and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from evaluation.report import config_hash  # noqa: E402
from security.steganalysis import (XUNET_ARCH, build_dataset, evaluate,  # noqa: E402
                                   train_xunet, XuNet)
from steganography import pipeline  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    cfg_all = pipeline.load_config()
    cfg = dict(cfg_all["security"]["steganalysis"])
    bcfg = cfg_all["baseline"]
    table = pipeline.read_json(paths.PROJECT_ROOT / bcfg["evaluation_output"])
    manifest = pipeline.read_json(paths.PROJECT_ROOT / bcfg["evaluation_manifest"])
    started = time.time()
    dataset = build_dataset(manifest, table, cfg)
    out_dir = paths.PROJECT_ROOT / "experiments" / "steganalysis"

    if args.eval_only and args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = paths.PROJECT_ROOT / ckpt
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = XuNet()
        model.load_state_dict(state["state_dict"])
        train_info = {"checkpoint": str(ckpt),
                      "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                      "best_epoch": state.get("best_epoch"), "history": []}
        splits = {s: evaluate(model, dataset, s) for s in ("train", "val", "test")}
    else:
        train_info = train_xunet(dataset, cfg, out_dir)
        splits = {s: evaluate(train_info["model"], dataset, s)
                  for s in ("train", "val", "test")}

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approximate": False,
        "split_role": "evaluation",
        "baseline_table_sha256": hashlib.sha256(
            (paths.PROJECT_ROOT / bcfg["evaluation_output"]).read_bytes()).hexdigest(),
        "model": {"name": "xunet", "architecture": XUNET_ARCH, "config_hash":
                  config_hash(cfg), "snr_db": dataset["snr_db"]},
        "dataset": {"n_images": dataset["n_images"],
                    "class_balance": dataset["class_balance"],
                    "patch_grid": dataset["patch_grid"],
                    "n_patches_per_image": dataset["n_patches_per_image"],
                    "split_strategy": cfg.get("split_strategy"),
                    "n_samples": {s: int(len(dataset["parts"][s][1]))
                                  for s in ("train", "val", "test")},
                    "split_hashes": dataset["split_hashes"],
                    "split_ids": {s: dataset.get(s + "_ids", [])
                                  for s in ("train", "val", "test")},
                    "leakage_check": "image_id disjoint by construction"},
        "training": {"epochs_configured": int(cfg["epochs"]),
                     "best_epoch": train_info["best_epoch"],
                     "batch_size": int(cfg["batch_size"]),
                     "learning_rate": float(cfg["learning_rate"]),
                     "seed": int(cfg["seed"]),
                     "history": train_info["history"]},
        "checkpoint_path": train_info["checkpoint"],
        "checkpoint_sha256": train_info["sha256"],
        "results": splits,
        "chance_rate": 0.5,
        "paper_anchor": {"accuracy": 0.5091, "label": "approximate"},
        "elapsed_s": time.time() - started,
    }
    pipeline.write_json(paths.PROJECT_ROOT / cfg_all["security"]["output_steganalysis"],
                        payload)
    print(f"[phase16] XuNet test accuracy={splits['test']['accuracy']:.3f} "
          f"(n={splits['test']['n']}) -> {cfg_all['security']['output_steganalysis']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
