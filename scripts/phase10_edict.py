"""
Phase 10 — EDICT round-trip sweep.

Usage:
    python scripts/phase10_edict.py
    python scripts/phase10_edict.py --quick
    python scripts/phase10_edict.py --split evaluation --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths
import common.hf_lowmem

common.hf_lowmem.install()

import numpy as np
import torch
import yaml
from PIL import Image

from common.seed import set_seed
from evaluation.splits import Split
from semantic.cache import read_cache_npz, sha256_json
from steganography.edict import EDICT


def _latent_stats(original: torch.Tensor, recovered: torch.Tensor) -> Dict[str, float]:
    diff = (recovered - original).abs()
    denom = original.abs().max().clamp_min(1e-12)
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "max_rel_error": float(diff.max() / denom),
    }


def _measure(edict: EDICT, latent: torch.Tensor, prompt: str, steps: int,
             mixing: float, guidance: float) -> Dict[str, float]:
    pair = edict.forward(
        latent, prompt=prompt, steps=steps, mixing=mixing,
        guidance_scale=guidance,
    )
    back = edict.inverse(
        pair, prompt=prompt, steps=steps, mixing=mixing,
        guidance_scale=guidance,
    )
    a = _latent_stats(latent, back[0])
    b = _latent_stats(latent, back[1])
    return {
        "max_abs_error": max(a["max_abs_error"], b["max_abs_error"]),
        "mean_abs_error": max(a["mean_abs_error"], b["mean_abs_error"]),
        "max_rel_error": max(a["max_rel_error"], b["max_rel_error"]),
        "finite": bool(np.isfinite(a["max_abs_error"]) and np.isfinite(b["max_abs_error"])),
    }


def _plot(cells: List[Dict], out_path: Path, steps_list, mixing_list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.full((len(steps_list), len(mixing_list)), np.nan)
    for cell in cells:
        if cell["steps"] not in steps_list or cell["mixing"] not in mixing_list:
            continue
        i = steps_list.index(cell["steps"])
        j = mixing_list.index(cell["mixing"])
        grid[i, j] = cell["max_abs_random"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(np.log10(np.maximum(grid, 1e-16)), cmap="viridis")
    ax.set_xticks(range(len(mixing_list)))
    ax.set_xticklabels([str(v) for v in mixing_list])
    ax.set_yticks(range(len(steps_list)))
    ax.set_yticklabels([str(v) for v in steps_list])
    ax.set_xlabel("mixing coefficient")
    ax.set_ylabel("steps")
    for i in range(len(steps_list)):
        for j in range(len(mixing_list)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1e}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10 max abs error")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10: EDICT round-trip sweep")
    parser.add_argument("--quick", action="store_true", help="2 latents per kind per cell")
    parser.add_argument("--latents", type=int, default=0,
                        help="override the number of latents per kind")
    parser.add_argument("--operating-only", action="store_true",
                        help="run only the configured operating point")
    parser.add_argument("--split", default="evaluation", choices=["calibration", "evaluation"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    set_seed(42, verbose=True)
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = config.get("edict", {})
    steps_list = [int(v) for v in cfg.get("sweep_steps", [10, 20, 50])]
    mixing_list = [float(v) for v in cfg.get("sweep_mixing", [0.5, 0.7, 0.93, 0.99])]
    guidance = float(cfg.get("guidance_scale", 7.5))
    n_random = 2 if args.quick else int(cfg.get("num_random_latents", 20))
    n_image = 2 if args.quick else int(cfg.get("num_image_latents", 20))
    if args.latents:
        n_random = n_image = args.latents
    if args.limit:
        n_image = min(n_image, args.limit)

    print(f"[edict] model load ...")
    edict = EDICT(
        dtype=torch.float16 if cfg.get("dtype", "float16") == "float16" else torch.float32,
        latent_dtype=torch.float32,
    )
    print(f"[edict] model = {edict.model_id}")

    # Random latents
    gen = torch.Generator(device="cpu").manual_seed(42)
    latent_size = int(cfg.get("latent_size", 32))
    image_size = int(cfg.get("image_size", 256))
    random_latents = [torch.randn((1, 4, latent_size, latent_size), generator=gen).to(edict.device)
                      for _ in range(n_random)]

    # Real image latents (fixed sorted order)
    split = Split.load(role=args.split, allow_calibration=(args.split == "calibration"))
    records = split.records()[:n_image]
    image_latents = []
    prompts = []
    for rec in records:
        image = Image.open(paths.PROJECT_ROOT / rec["path"]).convert("RGB")
        image_latents.append(edict.encode_image(image, size=image_size))
        cache_path = paths.RESULTS_DIR / "public_conditions" / f"{rec['image_id']}.npz"
        if cache_path.exists():
            prompts.append(read_cache_npz(cache_path)["kpub"])
        else:
            prompts.append("a photorealistic photograph, natural composition, detailed textures")

    cells: List[Dict] = []
    t_all = time.time()
    op_steps = int(cfg.get("steps", 50))
    op_mix = float(cfg.get("mixing", 0.93))
    out = paths.PROJECT_ROOT / cfg.get("results_path", "experiments/results/phase10_edict.json")
    existing_cells = []
    if args.operating_only and out.exists():
        try:
            with open(out, "r", encoding="utf-8") as fh:
                existing_cells = json.load(fh).get("cells", [])
        except Exception:  # noqa: BLE001
            existing_cells = []
    grid = ([(op_steps, op_mix)] if args.operating_only
            else [(s, m) for s in steps_list for m in mixing_list])
    for steps, mixing in grid:
            t0 = time.time()
            errs_by_kind = {}
            for kind, latents, prompt_list in (
                ("random", random_latents, ["a photorealistic photograph"] * len(random_latents)),
                ("image", image_latents, prompts),
            ):
                kind_errs = []
                for latent, prompt in zip(latents, prompt_list):
                    kind_errs.append(
                        _measure(edict, latent, prompt, steps, mixing, guidance)
                    )
                errs_by_kind[kind] = kind_errs
            errs = errs_by_kind["random"] + errs_by_kind["image"]
            cells.append({
                "steps": steps,
                "mixing": mixing,
                "n_random": n_random,
                "n_image": n_image,
                "max_abs_error": max(e["max_abs_error"] for e in errs),
                "max_abs_random": max(e["max_abs_error"] for e in errs_by_kind["random"]),
                "max_abs_image": max(e["max_abs_error"] for e in errs_by_kind["image"]),
                "mean_abs_error": float(np.mean([e["mean_abs_error"] for e in errs])),
                "max_rel_error": max(e["max_rel_error"] for e in errs),
                "finite": all(e["finite"] for e in errs),
                "elapsed_s": round(time.time() - t0, 2),
            })
            print(f"[cell] steps={steps} mixing={mixing} "
                  f"max_abs={cells[-1]['max_abs_error']:.3e} "
                  f"max_rel={cells[-1]['max_rel_error']:.3e} "
                  f"({cells[-1]['elapsed_s']}s)")

    if existing_cells:
        merged = [c for c in existing_cells
                  if not any(c["steps"] == n["steps"] and c["mixing"] == n["mixing"]
                             for n in cells)]
        cells = merged + cells
    op = next((c for c in cells if c["steps"] == op_steps and c["mixing"] == op_mix), None)
    result = {
        "schema_version": 1,
        "approximate": False,
        "model_id": edict.model_id,
        "config_hash": sha256_json(cfg),
        "seed": 42,
        "split_role": args.split,
        "split_version": split.version,
        "guidance_scale": guidance,
        "image_size": image_size,
        "latent_size": latent_size,
        "operating_point": {"steps": op_steps, "mixing": op_mix},
        "cells": cells,
        "operating_point_result": op,
        "decision": (
            f"Adopt {op_steps} steps and mixing {op_mix} if the operating-point "
            "relative error is <= 1e-6 and the sweep supports it."
        ),
        "elapsed_s": round(time.time() - t_all, 2),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, allow_nan=True)
        fh.write("\n")
    print(f"[edict] wrote {out}")

    _plot(cells, paths.PROJECT_ROOT / cfg.get("grid_path", "experiments/results/phase10_edict_sweep.png"),
          steps_list, mixing_list)
    edict.unload()

    if op is None:
        print("[edict] operating point not present in sweep; exit 1")
        sys.exit(1)
    print(f"[edict] operating point max_rel={op['max_rel_error']:.3e}")


if __name__ == "__main__":
    main()
