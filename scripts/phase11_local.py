"""Phase 11 — local hide/recover probe on 20 evaluation images."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from steganography import pipeline  # noqa: E402
from steganography.hide import hide_and_recover  # noqa: E402
from steganography.mask import token_delta  # noqa: E402

CONDITIONS = ("matched", "mismatched")


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reference", choices=("matched", "mismatched", "both"), default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg_all = pipeline.load_config()
    cfg = dict(cfg_all["steganography"])
    cfg["evaluation"] = cfg_all.get("evaluation", {})
    size = int(cfg["image_size"])
    class_name = args.class_name or cfg.get("probe_class", "facial")
    limit = int(args.limit or cfg["num_images"])

    set_seed(int(cfg_all.get("seed", 42)))
    out_dir = paths.PROJECT_ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.PROJECT_ROOT / cfg["manifest_path"]
    if manifest_path.exists() and not args.force and not args.limit:
        existing = pipeline.read_json(manifest_path)
        if existing.get("split_version") and existing.get("pass") is not None:
            pass  # rerun anyway; cheap guard against stale partial reads

    records = pipeline.probe_records("evaluation", class_name, limit)
    mismatched_pool = pipeline.probe_records("evaluation", "common" if class_name != "common" else "style", limit)

    from steganography.edict import EDICT

    edict = EDICT(latent_dtype=torch.float32)
    conditions = CONDITIONS if args.reference == "both" else (args.reference,)
    started = time.time()
    results = {}

    for condition in conditions:
        cells = []
        for idx, rec in enumerate(records):
            ref = rec if condition == "matched" else mismatched_pool[idx]
            prompt = ref["kpub"]
            secret = pipeline.load_pixels(rec["path"], size)
            reference = pipeline.load_pixels(ref["reference_path"], size)
            t0 = time.time()
            hidden = hide_and_recover(edict, secret, reference, prompt, rec["token"], cfg,
                                      recover_prompt=rec["kpub"])
            elapsed = time.time() - t0

            recovered = hidden["recovered_pixels"]
            stego = hidden["stego_pixels"]
            rec_metrics = pipeline.metrics(secret, recovered, cfg)
            stego_metrics = pipeline.metrics(secret, stego, cfg)

            stem = f"{rec['image_id']}__{condition}"
            pipeline.save_pixels(secret, out_dir / f"{stem}_secret.png")
            pipeline.save_pixels(reference, out_dir / f"{stem}_reference.png")
            pipeline.save_pixels(stego, out_dir / f"{stem}_stego.png")
            pipeline.save_pixels(recovered, out_dir / f"{stem}_recovered.png")
            torch.save({
                "z_stego_pair": [z.cpu() for z in hidden["z_stego_pair"]],
                "delta": hidden["delta"].cpu(),
                "prompt": prompt,
                "recover_prompt": rec["kpub"],
                "eta": hidden["eta"],
                "steps": hidden["steps"],
                "mixing": hidden["mixing"],
                "strength": hidden["strength"],
                "token": rec["token"],
                "image_id": rec["image_id"],
                "condition": condition,
                "secret_path": str(out_dir / f"{stem}_secret.png"),
                "stego_path": str(out_dir / f"{stem}_stego.png"),
                "stego_sha256": _hash_tensor(stego),
                "recovered_sha256": _hash_tensor(recovered),
            }, out_dir / f"{stem}_latents.pt")

            cells.append({
                "image_id": rec["image_id"],
                "condition": condition,
                "reference_image_id": ref["image_id"],
                "token": rec["token"],
                "mask": hidden["mask"],
                "metrics": rec_metrics,
                "stego_metrics": stego_metrics,
                "stego_sha256": _hash_tensor(stego),
                "recovered_sha256": _hash_tensor(recovered),
                "latency_s": elapsed,
            })
            print(f"[phase11] {condition:10s} {rec['image_id']} psnr={rec_metrics['psnr']:.2f} "
                  f"ssim={rec_metrics['ssim']:.3f} stego_psnr={stego_metrics['psnr']:.2f} "
                  f"({elapsed:.1f}s)", flush=True)

        results[condition] = {
            "n": len(cells),
            "distribution": {
                key: pipeline.summarize([c["metrics"][key] for c in cells])
                for key in ("mse", "psnr", "ssim", "lpips")
            },
            "stego_distribution": {
                key: pipeline.summarize([c["stego_metrics"][key] for c in cells])
                for key in ("psnr", "ssim", "lpips")
            },
            "cells": cells,
        }

    # Determinism probe: rerun the first matched round-trip and compare hashes.
    determinism = {"checked": False}
    if "matched" in results and results["matched"]["cells"]:
        rec = records[0]
        secret = pipeline.load_pixels(rec["path"], size)
        reference = pipeline.load_pixels(rec["reference_path"], size)
        rerun = hide_and_recover(edict, secret, reference, rec["kpub"], rec["token"], cfg,
                                 recover_prompt=rec["kpub"])
        first = results["matched"]["cells"][0]
        determinism = {
            "checked": True,
            "image_id": rec["image_id"],
            "mask_deterministic": token_delta(rec["token"], shape=tuple(rerun["delta"].shape),
                                              fraction=float(cfg["mask_fraction"])).equal(rerun["delta"]),
            "stego_match": _hash_tensor(rerun["stego_pixels"]) == first["stego_sha256"],
            "recovered_match": _hash_tensor(rerun["recovered_pixels"]) == first["recovered_sha256"],
        }

    grid_rows = []
    for cell in results.get("matched", {}).get("cells", [])[:4]:
        stem = f"{cell['image_id']}__matched"
        grid_rows.append([out_dir / f"{stem}_{name}.png"
                          for name in ("secret", "reference", "stego", "recovered")])
    if grid_rows:
        pipeline.grid(grid_rows, out_dir / "phase11_grid.png")

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": 11,
        "approximate": False,
        "split_role": "evaluation",
        "class": class_name,
        "n_images": len(records),
        "config": {k: cfg[k] for k in ("eta", "mask_fraction", "mask_domain", "edict_steps",
                                       "edict_mixing", "image_size", "guidance_scale")},
        "mask_example": results.get("matched", {}).get("cells", [{}])[0].get("mask"),
        "determinism": determinism,
        "conditions": results,
        "elapsed_s": time.time() - started,
    }
    pipeline.write_json(manifest_path, payload)
    print(f"[phase11] wrote {manifest_path}", flush=True)
    edict.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
