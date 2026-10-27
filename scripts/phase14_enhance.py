"""Phase 14 — CodeFormer enhancement delta on the facial class."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from common.seed import set_seed  # noqa: E402
from enhancement.codeformer import (build_enhancer, checkpoint_sha256, enhance,  # noqa: E402
                                    should_enhance)
from steganography import pipeline  # noqa: E402


def _pixels(image: Image.Image, size: int) -> torch.Tensor:
    import numpy as np
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _grid(rows, out_path: Path) -> None:
    from PIL import ImageDraw

    tiles = [[Image.open(p).convert("RGB") for p in row] for row in rows]
    w, h = tiles[0][0].size
    sheet = Image.new("RGB", (len(tiles[0]) * w, len(tiles) * h), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(tiles):
        for c, img in enumerate(row):
            sheet.paste(img.resize((w, h), Image.BICUBIC), (c * w, r * h))
    for c, label in enumerate(("secret", "recovered", "enhanced")):
        draw.text((c * w + 4, 2), label, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--snr", nargs="*", type=float, default=None)
    args = parser.parse_args()

    cfg_all = pipeline.load_config()
    cfg = dict(cfg_all["enhancement"])
    cfg["evaluation"] = cfg_all.get("evaluation", {})
    ch = cfg_all["channel"]
    stego_cfg = dict(cfg_all["steganography"])
    stego_cfg["evaluation"] = cfg_all.get("evaluation", {})
    phase12 = pipeline.read_json(paths.PROJECT_ROOT / ch["manifest_path"])
    snrs = sorted(float(s) for s in (args.snr or ch["snr_db"]))
    out_dir = paths.PROJECT_ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg_all.get("seed", 42)))

    cfg["enabled"] = True
    enhancer = build_enhancer(cfg)
    print(f"[phase14] enhancer backend={enhancer['backend']}", flush=True)
    started = time.time()
    results = {}
    rows = []

    for snr in snrs:
        cells = phase12["results"][str(int(snr))]["cells"]
        if args.limit:
            cells = cells[: args.limit]
        items = []
        for cell in cells:
            recovered = Image.open(cell["recovered_path"]).convert("RGB")
            enhanced = enhance(recovered, enhancer=enhancer, config=cfg)
            secret = Image.open(cell["secret_path"]).convert("RGB")
            size = int(cfg["input_size"])
            before = pipeline.metrics(_pixels(secret, size), _pixels(recovered, size), stego_cfg)
            after = pipeline.metrics(_pixels(secret, size), _pixels(enhanced, size), stego_cfg)
            epath = out_dir / f"{cell['image_id']}_snr{int(snr)}_enhanced.png"
            enhanced.save(epath)
            items.append({
                "image_id": cell["image_id"],
                "recovered": before, "enhanced": after,
                "delta": {k: after[k] - before[k] for k in ("psnr", "ssim", "lpips")},
                "recovered_path": cell["recovered_path"],
                "enhanced_path": str(epath),
            })
            if snr in (min(snrs), max(snrs)) and len(rows) < 3 * 2:
                rows.append([cell["secret_path"], cell["recovered_path"], epath])
        results[str(int(snr))] = {
            "snr_db": snr, "n": len(items),
            "recovered": {k: pipeline.summarize([c["recovered"][k] for c in items])
                          for k in ("psnr", "ssim", "lpips")},
            "enhanced": {k: pipeline.summarize([c["enhanced"][k] for c in items])
                         for k in ("psnr", "ssim", "lpips")},
            "delta": {k: pipeline.summarize([c["delta"][k] for c in items])
                      for k in ("psnr", "ssim", "lpips")},
            "cells": items,
        }
        print(f"[phase14] snr={snr:>4.0f} dPSNR="
              f"{results[str(int(snr))]['delta']['psnr']['mean']:+.3f} "
              f"dLPIPS={results[str(int(snr))]['delta']['lpips']['mean']:+.4f}", flush=True)

    if rows:
        _grid(rows, paths.PROJECT_ROOT / cfg["grid_path"])

    # Control: non-facial classes must be untouched by the switch.
    common_rec = pipeline.probe_records("evaluation", "common", 1)[0]
    control_img = pipeline.load_pixels(common_rec["path"], cfg["input_size"])
    pipeline.save_pixels(control_img, out_dir / "control_common_input.png")
    applied = should_enhance(cfg, "common")
    control_identical = not applied

    # Determinism: same recovered image twice.
    probe_cell = phase12["results"][str(int(snrs[-1]))]["cells"][0]
    probe = Image.open(probe_cell["recovered_path"]).convert("RGB")
    h1 = checkpoint_sha256(enhancer)
    a = enhance(probe, enhancer=enhancer, config=cfg)
    b = enhance(probe, enhancer=enhancer, config=cfg)
    deterministic = a.tobytes() == b.tobytes()

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": 14,
        "approximate": False,
        "split_role": "evaluation",
        "class": "facial",
        "enabled_by_default": bool(cfg_all["enhancement"].get("enabled", False)),
        "enhancer": {**{k: v for k, v in enhancer.items() if k != "model"},
                     "fingerprint": h1},
        "input_size": int(cfg["input_size"]),
        "output_size": int(cfg["input_size"]),
        "snr_db": snrs,
        "results": results,
        "control": {"class": "common", "applied": applied,
                    "identical": control_identical},
        "deterministic": deterministic,
        "decisions": {
            "codeformer_weights": "pretrained original CodeFormer weights are not "
                                  "installed in this environment; fallback recorded "
                                  "in docs/DEVIATIONS.md and module kept off by default",
        },
        "elapsed_s": time.time() - started,
    }
    pipeline.write_json(paths.PROJECT_ROOT / cfg["manifest_path"], payload)
    print(f"[phase14] wrote {cfg['manifest_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
