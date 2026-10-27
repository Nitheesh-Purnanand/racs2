"""Phase 13 — semantic codec isolation test and raw-vs-codec comparison."""

from __future__ import annotations

import argparse
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
from communication.semantic_codec import build_codec, codec_roundtrip  # noqa: E402
from steganography import pipeline  # noqa: E402
from steganography.recover import recover  # noqa: E402


def _plot(isolated: dict, comparison: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    snrs = sorted(isolated, key=int, reverse=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([int(s) for s in snrs], [isolated[s]["psnr"]["mean"] for s in snrs],
            "o-", label="codec isolated")
    ax.plot([int(s) for s in snrs], [comparison[s]["codec"]["psnr"]["mean"] for s in snrs],
            "s--", label="codec + stego recovery")
    ax.plot([int(s) for s in snrs], [comparison[s]["raw"]["psnr"]["mean"] for s in snrs],
            "^:", label="raw AWGN + recovery")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _grid(rows, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    tiles = [[Image.open(p).convert("RGB") for p in row] for row in rows]
    w, h = tiles[0][0].size
    sheet = Image.new("RGB", (len(tiles[0]) * w, len(tiles) * h), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(tiles):
        for c, img in enumerate(row):
            sheet.paste(img.resize((w, h), Image.BICUBIC), (c * w, r * h))
    for c, label in enumerate(("clean", "codec @20dB", "codec @5dB")):
        draw.text((c * w + 4, 2), label, fill="black")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg_all = pipeline.load_config()
    cfg = dict(cfg_all["codec"])
    cfg["evaluation"] = cfg_all.get("evaluation", {})
    ch = cfg_all["channel"]
    stego_cfg = dict(cfg_all["steganography"])
    stego_cfg["evaluation"] = cfg_all.get("evaluation", {})
    phase12 = pipeline.read_json(paths.PROJECT_ROOT / ch["manifest_path"])
    cells = pipeline.probe_records("evaluation", stego_cfg.get("probe_class", "facial"),
                                   int(args.limit or stego_cfg["num_images"]))
    snrs = [float(s) for s in ch["snr_db"]]
    base_seed = int(ch["seed"])
    out_dir = paths.PROJECT_ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg_all.get("seed", 42)))

    from steganography.edict import EDICT

    edict = EDICT(latent_dtype=torch.float32)
    codec = build_codec(cfg, edict)
    print(f"[phase13] codec kind={codec['kind']} sha256={codec['sha256'][:16]}", flush=True)
    started = time.time()

    # 1) Isolated codec reconstruction on clean images.
    isolated = {}
    grid_rows = []
    for idx, snr in enumerate(snrs):
        items = []
        for rec in cells:
            secret = pipeline.load_pixels(rec["path"], cfg["input_size"])
            out = codec_roundtrip(secret, snr_db=snr, seed=base_seed + idx,
                                  codec=codec, config=cfg)
            items.append({"image_id": rec["image_id"],
                          "metrics": pipeline.metrics(secret, out["output"], stego_cfg),
                          "latency_s": out["latency_s"],
                          "symbols_shape": out["symbols_shape"]})
        isolated[str(int(snr))] = {
            "snr_db": snr, "seed": base_seed + idx, "n": len(items),
            "distribution": {k: pipeline.summarize([it["metrics"][k] for it in items])
                             for k in ("mse", "psnr", "ssim", "lpips")},
            "cells": items,
        }
        print(f"[phase13] isolated snr={snr:>4.0f} psnr="
              f"{isolated[str(int(snr))]['distribution']['psnr']['mean']:.2f}", flush=True)

    for rec in cells[:3]:
        secret = pipeline.load_pixels(rec["path"], cfg["input_size"])
        for snr in (20.0, 5.0):
            idx = snrs.index(snr)
            out = codec_roundtrip(secret, snr_db=snr, seed=base_seed + idx,
                                  codec=codec, config=cfg)
            pipeline.save_pixels(out["output"], out_dir / f"{rec['image_id']}_codec_{int(snr)}.png")
        grid_rows.append([rec["path"],
                          out_dir / f"{rec['image_id']}_codec_20.png",
                          out_dir / f"{rec['image_id']}_codec_5.png"])
    if grid_rows:
        _grid(grid_rows, paths.PROJECT_ROOT / cfg["grid_path"])

    # 2) Raw AWGN vs codec + AWGN through the full stego path.
    phase11 = pipeline.read_json(paths.PROJECT_ROOT / stego_cfg["manifest_path"])
    p11_cells = phase11.get("conditions", {}).get("matched", {}).get("cells", [])
    comparison = {}
    for idx, snr in enumerate(snrs):
        codec_items = []
        for cell in p11_cells:
            blob_path = (paths.PROJECT_ROOT / stego_cfg["results_dir"] /
                         f"{cell['image_id']}__matched_latents.pt")
            blob = torch.load(blob_path, map_location="cpu", weights_only=False)
            stego = pipeline.decode(edict, pipeline.mean_pair(blob["z_stego_pair"]))
            received = codec_roundtrip(stego, snr_db=snr, seed=base_seed + idx,
                                       codec=codec, config=cfg)["output"]
            recovered = recover(edict, received, blob["token"],
                                blob.get("recover_prompt") or blob["prompt"], stego_cfg,
                                delta=blob["delta"], eta=blob["eta"],
                                steps=blob["steps"], mixing=blob["mixing"],
                                guidance=stego_cfg.get("guidance_scale", 1.0))
            secret = pipeline.load_pixels(blob["secret_path"], stego_cfg["image_size"])
            codec_items.append({"image_id": cell["image_id"],
                                "metrics": pipeline.metrics(secret, recovered, stego_cfg)})
        codec_dist = {k: pipeline.summarize([it["metrics"][k] for it in codec_items])
                      for k in ("mse", "psnr", "ssim", "lpips")}
        raw_dist = phase12["results"][str(int(snr))]["distribution"]
        comparison[str(int(snr))] = {
            "snr_db": snr, "n": len(codec_items),
            "raw": raw_dist, "codec": codec_dist,
            "delta_psnr": codec_dist["psnr"]["mean"] - raw_dist["psnr"]["mean"],
            "delta_ssim": codec_dist["ssim"]["mean"] - raw_dist["ssim"]["mean"],
            "delta_lpips": codec_dist["lpips"]["mean"] - raw_dist["lpips"]["mean"],
            "cells": codec_items,
        }
        print(f"[phase13] comparison snr={snr:>4.0f} raw="
              f"{raw_dist['psnr']['mean']:.2f} codec={codec_dist['psnr']['mean']:.2f}",
              flush=True)

    _plot({k: v["distribution"] for k, v in isolated.items()}, comparison,
          paths.PROJECT_ROOT / cfg["curve_path"])

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": 13,
        "approximate": False,
        "split_role": "evaluation",
        "codec": {"kind": codec["kind"], "model_id": codec["model_id"],
                  "sha256": codec["sha256"],
                  "compression_ratio": codec["compression_ratio"],
                  "trained_snr_db": codec["trained_snr_db"],
                  "frozen": True},
        "normalization": "pixels [0,1]; encoder input [-1,1]; per-call power "
                         "normalisation by sqrt(mean(z^2)), inverted exactly",
        "snr_db": snrs,
        "isolated": isolated,
        "comparison": comparison,
        "decisions": {
            "codec_substitute": "paper SwinJSCC/MDJCM weights unavailable; SD VAE "
                                "used as a frozen analog JSCC substitute",
        },
        "elapsed_s": time.time() - started,
    }
    pipeline.write_json(paths.PROJECT_ROOT / cfg["manifest_path"], payload)
    print(f"[phase13] wrote {cfg['manifest_path']}", flush=True)
    edict.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
