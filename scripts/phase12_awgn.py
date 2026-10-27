"""Phase 12 — stego through AWGN and recovery across the SNR grid."""

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
from communication.awgn import awgn_channel  # noqa: E402
from steganography import pipeline  # noqa: E402
from steganography.recover import recover  # noqa: E402


def _plot(curve: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    snrs = sorted(curve, reverse=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(snrs, [curve[s]["psnr"]["mean"] for s in snrs], "o-", label="PSNR")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("PSNR (dB)")
    ax2 = ax1.twinx()
    ax2.plot(snrs, [curve[s]["lpips"]["mean"] for s in snrs], "s--", color="tab:red",
             label="LPIPS")
    ax2.set_ylabel("LPIPS")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg_all = pipeline.load_config()
    cfg = dict(cfg_all["channel"])
    cfg["evaluation"] = cfg_all.get("evaluation", {})
    stego_cfg = dict(cfg_all["steganography"])
    stego_cfg["evaluation"] = cfg_all.get("evaluation", {})
    phase11 = pipeline.read_json(paths.PROJECT_ROOT / stego_cfg["manifest_path"])
    cells = phase11.get("conditions", {}).get("matched", {}).get("cells", [])
    if args.limit:
        cells = cells[: args.limit]
    if not cells:
        raise RuntimeError("phase11 matched cells missing; run phase11_local.py first")

    set_seed(int(cfg_all.get("seed", 42)))
    out_dir = paths.PROJECT_ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    snrs = [float(s) for s in cfg["snr_db"]]
    base_seed = int(cfg["seed"])
    started = time.time()
    results = {}

    from steganography.edict import EDICT

    edict = EDICT(latent_dtype=torch.float32)
    for idx, snr in enumerate(snrs):
        seed = base_seed + idx
        items = []
        for cell in cells:
            rel = f"{cell['image_id']}__matched_latents.pt"
            blob = torch.load(paths.PROJECT_ROOT / stego_cfg["results_dir"] / rel,
                              map_location="cpu", weights_only=False)
            stego = pipeline.decode(edict, pipeline.mean_pair(blob["z_stego_pair"]))
            received, meta = awgn_channel(stego, snr_db=snr, seed=seed, h=1 + 0j)
            recovered = recover(edict, received, blob["token"],
                                blob.get("recover_prompt") or blob["prompt"], stego_cfg,
                                delta=blob["delta"], eta=blob["eta"],
                                steps=blob["steps"], mixing=blob["mixing"],
                                guidance=stego_cfg.get("guidance_scale", 1.0))
            secret = pipeline.load_pixels(blob["secret_path"], stego_cfg["image_size"])
            stem = f"{cell['image_id']}_snr{int(snr)}"
            pipeline.save_pixels(received, out_dir / f"{stem}_received.png")
            pipeline.save_pixels(recovered, out_dir / f"{stem}_recovered.png")
            pipeline.save_pixels(secret, out_dir / f"{stem}_secret.png")
            items.append({
                "image_id": cell["image_id"],
                "metrics": pipeline.metrics(secret, recovered, stego_cfg),
                "channel": meta,
                "received_sha256": pipeline.tensor_sha256(received),
                "recovered_sha256": pipeline.tensor_sha256(recovered),
                "recovered_path": str(out_dir / f"{stem}_recovered.png"),
                "secret_path": str(out_dir / f"{stem}_secret.png"),
                "received_path": str(out_dir / f"{stem}_received.png"),
            })
            print(f"[phase12] snr={snr:>4.0f} {cell['image_id']} "
                  f"psnr={items[-1]['metrics']['psnr']:.2f}", flush=True)
        results[str(int(snr))] = {
            "snr_db": snr,
            "seed": seed,
            "n": len(items),
            "distribution": {
                key: pipeline.summarize([it["metrics"][key] for it in items])
                for key in ("mse", "psnr", "ssim", "lpips")
            },
            "cells": items,
        }

    ordered = [results[str(int(s))]["distribution"]["psnr"]["mean"] for s in snrs]
    monotone = all(ordered[i] <= ordered[i + 1] + 1e-9 for i in range(len(ordered) - 1))
    curve = {str(int(s)): results[str(int(s))]["distribution"] for s in snrs}
    _plot(curve, paths.PROJECT_ROOT / cfg["curve_path"])

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": 12,
        "approximate": False,
        "split_role": "evaluation",
        "class": phase11.get("class"),
        "model": phase11.get("config"),
        "channel": {k: cfg[k] for k in ("kind", "snr_db", "h", "complex_noise", "seed")},
        "normalization": "pixels [0,1]; received is not clipped before recovery, "
                         "PNG saves are clipped for display only",
        "snr_db": snrs,
        "results": results,
        "no_noise_control": phase11.get("conditions", {}).get("matched", {}).get("distribution"),
        "monotone_psnr": monotone,
        "elapsed_s": time.time() - started,
    }
    pipeline.write_json(paths.PROJECT_ROOT / cfg["manifest_path"], payload)
    print(f"[phase12] wrote {cfg['manifest_path']} monotone={monotone}", flush=True)
    edict.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
