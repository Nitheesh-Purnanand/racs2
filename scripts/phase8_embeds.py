"""Phase 8b — precompute the IPAdapter-plus embedding of every reference.

IPAdapter-plus conditions the EDICT sampler on ``Xref``. Its image encoder is
CLIP-H, a 2.5 GB checkpoint that cannot sit on the 6 GB card beside Stable
Diffusion, and keeping it in host memory next to SwinJSCC and CodeFormer
exhausts the Windows commit budget. So the embeddings are computed once here,
cached by the reference file's SHA-256, and read back by Phases 11, 15 and 16.

Both ends read the same cached tensor, which makes the transmitter and receiver
conditioning byte-identical by construction.

    .venv\\Scripts\\python.exe scripts\\phase8_embeds.py
    .venv\\Scripts\\python.exe scripts\\phase8_embeds.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
import yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="recompute even if a cache entry exists")
    args = parser.parse_args()

    config = yaml.safe_load(
        (paths.PROJECT_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    stego_cfg = config["steganography"]
    ip_cfg = dict(stego_cfg.get("ipadapter") or {})
    if not ip_cfg.get("enabled"):
        print("[phase8b] ipadapter disabled in config; nothing to do")
        return 0

    manifest_path = paths.PROJECT_ROOT / config["reference"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if not records:
        print(f"[phase8b] no reference records in {manifest_path}")
        return 1

    import torch
    from steganography.edict import EDICT
    from steganography.ipadapter import (
        build_ipadapter,
        cache_stats,
        reference_embeds_for_path,
        unload_ipadapter,
    )

    cache_dir = paths.PROJECT_ROOT / ip_cfg.get(
        "cache_dir", "experiments/results/ipadapter_embeds")
    if args.force and cache_dir.exists():
        removed = 0
        for stale in cache_dir.glob("*.npz"):
            stale.unlink()
            removed += 1
        print(f"[phase8b] cleared {removed} cached embeddings")

    # The adapter must be attached to a UNet to know its projection shape, so a
    # UNet is loaded, but no sampling happens here.
    edict = EDICT(latent_dtype=torch.float32)
    handle = build_ipadapter(
        edict.unet,
        root=ip_cfg.get("root", "models/ip-adapter"),
        subfolder=ip_cfg.get("subfolder", "models"),
        weight_name=ip_cfg.get("weight_name", "ip-adapter-plus_sd15.safetensors"),
        image_encoder_folder=ip_cfg.get("image_encoder_folder", "models/image_encoder"),
        scale=float(ip_cfg.get("scale", 1.0)),
        device=edict.device,
        dtype=torch.float16,
        encoder_device=ip_cfg.get("encoder_device", "cpu"),
        encoder_dtype=torch.float16 if str(
            ip_cfg.get("encoder_dtype", "float32")).lower() in ("float16", "fp16")
        else torch.float32,
        cache_dir=cache_dir,
        load_encoder=False,
    )

    guidance = float(stego_cfg.get("guidance_scale", 1.0))
    started = time.time()
    failures = []
    for index, record in enumerate(records, 1):
        reference_path = paths.PROJECT_ROOT / record["reference_path"]
        if not reference_path.exists():
            failures.append({"image_id": record["image_id"], "error": "missing reference"})
            continue
        try:
            reference_embeds_for_path(handle, reference_path, guidance_scale=guidance)
        except Exception as exc:  # noqa: BLE001
            failures.append({"image_id": record["image_id"], "error": repr(exc)})
            continue
        if index % 20 == 0 or index == len(records):
            stats = cache_stats(handle)
            print(f"[phase8b] {index}/{len(records)} hits={stats['hits']} "
                  f"misses={stats['misses']}", flush=True)

    stats = cache_stats(handle)
    unload_ipadapter(handle)
    edict.unload()

    payload = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "references": len(records),
        "cached_files": len(list(cache_dir.glob("*.npz"))),
        "hits": stats["hits"],
        "misses": stats["misses"],
        "guidance_scale": guidance,
        "weight_name": ip_cfg.get("weight_name"),
        "weight_sha256": handle["weight_sha256"],
        "image_encoder_folder": ip_cfg.get("image_encoder_folder"),
        "cache_dir": str(cache_dir.relative_to(paths.PROJECT_ROOT)),
        "elapsed_s": round(time.time() - started, 1),
        "failures": failures,
    }
    out_path = paths.RESULTS_DIR / "phase8_ipadapter_embeds.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[phase8b] wrote {out_path}")
    print(f"[phase8b] {payload['cached_files']} cached embeddings, "
          f"{len(failures)} failures, {payload['elapsed_s']}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
