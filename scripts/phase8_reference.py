"""Phase 8 — build token-seeded references for every Phase 7 condition record."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths
import common.hf_lowmem

common.hf_lowmem.install()

import torch
import yaml
from PIL import Image, ImageDraw

from common.seed import set_seed
from evaluation.splits import CLASSES, Split, load_tokens, sha256_file
from reference.conditions import control_image, load_conditions
from reference.generate import (
    _fingerprint,
    build_pipeline,
    generate_reference,
    unload,
)
from reference.seed import derive_seed
from semantic.cache import feature_sha256, read_cache_npz, sha256_string

TOKEN_MODES = ("correct", "wrong")


def _atomic_save(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)
    return sha256_file(path)


def _contact_sheet(rows: List[Dict[str, Any]], out_path: Path, thumb: int = 160) -> None:
    if not rows:
        return
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(row["role"], row["class"])].append(row)
    keys = sorted(by_cell)
    cols = max(len(v) for v in by_cell.values())
    width = 130 + cols * (thumb * 2 + 8)
    height = len(keys) * (thumb + 22) + 8
    sheet = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for r, key in enumerate(keys):
        role, cls = key
        y = r * (thumb + 22) + 4
        draw.text((4, y + thumb // 2), f"{role[:4]}/{cls}", fill=(255, 255, 255))
        for c, item in enumerate(by_cell[key][:cols]):
            x = 130 + c * (thumb * 2 + 8)
            if item.get("source") is not None:
                sheet.paste(item["source"].resize((thumb, thumb), Image.BICUBIC), (x, y))
            if item.get("reference") is not None:
                sheet.paste(item["reference"].resize((thumb, thumb), Image.BICUBIC),
                            (x + thumb + 2, y))
            draw.text((x, y + thumb + 2), item["image_id"][:22], fill=(190, 190, 190))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8: build references")
    parser.add_argument("--split", choices=["calibration", "evaluation"])
    parser.add_argument("--class", dest="class_name", choices=list(CLASSES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token-mode", choices=TOKEN_MODES, default="correct")
    parser.add_argument("--ids", default=None,
                        help="comma-separated image_ids to rebuild (patches the "
                             "existing manifest instead of rewriting it)")
    args = parser.parse_args()

    target_ids = None
    if args.ids:
        target_ids = {part.strip() for part in args.ids.split(",") if part.strip()}
        if not target_ids:
            parser.error("--ids was given but no image ids were parsed")

    set_seed(42, verbose=True)
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = config.get("reference", {})
    out_dir = paths.PROJECT_ROOT / cfg.get("cache_dir", "experiments/results/references")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.PROJECT_ROOT / cfg.get(
        "manifest_path", "experiments/results/phase8_references.json"
    )
    size = int(cfg.get("output_size", 256))
    steps = int(cfg.get("steps", 20))
    guidance = float(cfg.get("guidance_scale", 7.5))

    roles = [args.split] if args.split else ["calibration", "evaluation"]
    classes = [args.class_name] if args.class_name else list(CLASSES)
    splits = {
        role: Split.load(role=role, allow_calibration=(role == "calibration"))
        for role in roles
    }
    tokens, tokens_version = load_tokens()
    conditions_dir = paths.RESULTS_DIR / "public_conditions"

    # Provenance of the previous run: a cached reference may only be reused if
    # the public condition it was generated from is still identical.
    previous = {}
    previous_manifest = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            previous_manifest = json.load(fh)
        previous = {r["image_id"]: r for r in previous_manifest.get("records", [])}

    records: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    models: Dict[str, Any] = {}
    t_all = time.time()
    total = 0

    for cls in classes:
        print(f"[phase8] loading pipeline for {cls} ...", flush=True)
        pipe, controlnet = build_pipeline(cls, cfg)
        models[cls] = {
            "base": cfg.get("model_id"),
            "controlnet": None if controlnet is None else (
                cfg.get("controlnet_id") if cls == "common"
                else cfg.get("facial_controlnet_id", cfg.get("controlnet_id"))
            ),
            "unet_fingerprint": _fingerprint(pipe.unet),
            "controlnet_fingerprint": None if controlnet is None else _fingerprint(controlnet),
            # which checkpoint each component actually came from (PicX-Real, or
            # the SD v1.5 fallback for anything PicX-Real does not ship)
            "sources": getattr(pipe, "racs_sources", None),
        }
        try:
            for role in roles:
                split = splits[role]
                recs = split.records(cls)
                if args.limit:
                    recs = recs[: args.limit]
                for rec in recs:
                    image_id = rec["image_id"]
                    if target_ids is not None and image_id not in target_ids:
                        continue
                    src_path = paths.PROJECT_ROOT / rec["path"]
                    if sha256_file(src_path) != rec["sha256"]:
                        failures.append({"image_id": image_id, "error": "source sha256 mismatch"})
                        continue
                    cond = load_conditions(conditions_dir, image_id)
                    cond_kpub_sha = sha256_string(cond["kpub"])
                    cond_feat_sha = feature_sha256(cond.get("feature"))
                    token = tokens[image_id][
                        "token" if args.token_mode == "correct" else "wrong_token"
                    ]
                    ref_path = out_dir / f"{image_id}__{token}.png"
                    prev = previous.get(image_id, {})
                    provenance_ok = (
                        prev.get("token_mode") == args.token_mode
                        and prev.get("kpub_sha256") == cond_kpub_sha
                        and prev.get("feature_sha256") == cond_feat_sha
                        and prev.get("reference_path")
                        == str(ref_path.relative_to(paths.PROJECT_ROOT))
                    )
                    if ref_path.exists() and not args.force and provenance_ok:
                        ref_sha = sha256_file(ref_path)
                        status = "cached"
                        seed = derive_seed(token)
                        generator = torch.Generator(device="cuda").manual_seed(seed)
                        probe = torch.randn(
                            (1, 4, size // 8, size // 8),
                            generator=generator, device="cuda", dtype=torch.float16,
                        )
                        latents_sha = __import__("hashlib").sha256(
                            probe.detach().float().cpu().numpy().tobytes()
                        ).hexdigest()
                        peak = None
                        latency = None
                    else:
                        try:
                            result = generate_reference(
                                pipe,
                                prompt=cond["kpub"],
                                control=control_image(cond, size),
                                token=token,
                                output_size=size,
                                steps=steps,
                                guidance_scale=guidance,
                            )
                            ref_sha = _atomic_save(result["image"], ref_path)
                            status = "ok"
                            seed = result["seed"]
                            latents_sha = result["latents_sha256"]
                            peak = result["peak_vram_gb"]
                            latency = result["latency_s"]
                        except Exception as exc:  # noqa: BLE001
                            failures.append({"image_id": image_id, "error": repr(exc)})
                            print(f"[ERROR] {image_id}: {exc}", flush=True)
                            continue
                    records.append({
                        "image_id": image_id,
                        "role": role,
                        "class": cls,
                        "token_mode": args.token_mode,
                        "seed": seed,
                        "kpub_sha256": cond_kpub_sha,
                        "feature_sha256": cond_feat_sha,
                        "reference_path": str(ref_path.relative_to(paths.PROJECT_ROOT)),
                        "reference_sha256": ref_sha,
                        "latents_sha256": latents_sha,
                        "peak_vram_gb": peak,
                        "latency_s": latency,
                        "status": status,
                    })
                    total += 1
                    if total % 10 == 0:
                        print(f"  [{total}] {image_id} ({role}/{cls}) {status}", flush=True)
                    if args.limit == 0 and len(grid_rows) < 36:
                        ref_img = Image.open(ref_path).convert("RGB")
                        grid_rows.append({
                            "image_id": image_id,
                            "role": role,
                            "class": cls,
                            "source": Image.open(src_path).convert("RGB"),
                            "reference": ref_img,
                        })
        finally:
            unload(pipe)
            pipe = None
            controlnet = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_version": splits[roles[0]].version,
        "tokens_version": tokens_version,
        "config_hash": sha256_string(json.dumps(cfg, sort_keys=True)),
        "backend": cfg.get("backend", "sd_controlnet"),
        "output_size": size,
        "steps": steps,
        "guidance_scale": guidance,
        "eta": cfg.get("eta", 0.0),
        "models": models,
        "counts": {
            role: {cls: sum(1 for r in records if r["role"] == role and r["class"] == cls)
                   for cls in classes}
            for role in roles
        },
        "records": sorted(records, key=lambda r: (r["role"], r["class"], r["image_id"])),
        "failures": failures,
    }

    if target_ids is not None:
        if not manifest_path.exists():
            print(f"[ERROR] --ids requires an existing manifest at {manifest_path}")
            sys.exit(1)
        existing = previous_manifest
        replaced = {r["image_id"] for r in records}
        merged = [r for r in existing.get("records", [])
                  if r["image_id"] not in replaced]
        merged.extend(records)
        merged_counts = defaultdict(lambda: defaultdict(int))
        for rec in merged:
            merged_counts[rec["role"]][rec["class"]] += 1
        existing["records"] = sorted(
            merged, key=lambda r: (r["role"], r["class"], r["image_id"])
        )
        existing["counts"] = {
            role: dict(cls_counts) for role, cls_counts in merged_counts.items()
        }
        existing["split_version"] = manifest["split_version"]
        existing["tokens_version"] = manifest["tokens_version"]
        existing.setdefault("models", {}).update(manifest["models"])
        existing["failures"] = failures
        manifest = existing
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, manifest_path)
    print(f"[phase8] manifest -> {manifest_path}", flush=True)

    if grid_rows and target_ids is None:
        _contact_sheet(grid_rows, paths.RESULTS_DIR / "phase8_reference_grid.png")
    print(f"[phase8] records={len(records)} failures={len(failures)} "
          f"elapsed={time.time() - t_all:.1f}s", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
