"""
Phase 7 — Build Public Conditions (Kpub + Xfeat) for all split images.

Usage:
    python scripts/phase7_build_conditions.py
    python scripts/phase7_build_conditions.py --split evaluation
    python scripts/phase7_build_conditions.py --limit 2 --force
    python scripts/phase7_build_conditions.py --method clip_vocab
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Project bootstrap ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # side effect: HF cache redirect
import common.hf_lowmem

common.hf_lowmem.install()

import numpy as np
import torch
import yaml
from PIL import Image

from common.seed import set_seed
from evaluation.splits import Split, load_tokens, sha256_file, sha256_json, CLASSES
from semantic.keygen import build_keygen, validate_kpub
from semantic.features import build_feature_extractor
from semantic.cache import (
    DEFAULT_CACHE_DIR,
    build_manifest,
    build_manifest_record,
    compute_config_hash,
    feature_sha256,
    sha256_string,
    write_cache_npz,
    write_manifest,
    write_preview,
)


def make_contact_sheet(
    records_with_images: list,
    output_path: Path,
    cols_per_class: int = 3,
    thumb: int = 128,
):
    """Build a contact sheet: rows = (class, source | preview), cols = sample images."""
    from PIL import ImageDraw, ImageFont

    rows_data = defaultdict(list)
    for rec in records_with_images:
        rows_data[(rec["role"], rec["class"])].append(rec)

    sorted_keys = sorted(rows_data.keys())
    nrows = len(sorted_keys)
    if nrows == 0:
        return

    # Each row: label + source + preview (2 thumbnails per sample)
    label_w = 120
    ncols = min(cols_per_class, max(len(v) for v in rows_data.values()))
    img_w = label_w + ncols * (thumb * 2 + 10) + 10
    img_h = nrows * (thumb + 30) + 10

    sheet = Image.new("RGB", (img_w, img_h), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)

    for row_idx, key in enumerate(sorted_keys):
        role, cls = key
        y = row_idx * (thumb + 30) + 10
        label = f"{role[:4]}/{cls}"
        draw.text((5, y + thumb // 2), label, fill=(255, 255, 255))

        items = rows_data[key][:ncols]
        for col_idx, item in enumerate(items):
            x = label_w + col_idx * (thumb * 2 + 10)

            # Source thumbnail
            src = item.get("source_image")
            if src is not None:
                src_thumb = src.resize((thumb, thumb), Image.BICUBIC)
                sheet.paste(src_thumb, (x, y))

            # Preview thumbnail
            prev = item.get("preview_image")
            if prev is not None:
                prev_thumb = prev.resize((thumb, thumb), Image.BICUBIC)
                sheet.paste(prev_thumb, (x + thumb + 2, y))
            else:
                # For style: write "Kpub only"
                draw.text((x + thumb + 10, y + thumb // 2), "Kpub\nonly",
                          fill=(200, 200, 0))

            # Image ID
            draw.text((x, y + thumb + 2), item.get("image_id", "")[:20],
                      fill=(180, 180, 180))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output_path), format="PNG")
    print(f"[contact sheet] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7: build public conditions")
    parser.add_argument("--split", choices=["calibration", "evaluation"],
                        help="build only one role (default: both)")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit images per class per role (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if cache exists")
    parser.add_argument("--method", default=None,
                        help="override keygen method")
    args = parser.parse_args()

    # ── 1. Seed ──────────────────────────────────────────────────────────────
    set_seed(42, verbose=True)

    # ── 2. Config ────────────────────────────────────────────────────────────
    config_path = paths.PROJECT_ROOT / "configs" / "base.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    semantic_cfg = config.get("semantic", {})
    keygen_cfg = dict(semantic_cfg.get("keygen", {}))
    features_cfg = dict(semantic_cfg.get("features", {}))

    if args.method:
        keygen_cfg["method"] = args.method

    cache_dir = Path(semantic_cfg.get("cache_dir", "experiments/results/public_conditions"))
    if not cache_dir.is_absolute():
        cache_dir = paths.PROJECT_ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(semantic_cfg.get(
        "manifest_path", "experiments/results/public_conditions/manifest.json"
    ))
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    config_hash = compute_config_hash(semantic_cfg)
    print(f"[config] config_hash = {config_hash}")

    # ── 3. Load splits ───────────────────────────────────────────────────────
    roles_to_build = []
    if args.split:
        roles_to_build = [args.split]
    else:
        roles_to_build = ["calibration", "evaluation"]

    splits = {}
    for role in roles_to_build:
        splits[role] = Split.load(
            role=role,
            allow_calibration=(role == "calibration"),
            verify=True,
        )
        print(f"[split] {role}: {len(splits[role])} records loaded")

    split_version = splits[roles_to_build[0]].version

    # Load corpus_version from image_index.json
    with open(paths.IMAGE_INDEX_PATH, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    corpus_version = index.get("corpus_version", "")

    # Load tokens
    tokens_all, tokens_version = load_tokens()

    # ── 4. Build backends ────────────────────────────────────────────────────
    print(f"[keygen] method = {keygen_cfg.get('method', 'blip_template')}")
    keygen = build_keygen(keygen_cfg)
    print(f"[keygen] model loaded, weight_sha256 = {keygen.weight_sha256}")

    # Build feature extractors per class (reuse across images)
    extractors = {}
    for cls in CLASSES:
        extractors[cls] = build_feature_extractor(features_cfg, cls)
        print(f"[features] {cls}: method = {extractors[cls].method}, "
              f"model_id = {extractors[cls].model_id}")

    # ── 5. Process all images ────────────────────────────────────────────────
    manifest_records = []
    contact_sheet_data = []
    counts = defaultdict(lambda: defaultdict(int))
    failures = []
    total_processed = 0
    t_start = time.time()

    for role in roles_to_build:
        split = splits[role]
        for cls in CLASSES:
            records = split.records(cls)
            if args.limit > 0:
                records = records[:args.limit]

            for record in records:
                image_id = record["image_id"]
                source_path = paths.PROJECT_ROOT / record["path"]
                source_sha = record["sha256"]

                # Verify source
                actual_sha = sha256_file(source_path)
                if actual_sha != source_sha:
                    msg = f"SHA mismatch for {image_id}: expected {source_sha}, got {actual_sha}"
                    print(f"[ERROR] {msg}")
                    failures.append({"image_id": image_id, "error": msg})
                    continue

                # Check cache
                cache_path = cache_dir / f"{image_id}.npz"
                preview_path = cache_dir / f"{image_id}_preview.png"

                if not args.force and cache_path.exists():
                    # Verify config hash matches
                    try:
                        cached = np.load(str(cache_path), allow_pickle=False)
                        if str(cached.get("config_hash", "")) == config_hash:
                            # Cache is valid, read the record for manifest
                            kpub = str(cached["kpub"])
                            feature_kind = str(cached["feature_kind"])
                            feat = cached.get("feature")
                            feat_sha = feature_sha256(feat) if feat is not None else None
                            kpub_sha = sha256_string(kpub)
                            
                            rel_cache = str(cache_path.relative_to(paths.PROJECT_ROOT))
                            rel_preview = str(preview_path.relative_to(paths.PROJECT_ROOT)) \
                                if preview_path.exists() else None

                            manifest_records.append(build_manifest_record(
                                image_id=image_id,
                                role=role,
                                class_name=cls,
                                source_sha256=source_sha,
                                cache_path=rel_cache,
                                preview_path=rel_preview,
                                kpub_sha256=kpub_sha,
                                feature_sha256_val=feat_sha,
                                feature_kind=feature_kind,
                                status="ok",
                            ))
                            counts[role][cls] += 1
                            total_processed += 1
                            continue
                    except Exception:
                        pass  # cache invalid, rebuild

                # Load image
                try:
                    image = Image.open(source_path).convert("RGB")
                    image = image.resize((512, 512), Image.BICUBIC)
                except Exception as exc:
                    msg = f"Failed to load {image_id}: {exc}"
                    print(f"[ERROR] {msg}")
                    failures.append({"image_id": image_id, "error": msg})
                    continue

                # Generate Kpub
                try:
                    token_info = tokens_all.get(image_id, {})
                    context = {
                        "image_id": image_id,
                        "role": role,
                        "class": cls,
                    }
                    kpub_result = keygen.generate_kpub(image, context=context)
                    kpub = kpub_result["kpub"]

                    # Validate
                    kpub_errors = validate_kpub(
                        kpub,
                        max_chars=keygen_cfg.get("max_chars", 300),
                        image_id=image_id,
                        token=token_info.get("token", ""),
                        wrong_token=token_info.get("wrong_token", ""),
                        path=str(record["path"]),
                        sha256=source_sha,
                    )
                    if kpub_errors:
                        msg = f"Kpub validation failed for {image_id}: {kpub_errors}"
                        print(f"[ERROR] {msg}")
                        failures.append({"image_id": image_id, "error": msg})
                        continue
                except Exception as exc:
                    msg = f"Keygen failed for {image_id}: {exc}"
                    print(f"[ERROR] {msg}")
                    failures.append({"image_id": image_id, "error": msg})
                    continue

                # Extract feature
                try:
                    feat_result = extractors[cls].extract(image)
                    feature_kind = feat_result["feature_kind"]
                    feat_array = feat_result.get("feature")
                    feat_preview = feat_result.get("preview")
                    feat_status = feat_result.get("status", "ok")

                    if feat_status == "no_face" and cls == "facial":
                        msg = f"No face detected in {image_id}"
                        print(f"[WARN] {msg}")
                        failures.append({"image_id": image_id, "error": msg})
                        continue
                except Exception as exc:
                    msg = f"Feature extraction failed for {image_id}: {exc}"
                    print(f"[ERROR] {msg}")
                    failures.append({"image_id": image_id, "error": msg})
                    continue

                # Write cache
                try:
                    write_cache_npz(
                        cache_path,
                        image_id=image_id,
                        role=role,
                        class_name=cls,
                        source_sha256=source_sha,
                        kpub=kpub,
                        kpub_method=kpub_result["method"],
                        kpub_model_id=kpub_result["model_id"],
                        feature_kind=feature_kind,
                        feature=feat_array,
                        config_hash=config_hash,
                    )

                    # Write preview
                    rel_preview = None
                    if feat_preview is not None and cls in ("common", "facial"):
                        write_preview(preview_path, feat_preview)
                        rel_preview = str(preview_path.relative_to(paths.PROJECT_ROOT))
                except Exception as exc:
                    msg = f"Cache write failed for {image_id}: {exc}"
                    print(f"[ERROR] {msg}")
                    failures.append({"image_id": image_id, "error": msg})
                    continue

                # Build manifest record
                feat_sha = feature_sha256(feat_array)
                kpub_sha = sha256_string(kpub)
                rel_cache = str(cache_path.relative_to(paths.PROJECT_ROOT))

                manifest_records.append(build_manifest_record(
                    image_id=image_id,
                    role=role,
                    class_name=cls,
                    source_sha256=source_sha,
                    cache_path=rel_cache,
                    preview_path=rel_preview,
                    kpub_sha256=kpub_sha,
                    feature_sha256_val=feat_sha,
                    feature_kind=feature_kind,
                    status="ok",
                ))
                counts[role][cls] += 1
                total_processed += 1

                # Collect for contact sheet
                contact_sheet_data.append({
                    "image_id": image_id,
                    "role": role,
                    "class": cls,
                    "source_image": image.copy(),
                    "preview_image": feat_preview,
                })

                # Progress
                if total_processed % 10 == 0:
                    print(f"  [{total_processed}] {image_id} ({cls}) kpub={kpub[:60]}...")

    elapsed = time.time() - t_start

    # ── 6. Write manifest ────────────────────────────────────────────────────
    models_info = {
        "keygen": {
            "method": keygen_cfg.get("method", "blip_template"),
            "model_id": keygen_cfg.get("model_id", ""),
            "weight_sha256": keygen.weight_sha256,
        },
    }
    for cls in CLASSES:
        ext = extractors[cls]
        models_info[cls] = {
            "method": ext.method,
            "model_id": ext.model_id,
            "weight_sha256": ext.weight_sha256,
        }

    manifest = build_manifest(
        records=manifest_records,
        split_version=split_version,
        corpus_version=corpus_version,
        config_hash=config_hash,
        models=models_info,
        counts={role: dict(cls_counts) for role, cls_counts in counts.items()},
    )
    write_manifest(manifest, manifest_path)
    print(f"\n[manifest] written to {manifest_path}")

    # ── 7. Contact sheet ─────────────────────────────────────────────────────
    grid_path = paths.RESULTS_DIR / "phase7_conditions_grid.png"
    if contact_sheet_data:
        make_contact_sheet(contact_sheet_data, grid_path, cols_per_class=3, thumb=128)

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 7 Build Summary")
    print(f"{'='*60}")
    for role in roles_to_build:
        role_counts = counts.get(role, {})
        parts = [f"{cls}={role_counts.get(cls, 0)}" for cls in CLASSES]
        print(f"  {role:15s}  {', '.join(parts)}")
    print(f"  total processed: {total_processed}")
    print(f"  failed:          {len(failures)}")
    print(f"  config_hash:     {config_hash}")
    print(f"  split_version:   {split_version}")
    print(f"  elapsed:         {elapsed:.1f}s")

    if failures:
        print(f"\n[FAILURES]")
        for f in failures:
            print(f"  {f['image_id']}: {f['error']}")

    # ── 9. Cleanup ───────────────────────────────────────────────────────────
    keygen.unload()
    for ext in extractors.values():
        ext.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if failures:
        print(f"\n[EXIT] {len(failures)} failures — exit 1")
        sys.exit(1)
    else:
        print(f"\n[EXIT] all records OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
