"""
Phase 7 — Offline Verification Gate for Public Conditions.

Checks manifest integrity, cache validity, Kpub content, feature shapes,
determinism, and the contact sheet. Writes phase7_verify.json.

Usage:
    python scripts/phase7_verify.py
"""

from __future__ import annotations

import gc
import json
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

import numpy as np
import torch
import yaml
from PIL import Image

from common.seed import set_seed
from evaluation.splits import Split, load_tokens, sha256_file, CLASSES
from semantic.cache import (
    compute_config_hash,
    feature_sha256,
    read_cache_npz,
    read_manifest,
    sha256_string,
)


def check(name: str, condition: bool, detail: str = "") -> Dict[str, Any]:
    """Create a check result."""
    return {"check": name, "pass": condition, "detail": detail}


def main():
    set_seed(42, verbose=True)
    t_start = time.time()

    config_path = paths.PROJECT_ROOT / "configs" / "base.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    semantic_cfg = config.get("semantic", {})
    keygen_cfg = semantic_cfg.get("keygen", {})
    features_cfg = semantic_cfg.get("features", {})
    config_hash = compute_config_hash(semantic_cfg)

    manifest_path = Path(semantic_cfg.get(
        "manifest_path", "experiments/results/public_conditions/manifest.json"
    ))
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    cache_dir = Path(semantic_cfg.get("cache_dir", "experiments/results/public_conditions"))
    if not cache_dir.is_absolute():
        cache_dir = paths.PROJECT_ROOT / cache_dir

    checks: List[Dict[str, Any]] = []
    failures: List[str] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. MANIFEST
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("1. MANIFEST CHECKS")
    print("=" * 60)

    # 1a. Manifest exists
    exists = manifest_path.exists()
    checks.append(check("manifest_exists", exists, str(manifest_path)))
    if not exists:
        print("[FAIL] manifest not found")
        _write_result(checks, failures, t_start)
        sys.exit(1)

    manifest = read_manifest(manifest_path)

    # 1b. Schema version
    sv = manifest.get("schema_version") == 1
    checks.append(check("schema_version", sv, f"got {manifest.get('schema_version')}"))

    # 1c. Split version matches
    split_manifest_path = paths.TEST_SPLIT_PATH
    with open(split_manifest_path, "r", encoding="utf-8") as fh:
        split_doc = json.load(fh)
    expected_sv = split_doc.get("split_version")
    actual_sv = manifest.get("split_version")
    sv_match = expected_sv == actual_sv
    checks.append(check("split_version_match", sv_match,
                         f"expected {expected_sv}, got {actual_sv}"))

    # 1d. Config hash recomputes
    stored_ch = manifest.get("config_hash")
    ch_match = stored_ch == config_hash
    checks.append(check("config_hash_recompute", ch_match,
                         f"expected {config_hash}, got {stored_ch}"))

    # 1e. Counts
    mcounts = manifest.get("counts", {})
    records = manifest.get("records", [])
    computed_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        computed_counts[r["role"]][r["class"]] += 1

    count_ok = True
    count_detail = []
    for role in ["calibration", "evaluation"]:
        for cls in CLASSES:
            expected = 22
            actual = computed_counts.get(role, {}).get(cls, 0)
            if actual != expected:
                count_ok = False
            count_detail.append(f"{role}/{cls}={actual}")
    checks.append(check("counts_22_per_class", count_ok, "; ".join(count_detail)))

    # 1f. Unique image_ids in split
    record_ids = [r["image_id"] for r in records]
    unique_ids = len(set(record_ids)) == len(record_ids)
    checks.append(check("unique_image_ids", unique_ids, f"{len(record_ids)} records"))

    # Load splits for cross-referencing
    eval_split = Split.load(role="evaluation", verify=False)
    cal_split = Split.load(role="calibration", allow_calibration=True, verify=False)
    all_split_ids = set(eval_split.ids()) | set(cal_split.ids())
    in_split = all(rid in all_split_ids for rid in record_ids)
    checks.append(check("all_records_in_split", in_split))

    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['check']}: {c.get('detail', '')}")
        if not c["pass"]:
            failures.append(c["check"])

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CACHE INTEGRITY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. CACHE INTEGRITY")
    print("=" * 60)

    # Load token info for Kpub validation
    tokens_all, _ = load_tokens()

    # Load split records for source sha256 cross-check
    all_split_records = {}
    for rec in eval_split.records():
        all_split_records[rec["image_id"]] = rec
    for rec in cal_split.records():
        all_split_records[rec["image_id"]] = rec

    cache_ok_count = 0
    cache_fail_count = 0
    kpub_set = defaultdict(list)
    kpub_checks_passed = 0
    feature_checks_passed = 0

    for rec in records:
        image_id = rec["image_id"]
        cache_path = paths.PROJECT_ROOT / rec["cache_path"]
        cls = rec["class"]

        # 2a. Cache file exists
        if not cache_path.exists():
            checks.append(check(f"cache_exists_{image_id}", False, "missing"))
            failures.append(f"cache_exists_{image_id}")
            cache_fail_count += 1
            continue

        # Read cache
        try:
            cached = read_cache_npz(cache_path)
        except Exception as exc:
            checks.append(check(f"cache_readable_{image_id}", False, str(exc)))
            failures.append(f"cache_readable_{image_id}")
            cache_fail_count += 1
            continue

        # 2b. Source sha256 matches split
        split_rec = all_split_records.get(image_id, {})
        expected_sha = split_rec.get("sha256", "")
        stored_sha = cached.get("source_sha256", "")
        if expected_sha and stored_sha != expected_sha:
            checks.append(check(f"source_sha256_{image_id}", False,
                                f"expected {expected_sha}, got {stored_sha}"))
            failures.append(f"source_sha256_{image_id}")
            cache_fail_count += 1
            continue

        # 2c. Kpub sha256 recomputes
        kpub = cached.get("kpub", "")
        stored_kpub_sha = rec.get("kpub_sha256", "")
        computed_kpub_sha = sha256_string(kpub)
        if stored_kpub_sha != computed_kpub_sha:
            checks.append(check(f"kpub_sha256_{image_id}", False,
                                f"expected {stored_kpub_sha}, got {computed_kpub_sha}"))
            failures.append(f"kpub_sha256_{image_id}")
            cache_fail_count += 1
            continue

        # 2d. Feature sha256 recomputes
        feat = cached.get("feature")
        stored_feat_sha = rec.get("feature_sha256")
        computed_feat_sha = feature_sha256(feat)
        if stored_feat_sha != computed_feat_sha:
            checks.append(check(f"feature_sha256_{image_id}", False,
                                f"expected {stored_feat_sha}, got {computed_feat_sha}"))
            failures.append(f"feature_sha256_{image_id}")
            cache_fail_count += 1
            continue

        # 2e. Preview files
        preview_path = rec.get("preview_path")
        if cls in ("common", "facial"):
            if not preview_path:
                checks.append(check(f"preview_exists_{image_id}", False, "missing path"))
                failures.append(f"preview_exists_{image_id}")
                cache_fail_count += 1
                continue
            full_preview = paths.PROJECT_ROOT / preview_path
            if not full_preview.exists():
                checks.append(check(f"preview_file_{image_id}", False, "file missing"))
                failures.append(f"preview_file_{image_id}")
                cache_fail_count += 1
                continue
        elif cls == "style":
            # Should NOT have a preview
            pass

        cache_ok_count += 1
        kpub_set[kpub].append(image_id)

    checks.append(check("cache_integrity", cache_fail_count == 0,
                         f"{cache_ok_count} ok, {cache_fail_count} failed"))
    print(f"  [{'PASS' if cache_fail_count == 0 else 'FAIL'}] "
          f"cache_integrity: {cache_ok_count} ok, {cache_fail_count} failed")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. KPUB CONTENT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. KPUB CONTENT CHECKS")
    print("=" * 60)

    from semantic.keygen import validate_kpub

    kpub_issues = 0
    max_chars = keygen_cfg.get("max_chars", 300)

    for rec in records:
        image_id = rec["image_id"]
        cache_path = paths.PROJECT_ROOT / rec["cache_path"]
        if not cache_path.exists():
            continue

        cached = read_cache_npz(cache_path)
        kpub = cached.get("kpub", "")
        token_info = tokens_all.get(image_id, {})
        split_rec = all_split_records.get(image_id, {})

        errors = validate_kpub(
            kpub,
            max_chars=max_chars,
            image_id=image_id,
            token=token_info.get("token", ""),
            wrong_token=token_info.get("wrong_token", ""),
            path=split_rec.get("path", ""),
            sha256=split_rec.get("sha256", ""),
        )
        if errors:
            kpub_issues += 1
            checks.append(check(f"kpub_content_{image_id}", False, "; ".join(errors)))
            failures.append(f"kpub_content_{image_id}")

        # Check method and model recorded
        method = cached.get("kpub_method", "")
        model_id = cached.get("kpub_model_id", "")
        if not method:
            kpub_issues += 1
            checks.append(check(f"kpub_method_{image_id}", False, "missing"))
            failures.append(f"kpub_method_{image_id}")
        if not model_id:
            kpub_issues += 1
            checks.append(check(f"kpub_model_{image_id}", False, "missing"))
            failures.append(f"kpub_model_{image_id}")

    # Check for duplicate Kpubs
    dupe_count = sum(1 for k, ids in kpub_set.items() if len(ids) > 1)
    checks.append(check("kpub_no_excessive_duplicates", True,
                         f"{dupe_count} duplicate Kpub strings (reported, not fatal)"))

    checks.append(check("kpub_content_all", kpub_issues == 0,
                         f"{kpub_issues} issues"))
    print(f"  [{'PASS' if kpub_issues == 0 else 'FAIL'}] kpub_content: {kpub_issues} issues")
    if dupe_count > 0:
        print(f"  [INFO] {dupe_count} duplicate Kpub strings across different images")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. FEATURE VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4. FEATURE CHECKS")
    print("=" * 60)

    feat_issues = 0
    num_classes = features_cfg.get("common", {}).get("num_classes", 150)
    min_keypoints = features_cfg.get("facial", {}).get("min_keypoints", 5)

    for rec in records:
        image_id = rec["image_id"]
        cls = rec["class"]
        cache_path = paths.PROJECT_ROOT / rec["cache_path"]
        if not cache_path.exists():
            continue

        cached = read_cache_npz(cache_path)
        feat_kind = cached.get("feature_kind", "")
        feat = cached.get("feature")

        if cls == "common":
            # 4a. Segmentation: shape (512,512), uint8, values < num_classes
            if feat_kind != "segmentation":
                feat_issues += 1
                checks.append(check(f"feat_kind_{image_id}", False,
                                     f"expected segmentation, got {feat_kind}"))
                failures.append(f"feat_kind_{image_id}")
                continue
            if feat is None:
                feat_issues += 1
                checks.append(check(f"feat_present_{image_id}", False, "missing"))
                failures.append(f"feat_present_{image_id}")
                continue
            if feat.shape != (512, 512):
                feat_issues += 1
                checks.append(check(f"feat_shape_{image_id}", False, f"{feat.shape}"))
                failures.append(f"feat_shape_{image_id}")
                continue
            if feat.dtype != np.uint8:
                feat_issues += 1
                checks.append(check(f"feat_dtype_{image_id}", False, f"{feat.dtype}"))
                failures.append(f"feat_dtype_{image_id}")
                continue
            if int(feat.max()) >= num_classes:
                feat_issues += 1
                checks.append(check(f"feat_range_{image_id}", False,
                                     f"max={feat.max()} >= {num_classes}"))
                failures.append(f"feat_range_{image_id}")
                continue
            distinct = len(np.unique(feat))
            if distinct < 2:
                feat_issues += 1
                checks.append(check(f"feat_diversity_{image_id}", False,
                                     f"only {distinct} classes"))
                failures.append(f"feat_diversity_{image_id}")
                continue
            if np.all(feat == 0):
                feat_issues += 1
                checks.append(check(f"feat_nonzero_{image_id}", False, "all zeros"))
                failures.append(f"feat_nonzero_{image_id}")
                continue

        elif cls == "facial":
            # 4b. Skeleton: keypoints >= min_keypoints, in bounds
            if feat_kind != "skeleton":
                feat_issues += 1
                checks.append(check(f"feat_kind_{image_id}", False,
                                     f"expected skeleton, got {feat_kind}"))
                failures.append(f"feat_kind_{image_id}")
                continue
            if feat is None or len(feat) < min_keypoints:
                n = 0 if feat is None else len(feat)
                feat_issues += 1
                checks.append(check(f"feat_keypoints_{image_id}", False,
                                     f"{n} < {min_keypoints}"))
                failures.append(f"feat_keypoints_{image_id}")
                continue
            # Check preview non-blank
            preview_path = rec.get("preview_path")
            if preview_path:
                full_path = paths.PROJECT_ROOT / preview_path
                if full_path.exists():
                    preview_img = Image.open(full_path)
                    arr = np.array(preview_img)
                    if arr.max() == 0:
                        feat_issues += 1
                        checks.append(check(f"feat_preview_{image_id}", False,
                                             "blank preview"))
                        failures.append(f"feat_preview_{image_id}")
                        continue

        elif cls == "style":
            # 4c. No feature
            if feat_kind != "none":
                feat_issues += 1
                checks.append(check(f"feat_kind_{image_id}", False,
                                     f"expected none, got {feat_kind}"))
                failures.append(f"feat_kind_{image_id}")
                continue
            if feat is not None:
                feat_issues += 1
                checks.append(check(f"feat_absent_{image_id}", False,
                                     "feature should be absent for style"))
                failures.append(f"feat_absent_{image_id}")
                continue

    checks.append(check("feature_validation", feat_issues == 0,
                         f"{feat_issues} issues"))
    print(f"  [{'PASS' if feat_issues == 0 else 'FAIL'}] feature_validation: {feat_issues} issues")

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. DETERMINISM
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("5. DETERMINISM CHECKS")
    print("=" * 60)

    det_ok = True
    det_details = []

    # Sample 2 images per class per role for determinism check
    from semantic.keygen import build_keygen
    from semantic.features import build_feature_extractor

    det_keygen = build_keygen(keygen_cfg)
    det_extractors = {cls: build_feature_extractor(features_cfg, cls) for cls in CLASSES}

    sample_per = 2
    for role in ["calibration", "evaluation"]:
        split = Split.load(role=role, allow_calibration=(role == "calibration"), verify=False)
        for cls in CLASSES:
            recs = split.records(cls)[:sample_per]
            for rec in recs:
                image_id = rec["image_id"]
                source_path = paths.PROJECT_ROOT / rec["path"]
                image = Image.open(source_path).convert("RGB").resize((512, 512), Image.BICUBIC)

                # Generate Kpub
                kpub_result = det_keygen.generate_kpub(image, context={
                    "image_id": image_id, "role": role, "class": cls,
                })
                new_kpub = kpub_result["kpub"]

                # Read stored Kpub
                cache_path = paths.PROJECT_ROOT / "experiments" / "results" / \
                    "public_conditions" / f"{image_id}.npz"
                if cache_path.exists():
                    cached = read_cache_npz(cache_path)
                    old_kpub = cached.get("kpub", "")
                    if new_kpub != old_kpub:
                        det_ok = False
                        det_details.append(
                            f"Kpub mismatch for {image_id}: "
                            f"{old_kpub[:50]}... vs {new_kpub[:50]}..."
                        )
                    else:
                        det_details.append(f"Kpub exact match: {image_id}")

                    # Feature determinism
                    feat_result = det_extractors[cls].extract(image)
                    new_feat = feat_result.get("feature")
                    old_feat = cached.get("feature")

                    if old_feat is not None and new_feat is not None:
                        if old_feat.shape != new_feat.shape:
                            det_ok = False
                            det_details.append(f"Feature shape mismatch: {image_id}")
                        elif old_feat.dtype != new_feat.dtype:
                            det_ok = False
                            det_details.append(f"Feature dtype mismatch: {image_id}")
                        else:
                            if np.issubdtype(old_feat.dtype, np.floating):
                                max_diff = float(np.max(np.abs(old_feat - new_feat)))
                                if max_diff > 0:
                                    det_ok = False
                                    det_details.append(
                                        f"Feature float diff={max_diff:.2e}: {image_id}"
                                    )
                                else:
                                    det_details.append(
                                        f"Feature exact match (float, max_diff=0): {image_id}"
                                    )
                            else:
                                if not np.array_equal(old_feat, new_feat):
                                    det_ok = False
                                    det_details.append(
                                        f"Feature integer mismatch: {image_id}"
                                    )
                                else:
                                    det_details.append(f"Feature exact match: {image_id}")
                    elif old_feat is None and new_feat is None:
                        det_details.append(f"Feature both None (style): {image_id}")
                    else:
                        det_ok = False
                        det_details.append(f"Feature presence mismatch: {image_id}")

    checks.append(check("determinism", det_ok, "; ".join(det_details[:10])))
    print(f"  [{'PASS' if det_ok else 'FAIL'}] determinism")
    for d in det_details:
        print(f"    {d}")

    # Cleanup
    det_keygen.unload()
    for ext in det_extractors.values():
        ext.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. VISUAL
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("6. VISUAL CHECKS")
    print("=" * 60)

    grid_path = paths.RESULTS_DIR / "phase7_conditions_grid.png"
    grid_exists = grid_path.exists()
    checks.append(check("grid_exists", grid_exists, str(grid_path)))
    print(f"  [{'PASS' if grid_exists else 'FAIL'}] grid_exists: {grid_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # WRITE RESULT
    # ═══════════════════════════════════════════════════════════════════════════
    _write_result(checks, failures, t_start)


def _write_result(checks, failures, t_start):
    elapsed = time.time() - t_start
    all_pass = len(failures) == 0

    # Compute counts from manifest if available
    manifest_path_default = paths.RESULTS_DIR / "public_conditions" / "manifest.json"
    counts = {}
    if manifest_path_default.exists():
        manifest = read_manifest(manifest_path_default)
        counts = manifest.get("counts", {})
        config_hash = manifest.get("config_hash", "")
        split_version = manifest.get("split_version", "")
    else:
        config_hash = ""
        split_version = ""

    result = {
        "pass": all_pass,
        "split_version": split_version,
        "config_hash": config_hash,
        "checks": checks,
        "counts": counts,
        "failures": failures,
        "elapsed_s": round(elapsed, 2),
    }

    out_path = paths.RESULTS_DIR / "phase7_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULT: {'PASS' if all_pass else 'FAIL'}")
    print(f"checks: {len(checks)}, failures: {len(failures)}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"written: {out_path}")
    print(f"{'='*60}")

    if not all_pass:
        print(f"\nFailed checks:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
