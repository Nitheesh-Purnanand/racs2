"""
Phase 4 (Step 8) — Verify the Retrieval Corpus for RACS2.

Integrity + determinism gate for the frozen corpus built by
`scripts/phase4_build_corpus.py`. It must run **offline**: it only touches the
on-disk corpus and index, never the network.

Checks:
  1. index loads; schema_version == 1; count == len(images) == expected count
  2. every image_path exists on disk
  3. every image opens as RGB and is exactly canonical_size x canonical_size
  4. no duplicate image_ids
  5. loading the corpus twice yields the identical canonical id order
  6. recomputed corpus_version == stored corpus_version
  7. re-hashing a random sample of images matches the index sha256
  8. reports total bytes on disk and mean image size
  9. writes a contact sheet of the first N images to experiments/results/

Exits 0 on PASS, 1 on FAIL.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\phase4_verify.py
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.paths  # noqa: E402,F401

import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from common.paths import PROJECT_ROOT as _ROOT, RESULTS_DIR  # noqa: E402
from retrieval.corpus import Corpus  # noqa: E402
from retrieval.schema import (  # noqa: E402
    SCHEMA_VERSION,
    compute_corpus_version,
    sha256_file,
    validate_index,
)


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def contact_sheet(images, cols: int = 8, cell: int = 128):
    """Compose a grid of RGB images on a white canvas (no torch required)."""
    n = len(images)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, img in enumerate(images):
        thumb = img.convert("RGB").resize((cell, cell), Image.BICUBIC)
        r, c = divmod(i, cols)
        sheet.paste(thumb, (c * cell, r * cell))
    return sheet


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify the RACS2 retrieval corpus.")
    p.add_argument("--config", default=str(_ROOT / "configs" / "base.yaml"))
    p.add_argument("--index-path", default=None,
                   help="index json path (default: config index_path)")
    p.add_argument("--expected-count", type=int, default=None,
                   help="expected image count (default: config corpus_size)")
    p.add_argument("--sample", type=int, default=20,
                   help="number of images to re-hash (default: 20)")
    p.add_argument("--grid-n", type=int, default=64,
                   help="how many images to put on the contact sheet (default: 64)")
    p.add_argument("--no-grid", action="store_true", help="skip the contact sheet")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    section("1. Load config")
    with open(Path(args.config), "r", encoding="utf-8") as fh:
        cfg = (yaml.safe_load(fh) or {}).get("retrieval", {}) or {}
    index_path = Path(args.index_path) if args.index_path else (_ROOT / cfg.get("index_path", "data/metadata/image_index.json"))
    expected_count = args.expected_count if args.expected_count is not None else cfg.get("corpus_size")
    canonical_size = int(cfg.get("canonical_size", 512))
    print(f"index_path     = {index_path}")
    print(f"expected_count = {expected_count}")

    failures = []
    if not index_path.exists():
        print(f"FAIL: index does not exist: {index_path}")
        return 1

    section("2. Load + validate index")
    corpus = Corpus(index_path=index_path, validate=False)
    index = corpus.index
    problems = validate_index(index, require_files=True, expected_count=expected_count)
    if problems:
        for p in problems:
            print(f"  [x] {p}")
        failures.extend(problems)
    else:
        print(f"  [ok] schema valid, {len(corpus)} records, all files present")
    print(f"  schema_version : {index['schema_version']} (expected {SCHEMA_VERSION})")
    print(f"  source         : {index['source']}")
    print(f"  canonical_size : {index['canonical_size']}")
    print(f"  resize_mode    : {index['resize_mode']}")
    print(f"  corpus_version : {index['corpus_version']}")

    section("3. Per-image decode + size")
    bad_size, bad_mode, unreadable = [], [], []
    total_bytes = 0
    for rec in corpus.records():
        path = corpus.resolve_path(rec["image_id"])
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                size = rgb.size
        except Exception as exc:  # noqa: BLE001
            unreadable.append((rec["image_id"], repr(exc)))
            continue
        total_bytes += path.stat().st_size
        if size != (canonical_size, canonical_size):
            bad_size.append((rec["image_id"], size))
        if rgb.mode != "RGB":
            bad_mode.append((rec["image_id"], rgb.mode))
    n = len(corpus)
    if unreadable:
        failures.append(f"{len(unreadable)} unreadable image(s)")
        print(f"  [x] unreadable: {unreadable[:5]}")
    if bad_size:
        failures.append(f"{len(bad_size)} image(s) with wrong size")
        print(f"  [x] wrong size: {bad_size[:5]}")
    if bad_mode:
        failures.append(f"{len(bad_mode)} image(s) not RGB")
        print(f"  [x] wrong mode: {bad_mode[:5]}")
    if not (unreadable or bad_size or bad_mode):
        print(f"  [ok] all {n} images decode as RGB {canonical_size}x{canonical_size}")

    section("4. Duplicate ids")
    ids = corpus.ids()
    if len(set(ids)) != len(ids):
        failures.append("duplicate image_ids")
        print("  [x] duplicate image_ids found")
    else:
        print(f"  [ok] {len(ids)} unique image_ids")

    section("5. Determinism (stable order across loads)")
    a = Corpus(index_path=index_path).ids()
    b = Corpus(index_path=index_path).ids()
    if a == b == ids:
        print("  [ok] canonical order identical across loads")
    else:
        failures.append("non-deterministic id order")
        print("  [x] id order differs across loads")

    section("6. corpus_version recompute")
    recomputed = compute_corpus_version(ids)
    if recomputed == index["corpus_version"]:
        print(f"  [ok] matches stored ({recomputed[:16]}...)")
    else:
        failures.append("corpus_version mismatch")
        print(f"  [x] stored={index['corpus_version']} recomputed={recomputed}")

    section(f"7. sha256 sample ({min(args.sample, n)} of {n})")
    rng = random.Random(1234)
    sample_ids = rng.sample(ids, min(args.sample, n))
    mismatches = []
    for image_id in sample_ids:
        path = corpus.resolve_path(image_id)
        actual = sha256_file(path)
        if actual != corpus.get(image_id)["sha256"]:
            mismatches.append(image_id)
    if mismatches:
        failures.append(f"{len(mismatches)} sha256 mismatch(es)")
        print(f"  [x] mismatches: {mismatches[:5]}")
    else:
        print(f"  [ok] all sampled sha256 digests match the index")

    section("8. Size report")
    print(f"  total on disk : {total_bytes/1e6:.1f} MB")
    print(f"  mean per image: {total_bytes/max(n,1)/1e3:.1f} kB")

    section("9. Contact sheet")
    if args.no_grid:
        print("  skipped (--no-grid)")
    elif not ids:
        print("  skipped (empty corpus)")
    else:
        loaded = [corpus.load(i) for i in ids[: args.grid_n]]
        sheet = contact_sheet(loaded)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "phase4_corpus_grid.png"
        sheet.save(out)
        print(f"  [ok] saved {out}  ({sheet.size[0]}x{sheet.size[1]})")

    section("RESULT")
    if failures:
        print(f"FAIL - {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS - corpus is deterministic, complete and matches its index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
