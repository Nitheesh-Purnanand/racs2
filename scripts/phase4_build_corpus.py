"""
Phase 4 (Step 5) — Build the Retrieval Image Corpus for RACS2.

One-time builder that turns a raw image source (default: COCO val2017) into the
frozen retrieval corpus:

  raw images  ->  deterministic subset  ->  canonical 512x512 RGB PNGs
              ->  data/metadata/image_index.json   (schema: retrieval/schema.py)

The result is a **shared protocol resource**: the same bytes, same order and the
same `corpus_version` on the transmitter and the receiver. Everything downstream
(Phase 5 embeddings, Phase 6 FAISS, Phase 7 token selection) indexes into this
list, so it is built once and then frozen.

Typical use (COCO already downloaded into data/_raw_coco):

    .venv\\Scripts\\python.exe scripts\\phase4_build_corpus.py

Useful flags:

    --limit 50            build a small corpus (fast smoke test)
    --skip-download       never touch the network
    --keep-raw            do not delete data/_raw_coco afterwards
    --out-dir, --index-path   write somewhere else (used by the test harness)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# Make the project root importable + redirect HF cache to D: before any model lib.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.paths  # noqa: E402,F401

import requests  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from common.seed import set_seed  # noqa: E402
from common.paths import (  # noqa: E402
    IMAGE_INDEX_PATH,
    PROJECT_ROOT as _ROOT,
    RAW_CORPUS_DIR,
)
from retrieval.schema import (  # noqa: E402
    build_index,
    sha256_file,
    write_index_atomic,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESIZE_FILTER = Image.BICUBIC


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def load_retrieval_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg.get("retrieval", {}) or {}


def find_raw_images(raw_dir: Path) -> list:
    """Recursively find source images (handles double-nested zip extraction)."""
    return sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def find_captions_file(raw_dir: Path) -> Optional[Path]:
    """Locate COCO captions json anywhere under the raw dir."""
    for name in ("captions_val2017.json", "captions_train2017.json"):
        hits = sorted(raw_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def load_captions(captions_path: Optional[Path]) -> Dict[str, str]:
    """Map source file stem (e.g. '000000000139') -> first caption string."""
    if captions_path is None or not captions_path.exists():
        return {}
    with open(captions_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    id_by_file = {im["file_name"]: im["id"] for im in data.get("images", [])}
    first_caption: Dict[int, str] = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in first_caption and ann.get("caption"):
            first_caption[img_id] = ann["caption"]
    out: Dict[str, str] = {}
    for file_name, img_id in id_by_file.items():
        cap = first_caption.get(img_id, "")
        if cap:
            out[Path(file_name).stem] = cap
    return out


def resize_canonical(img: Image.Image, size: int, mode: str) -> Image.Image:
    """Apply the documented, protocol-fixed preprocessing rule."""
    if mode != "shortest_side_then_center_crop":
        raise ValueError(f"unsupported resize_mode: {mode!r}")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), RESIZE_FILTER)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def download_and_extract(raw_dir: Path, urls: list, keep_zip: bool = False) -> None:
    """Download + unzip the source archives into ``raw_dir`` (idempotent-ish)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        if not url:
            continue
        zip_path = raw_dir / Path(url).name
        if not zip_path.exists():
            print(f"  downloading {url} ...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(zip_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            print(f"\r    {pct:5.1f}%  {done/1e6:7.1f}/{total/1e6:.1f} MB",
                                  end="", flush=True)
                print()
        print(f"  extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        if not keep_zip:
            zip_path.unlink(missing_ok=True)


def rel_or_abs(path: Path) -> str:
    """Store project-relative paths when possible so the index is portable."""
    try:
        return str(path.resolve().relative_to(_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the RACS2 retrieval image corpus.")
    p.add_argument("--config", default=str(_ROOT / "configs" / "base.yaml"))
    p.add_argument("--raw-dir", default=str(RAW_CORPUS_DIR),
                   help="directory holding the source images (default: data/_raw_coco)")
    p.add_argument("--out-dir", default=None,
                   help="canonical image output dir (default: config corpus_dir)")
    p.add_argument("--index-path", default=None,
                   help="index json path (default: config index_path)")
    p.add_argument("--limit", type=int, default=None,
                   help="build only N images (smoke test); default: config corpus_size")
    p.add_argument("--seed", type=int, default=None,
                   help="subset seed; default: config subset_seed")
    p.add_argument("--source", default=None, help="source label; default: config source")
    p.add_argument("--download", action="store_true",
                   help="download the source archives if no raw images are found")
    p.add_argument("--skip-download", action="store_true",
                   help="never download; fail if raw images are missing")
    p.add_argument("--keep-raw", action="store_true",
                   help="do not delete the raw source directory afterwards")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    section("1. Setup")
    seed = args.seed if args.seed is not None else 42
    set_seed(seed, verbose=True)

    cfg = load_retrieval_config(Path(args.config))
    source = args.source or cfg.get("source", "coco_val2017")
    canonical_size = int(cfg.get("canonical_size", 512))
    resize_mode = cfg.get("resize_mode", "shortest_side_then_center_crop")
    subset_seed = args.seed if args.seed is not None else int(cfg.get("subset_seed", 42))
    corpus_size = args.limit if args.limit is not None else int(cfg.get("corpus_size", 1000))

    out_dir = Path(args.out_dir) if args.out_dir else (_ROOT / cfg.get("corpus_dir", "data/retrieval_images"))
    index_path = Path(args.index_path) if args.index_path else (_ROOT / cfg.get("index_path", str(IMAGE_INDEX_PATH.relative_to(_ROOT))))
    raw_dir = Path(args.raw_dir)

    print(f"source        = {source}")
    print(f"corpus_size   = {corpus_size}")
    print(f"subset_seed   = {subset_seed}")
    print(f"canonical     = {canonical_size}  ({resize_mode})")
    print(f"raw_dir       = {raw_dir}")
    print(f"out_dir       = {out_dir}")
    print(f"index_path    = {index_path}")

    # ------------------------------------------------------------------
    section("2. Locate raw source images")
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
    raw_images = find_raw_images(raw_dir)
    if not raw_images:
        if args.skip_download:
            print(f"ERROR: no images under {raw_dir} and --skip-download was set.")
            return 2
        if not args.download:
            print(
                f"ERROR: no source images under {raw_dir}.\n"
                "  Download once, e.g.:\n"
                "    $raw = 'data\\_raw_coco'\n"
                "    Invoke-WebRequest http://images.cocodataset.org/zips/val2017.zip -OutFile \"$raw\\val2017.zip\"\n"
                "    Invoke-WebRequest http://images.cocodataset.org/annotations/annotations_trainval2017.zip -OutFile \"$raw\\annotations.zip\"\n"
                "  (or pass --download to let this script fetch them)."
            )
            return 2
        urls = [
            cfg.get("source_ids_url", "http://images.cocodataset.org/zips/val2017.zip"),
            cfg.get("source_annotations_url",
                    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
        ]
        download_and_extract(raw_dir, urls, keep_zip=args.keep_raw)
        raw_images = find_raw_images(raw_dir)

    print(f"found {len(raw_images)} raw image(s)")
    if not raw_images:
        print("ERROR: still no images after download.")
        return 2

    captions_path = find_captions_file(raw_dir)
    captions = load_captions(captions_path)
    print(f"captions      = {'yes' if captions else 'no'} ({captions_path})")

    # ------------------------------------------------------------------
    section("3. Deterministic subset")
    all_stems = sorted({p.stem for p in raw_images})
    if corpus_size > len(all_stems):
        print(f"WARNING: requested {corpus_size} > available {len(all_stems)}; using all.")
        corpus_size = len(all_stems)
    rng = random.Random(subset_seed)
    chosen = sorted(rng.sample(all_stems, corpus_size))
    by_stem = {p.stem: p for p in raw_images}
    print(f"selected {len(chosen)} / {len(all_stems)} ids (seed={subset_seed})")
    print(f"first ids: {chosen[:3]}  last ids: {chosen[-1:]}")

    # ------------------------------------------------------------------
    section("4. Canonicalize -> PNG")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    records = []
    failures = []
    for n, stem in enumerate(chosen, 1):
        src = by_stem[stem]
        image_id = f"{source}_{stem}"
        dst = out_dir / f"{image_id}.png"
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im = resize_canonical(im, canonical_size, resize_mode)
                im.save(dst, format="PNG")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append((stem, repr(exc)))
            continue

        records.append({
            "image_id": image_id,
            "image_path": rel_or_abs(dst),
            "source_id": stem,
            "caption": captions.get(stem, ""),
            "width": canonical_size,
            "height": canonical_size,
            "sha256": sha256_file(dst),
        })
        if n % 100 == 0 or n == len(chosen):
            print(f"  [{n:>4}/{len(chosen)}] {image_id}")

    elapsed = time.perf_counter() - t0
    if failures:
        print(f"WARNING: {len(failures)} image(s) failed to process:")
        for stem, err in failures[:10]:
            print(f"  - {stem}: {err}")
    if not records:
        print("ERROR: no images were written.")
        return 3

    # ------------------------------------------------------------------
    section("5. Write index")
    index = build_index(
        records,
        source=source,
        canonical_size=canonical_size,
        resize_mode=resize_mode,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_index_atomic(index, index_path)

    total_bytes = sum((out_dir / f"{r['image_id']}.png").stat().st_size for r in records)
    print(f"images written : {len(records)}")
    print(f"bytes on disk  : {total_bytes/1e6:.1f} MB (canonical PNGs)")
    print(f"corpus_version : {index['corpus_version']}")
    print(f"elapsed        : {elapsed:.1f}s")
    print(f"index          : {index_path}")

    # ------------------------------------------------------------------
    section("6. Cleanup")
    if args.keep_raw:
        print(f"keeping raw source dir: {raw_dir}")
    elif raw_dir.exists():
        shutil.rmtree(raw_dir)
        print(f"removed raw source dir: {raw_dir}")

    print("\nPhase 4 corpus build complete.")
    print("Next: .venv\\Scripts\\python.exe scripts\\phase4_verify.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
