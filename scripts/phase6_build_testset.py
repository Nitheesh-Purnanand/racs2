"""
Phase 6: build the frozen, role-tagged test splits.

Two roles, never mixed:
  - calibration: the paper's UniStega release. Used only for the approximate
    reproduction check in Phase 15A.
  - evaluation: our own authoritative dataset. Used for every claim and every
    baseline-versus-RACS comparison.

The builder is deterministic: sources are sorted, sampling is fixed, canonical
PNGs are byte-stable, and both manifests carry recomputable version hashes.

Usage:
    python scripts/phase6_build_testset.py
    python scripts/phase6_build_testset.py --rebuild --per-class 25
    python scripts/phase6_build_testset.py --skip-download   # reuse raw cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import yaml
from PIL import Image, ImageDraw

import common.paths  # noqa: F401  (side effect: caches on D:)
from common.paths import (
    CALIBRATION_DIR,
    DEVIATIONS_PATH,
    IMAGE_INDEX_PATH,
    PROJECT_ROOT,
    RAW_SOURCES_DIR,
    RESULTS_DIR,
    TEST_SET_DIR,
    TEST_SPLIT_PATH,
    TOKENS_PATH,
)
from common.seed import set_seed
from evaluation.splits import (
    CLASSES,
    compute_split_version,
    compute_tokens_version,
    sha256_file,
    write_manifests,
)

USER_AGENT = {"User-Agent": "racs2-phase6/1.0 (research; local evaluation harness)"}
CANONICAL = "shortest_side_512_bicubic -> center_crop_512"
UNISTEGA_DRIVE_ID = "1ITaNvYAP8hB32TxwEo4Rdf-515plUOdA"
UNISTEGA_ZIP_URL = "https://drive.usercontent.google.com/download"
PAPER_CLASS_ORDER = {name: i for i, name in enumerate(CLASSES)}


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def http_get(url, timeout=90, retries=4, backoff=1.5):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - network errors are heterogeneous
            last_error = error
            if attempt + 1 < retries:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last_error})")


def download(url, destination, timeout=120):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    payload = http_get(url, timeout=timeout)
    if not payload:
        raise RuntimeError(f"empty response: {url}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(destination)
    return destination


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------
def canonicalise(source, destination, size=512):
    """RGB, shortest side to `size` bicubic, centre crop, deterministic PNG."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"degenerate image: {source}")
        if width <= height:
            new_w, new_h = size, max(size, round(height * size / width))
        else:
            new_w, new_h = max(size, round(width * size / height)), size
        if (new_w, new_h) != (width, height):
            image = image.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        image = image.crop((left, top, left + size, top + size))
        image.save(destination, format="PNG", compress_level=6, optimize=False)
    return destination


# ---------------------------------------------------------------------------
# Calibration: UniStega
# ---------------------------------------------------------------------------
def ensure_unistega(cfg, skip_download=False):
    raw_dir = RAW_SOURCES_DIR / "unistega"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(cfg.get("zip_path") or raw_dir / "Unistega.zip")
    if not zip_path.is_absolute():
        zip_path = PROJECT_ROOT / zip_path

    local_temp = PROJECT_ROOT / "temp" / "Unistega.zip"
    if not zip_path.exists() and local_temp.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_temp, zip_path)
        log(f"[unistega] copied cached zip from {local_temp}")

    if not zip_path.exists():
        if skip_download:
            raise RuntimeError(f"UniStega zip missing and --skip-download set: {zip_path}")
        log("[unistega] downloading 31 MB release")
        url = (
            f"{cfg.get('url', UNISTEGA_ZIP_URL)}"
            f"?id={cfg.get('drive_id', UNISTEGA_DRIVE_ID)}&export=download&confirm=t"
        )
        download(url, zip_path, timeout=180)
        log(f"[unistega] saved {zip_path} ({zip_path.stat().st_size} bytes)")

    extracted = raw_dir / "extracted"
    marker = extracted / ".complete"
    if not marker.exists():
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted)
        marker.write_text("ok\n", encoding="utf-8")
        log(f"[unistega] extracted to {extracted}")
    return extracted


def unistega_allowed_names(subset_dir):
    """Returns the file names that belong to the subset's declared class.

    The released `similar_prompts` config mixes FFHQ and CelebA-HQ faces with
    COCO non-face images. Only the two face sources belong to the facial class.
    """
    config_path = subset_dir.parent / "config.yaml"
    if not config_path.exists():
        return None
    entries = yaml.safe_load(config_path.read_text(encoding="utf-8")) or []
    allowed = set()
    subset_name = subset_dir.parent.name
    for entry in entries:
        dataset = str(entry.get("original_dataset", ""))
        if subset_name == "similar_prompts" and dataset not in {"FFHQ", "CelebA-HQ"}:
            continue
        image_path = entry.get("image_path")
        if image_path:
            allowed.add(Path(image_path).name)
    return allowed


def collect_unistega(extracted, class_map):
    """Returns {class: [{path, source, license, source_ref, original_id}]}."""
    collected = {name: [] for name in CLASSES}
    for subset, target_class in class_map.items():
        data_dir = extracted / subset / "data"
        if not data_dir.exists():
            log(f"[unistega] missing subset directory: {data_dir}")
            continue
        allowed = unistega_allowed_names(data_dir)
        kept = 0
        for path in sorted(data_dir.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if allowed is not None and path.name not in allowed:
                continue
            kept += 1
            collected[target_class].append({
                "path": path,
                "source": f"unistega:{subset}",
                "license": "research use (paper release)",
                "source_ref": "DiffStega release linked by paper ref. [46]",
                "original_id": path.stem,
            })
        log(f"[unistega] {subset}: {kept} images -> {target_class}")
    return collected


# ---------------------------------------------------------------------------
# Evaluation: COCO common
# ---------------------------------------------------------------------------
def load_coco_ids(cache_dir=None):
    """Returns the sorted, zero-padded COCO val2017 image ids.

    The id list comes from the official annotation archive. It is cached so the
    253 MB archive is downloaded at most once.
    """
    cache_dir = Path(cache_dir or (RAW_SOURCES_DIR / "coco_eval"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "val2017_image_ids.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    archive = cache_dir / "annotations_trainval2017.zip"
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    if not archive.exists() or archive.stat().st_size < 200_000_000:
        log("[coco] downloading 253 MB annotation archive (one time)")
        download(url, archive, timeout=600)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        member = None
        for candidate in ("annotations/image_info_val2017.json",
                          "annotations/instances_val2017.json",
                          "annotations/captions_val2017.json"):
            if candidate in names:
                member = candidate
                break
        if member is None:
            raise RuntimeError(
                "annotation archive has no val2017 image list; members: "
                + ", ".join(sorted(names))
            )
        payload = json.loads(bundle.read(member).decode("utf-8"))
    identifiers = sorted(f"{image['id']:012d}" for image in payload.get("images", []))
    cache.write_text(json.dumps(identifiers) + "\n", encoding="utf-8")
    log(f"[coco] cached {len(identifiers)} val2017 ids")
    try:
        archive.unlink()
    except OSError:
        pass
    return identifiers


def fetch_coco(count, corpus_ids, base_url):
    raw_dir = RAW_SOURCES_DIR / "coco_eval"
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus = set(corpus_ids)
    pool = [identifier for identifier in load_coco_ids(raw_dir)
            if identifier not in corpus]
    chosen = []
    for identifier in pool:
        if len(chosen) >= count:
            break
        destination = raw_dir / f"{identifier}.jpg"
        if not destination.exists():
            url = f"{base_url}/{identifier}.jpg"
            try:
                download(url, destination, timeout=45)
            except Exception as error:  # noqa: BLE001
                log(f"[coco] skip {identifier}: {error}")
                continue
        try:
            with Image.open(destination) as image:
                image.verify()
        except Exception:  # noqa: BLE001
            destination.unlink(missing_ok=True)
            continue
        chosen.append({
            "path": destination,
            "source": "coco_val2017",
            "license": "COCO val2017 (Flickr imagery; per-image licence)",
            "source_ref": "http://cocodataset.org/#download",
            "original_id": identifier,
        })
        if len(chosen) % 5 == 0:
            log(f"[coco] {len(chosen)}/{count}")
    return chosen


# ---------------------------------------------------------------------------
# Evaluation: HuggingFace datasets-server rows
# ---------------------------------------------------------------------------
def hf_rows(dataset, config, split, offset, length=100):
    query = urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    payload = http_get(f"https://datasets-server.huggingface.co/rows?{query}", timeout=90)
    data = json.loads(payload.decode("utf-8"))
    return data.get("rows", [])


def hf_image_url(row):
    image = row.get("row", {}).get("image")
    if isinstance(image, dict):
        return image.get("src") or image.get("url")
    if isinstance(image, str) and image.startswith("http"):
        return image
    return None


def fetch_hf_images(spec, count, tag):
    dataset = spec["dataset"]
    config = spec.get("config", "default")
    split = spec.get("split", "train")
    raw_dir = RAW_SOURCES_DIR / tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    items = []
    offsets = [0, 100, 200, 300, 400]
    for offset in offsets:
        if len(items) >= count:
            break
        try:
            rows = hf_rows(dataset, config, split, offset=offset, length=100)
        except Exception as error:  # noqa: BLE001
            log(f"[{tag}] rows offset {offset} failed: {error}")
            break
        if not rows:
            break
        for index, row in enumerate(rows):
            if len(items) >= count:
                break
            url = hf_image_url(row)
            if not url:
                continue
            identifier = f"{tag}_{offset + index:04d}"
            destination = raw_dir / f"{identifier}.jpg"
            if not destination.exists():
                try:
                    download(url, destination, timeout=60)
                except Exception as error:  # noqa: BLE001
                    log(f"[{tag}] skip {identifier}: {error}")
                    continue
            try:
                with Image.open(destination) as image:
                    image.verify()
            except Exception:  # noqa: BLE001
                destination.unlink(missing_ok=True)
                continue
            items.append({
                "path": destination,
                "source": f"hf:{dataset}",
                "license": "see dataset card (research use)",
                "source_ref": f"https://huggingface.co/datasets/{dataset}",
                "original_id": identifier,
            })
        log(f"[{tag}] {len(items)}/{count}")
    return items


# ---------------------------------------------------------------------------
# Evaluation: Wikimedia Commons fallback
# ---------------------------------------------------------------------------
def fetch_commons(search, count, tag):
    raw_dir = RAW_SOURCES_DIR / tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": search, "gsrnamespace": 6, "gsrlimit": 50,
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 512,
    })
    payload = http_get(f"https://commons.wikimedia.org/w/api.php?{query}", timeout=60)
    data = json.loads(payload.decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    items = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        if len(items) >= count:
            break
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        title = page.get("title", "")
        if not url or Path(title).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        metadata = info.get("extmetadata", {})
        license_name = metadata.get("LicenseShortName", {}).get("value", "see source")
        identifier = f"{tag}_{len(items):04d}"
        destination = raw_dir / f"{identifier}{Path(title).suffix.lower()}"
        if not destination.exists():
            try:
                download(url, destination, timeout=60)
            except Exception as error:  # noqa: BLE001
                log(f"[{tag}] skip {title}: {error}")
                continue
        try:
            with Image.open(destination) as image:
                image.verify()
        except Exception:  # noqa: BLE001
            destination.unlink(missing_ok=True)
            continue
        items.append({
            "path": destination,
            "source": f"commons:{search}",
            "license": license_name,
            "source_ref": page.get("title", ""),
            "original_id": identifier,
        })
    log(f"[{tag}] {len(items)}/{count} from Commons fallback")
    return items


def fetch_eval_class(name, spec, count, corpus_ids=None):
    if name == "common":
        if corpus_ids is None:
            index = json.loads(IMAGE_INDEX_PATH.read_text(encoding="utf-8"))
            corpus_ids = [record["source_id"] for record in index.get("images", [])]
        return fetch_coco(count, corpus_ids, spec["base_url"])
    if name == "facial":
        items = []
        if spec.get("dataset"):
            items = fetch_hf_images(spec, count, "ffhq_eval")
        if len(items) < count:
            log("[facial] topping up from Wikimedia Commons")
            items += fetch_commons(
                spec.get("commons_fallback", "filetype:bitmap portrait photograph person"),
                count - len(items), "ffhq_eval_commons")
        return items
    if name == "style":
        if spec.get("prefer_commons") or not spec.get("dataset"):
            return fetch_commons(spec.get("commons", "filetype:bitmap cat photograph"),
                                 count, "afhq_eval_commons")
        items = fetch_hf_images(spec, count, "afhq_eval")
        if len(items) < count:
            log("[style] topping up from Wikimedia Commons")
            items += fetch_commons(
                spec.get("commons", "filetype:bitmap cat photograph"),
                count - len(items), "afhq_eval_commons")
        return items
    raise ValueError(f"unknown class {name!r}")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def materialise(items, role, cls, size, exclude_sha, wanted):
    out_dir = (CALIBRATION_DIR if role == "calibration" else TEST_SET_DIR) / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()
    records = []
    used_sha = set()
    for item in items:
        if len(records) >= wanted:
            break
        image_id = f"{role[:3]}_{cls}_{len(records):04d}"
        destination = out_dir / f"{image_id}.png"
        try:
            canonicalise(item["path"], destination, size=size)
        except Exception as error:  # noqa: BLE001
            log(f"[{role}/{cls}] skip {item.get('original_id')}: {error}")
            destination.unlink(missing_ok=True)
            continue
        digest = sha256_file(destination)
        if digest in exclude_sha or digest in used_sha:
            log(f"[{role}/{cls}] duplicate image dropped: {item.get('original_id')}")
            destination.unlink(missing_ok=True)
            continue
        used_sha.add(digest)
        records.append({
            "image_id": image_id,
            "role": role,
            "class": cls,
            "path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": digest,
            "width": size,
            "height": size,
            "mode": "RGB",
            "source": item["source"],
            "license": item["license"],
            "source_ref": item["source_ref"],
            "original_id": str(item["original_id"]),
        })
    return records


def make_tokens(records):
    tokens = []
    for record in records:
        digest = hashlib.sha256(record["image_id"].encode("utf-8")).hexdigest()
        value = int(digest[:16], 16)
        token = value % 100_000_000
        wrong = (token + 1 + (value >> 17) % 9_999_999) % 100_000_000
        if wrong == token:
            wrong = (token + 7) % 100_000_000
        tokens.append({
            "image_id": record["image_id"],
            "token": f"{token:08d}",
            "wrong_token": f"{wrong:08d}",
        })
    return tokens


def write_contact_sheet(records, path, per_row=5, thumb=160, pad=6,
                        label_height=18):
    """Writes a labelled grid: one row per role/class, `per_row` samples each."""
    groups = [(role, cls) for role in ("calibration", "evaluation") for cls in CLASSES]
    width = pad + per_row * (thumb + pad)
    height = pad + len(groups) * (thumb + pad + label_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (role, cls) in enumerate(groups):
        y = pad + row * (thumb + pad + label_height)
        draw.text((pad, y + 2), f"{role}/{cls}", fill="black")
        subset = [r for r in records if r["role"] == role and r["class"] == cls]
        for column, record in enumerate(subset[:per_row]):
            with Image.open(PROJECT_ROOT / record["path"]) as image:
                tile = image.convert("RGB").resize((thumb, thumb), Image.BICUBIC)
            canvas.paste(tile, (pad + column * (thumb + pad), y + label_height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=6)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build the Phase 6 role-tagged test splits.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true",
                        help="delete existing split folders before building")
    parser.add_argument("--skip-download", action="store_true",
                        help="reuse raw source cache; fail if something is missing")
    args = parser.parse_args()

    set_seed(42, verbose=True)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    eval_cfg = config["evaluation"]
    per_class = int(args.per_class or eval_cfg.get("per_class", 25))
    size = int(eval_cfg.get("image_size", 512))

    if per_class < 20:
        raise SystemExit("per_class must be at least 20 for the Phase 6 gate")

    if args.rebuild:
        for folder in (CALIBRATION_DIR, TEST_SET_DIR):
            if folder.exists():
                shutil.rmtree(folder)
        log("[build] removed existing split folders")

    index = json.loads(IMAGE_INDEX_PATH.read_text(encoding="utf-8"))
    corpus_version = index.get("corpus_version")
    corpus_sha = {record["sha256"] for record in index.get("images", [])}

    log("[build] preparing calibration sources (UniStega)")
    extracted = ensure_unistega(eval_cfg.get("unistega", {}), skip_download=args.skip_download)
    calibration_sources = collect_unistega(
        extracted, eval_cfg["unistega"]["class_map"])

    log("[build] preparing evaluation sources")
    sources = eval_cfg.get("sources", {})
    evaluation_sources = {
        "common": fetch_eval_class("common", sources["coco"], per_class + 10,
                                   corpus_ids=[r["source_id"] for r in index.get("images", [])]),
        "facial": fetch_eval_class("facial", sources["facial"], per_class + 10),
        "style": fetch_eval_class("style", sources["style"], per_class + 10),
    }

    log("[build] canonicalising calibration split")
    calibration_records = []
    for cls in CLASSES:
        records = materialise(calibration_sources.get(cls, []), "calibration", cls,
                              size, corpus_sha, per_class)
        log(f"[build] calibration/{cls}: {len(records)}")
        calibration_records.extend(records)

    calibration_sha = {record["sha256"] for record in calibration_records}
    log("[build] canonicalising evaluation split")
    evaluation_records = []
    for cls in CLASSES:
        records = materialise(evaluation_sources.get(cls, []), "evaluation", cls,
                              size, corpus_sha | calibration_sha, per_class)
        log(f"[build] evaluation/{cls}: {len(records)}")
        evaluation_records.extend(records)

    ordered = (sorted(calibration_records, key=lambda r: (PAPER_CLASS_ORDER[r["class"]], r["image_id"]))
               + sorted(evaluation_records, key=lambda r: (PAPER_CLASS_ORDER[r["class"]], r["image_id"])))

    for role in ("calibration", "evaluation"):
        for cls in CLASSES:
            count = sum(1 for r in ordered if r["role"] == role and r["class"] == cls)
            if count < 20:
                raise SystemExit(
                    f"role {role}/{cls} has only {count} images; gate requires >= 20")

    split_version = compute_split_version(ordered)
    split_payload = {
        "schema_version": 1,
        "split_version": split_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical": CANONICAL,
        "classes": list(CLASSES),
        "counts": {
            role: {cls: sum(1 for r in ordered if r["role"] == role and r["class"] == cls)
                   for cls in CLASSES}
            for role in ("calibration", "evaluation")
        },
        "corpus_version": corpus_version,
        "records": ordered,
    }

    tokens = make_tokens(ordered)
    tokens_version = compute_tokens_version(tokens)
    token_payload = {
        "schema_version": 1,
        "tokens_version": tokens_version,
        "created_utc": split_payload["created_utc"],
        "records": tokens,
    }

    # Idempotent rerun: identical records must not churn the tracked manifests.
    unchanged = False
    if TEST_SPLIT_PATH.exists() and TOKENS_PATH.exists():
        try:
            existing_split = json.loads(TEST_SPLIT_PATH.read_text(encoding="utf-8"))
            existing_tokens = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
            unchanged = (
                existing_split.get("split_version") == split_version
                and existing_tokens.get("tokens_version") == tokens_version
            )
            if unchanged:
                split_payload["created_utc"] = existing_split.get("created_utc")
                token_payload["created_utc"] = existing_tokens.get("created_utc")
        except (OSError, ValueError):
            unchanged = False
    if not unchanged:
        split_path, token_path = write_manifests(split_payload, token_payload)
        log(f"[build] wrote {split_path}")
        log(f"[build] wrote {token_path}")
    else:
        log("[build] manifests unchanged; left files untouched")
    bytes_total = sum((PROJECT_ROOT / r["path"]).stat().st_size for r in ordered)
    summary = {
        "split_version": split_version,
        "tokens_version": tokens_version,
        "counts": split_payload["counts"],
        "manifest_records": len(ordered),
        "bytes_on_disk": bytes_total,
        "canonical": CANONICAL,
        "per_class": per_class,
        "corpus_version": corpus_version,
        "created_utc": split_payload["created_utc"],
    }
    summary_path = RESULTS_DIR / "phase6_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    grid_path = write_contact_sheet(ordered, RESULTS_DIR / "phase6_splits_grid.png")
    log(f"[build] wrote {grid_path}")

    log("")
    log("=== Phase 6 split build ===")
    for role in ("calibration", "evaluation"):
        counts = split_payload["counts"][role]
        log(f"{role:12s} common={counts['common']:3d} facial={counts['facial']:3d} "
            f"style={counts['style']:3d}")
    log(f"split_version  : {split_version}")
    log(f"tokens_version : {tokens_version}")
    log(f"bytes on disk  : {bytes_total / (1024 * 1024):.1f} MB")
    if DEVIATIONS_PATH.exists():
        log(f"deviations     : {DEVIATIONS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
