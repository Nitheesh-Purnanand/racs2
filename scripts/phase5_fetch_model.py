"""
Phase 5 — fetch the CLIP ViT-B/32 checkpoint into a local model directory.

HuggingFace's own resumable downloader is unreliable on this network: it resets
mid-transfer and discards the partial bytes, so a 605 MB download never finishes.
This helper downloads the weight file in parallel HTTP Range chunks with per-chunk
retries and a resume state file, then verifies the SHA-256 before use. Small
config/tokenizer files are fetched with ``huggingface_hub``.

Usage::

    .venv\\Scripts\\python.exe scripts\\phase5_fetch_model.py

The result is a self-contained directory (default ``models/clip-vit-base-patch32``)
that ``models.clip_encoder`` loads when ``embedding.model_path`` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Redirect the HF cache to D: and disable Xet before huggingface_hub is imported.
import common.paths  # noqa: E402,F401
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

DEFAULT_REPO = "openai/clip-vit-base-patch32"
DEFAULT_REVISION = "main"
WEIGHT_FILE = "pytorch_model.bin"
SMALL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
]


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}", flush=True)


def http_get(url: str, **kwargs) -> requests.Response:
    """GET with a few retries for flaky DNS/connections."""
    last = None
    for attempt in range(1, 6):
        try:
            return requests.get(url, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"request failed after retries: {last}")


def remote_file_metadata(repo_id: str, filename: str) -> tuple[int, str | None]:
    """Return (size, sha256) for a repo file from the HF API."""
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    response = http_get(url, timeout=30)
    response.raise_for_status()
    for sibling in response.json().get("siblings", []):
        if sibling.get("rfilename") == filename:
            size = int(sibling["size"])
            lfs = sibling.get("lfs") or {}
            return size, lfs.get("sha256")
    raise SystemExit(f"{filename!r} not found in {repo_id}")


def load_state(path: Path, total: int, chunk_size: int) -> set[int]:
    if not path.exists():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    if state.get("total") != total or state.get("chunk_size") != chunk_size:
        return set()
    count = (total + chunk_size - 1) // chunk_size
    return {int(i) for i in state.get("done", []) if 0 <= int(i) < count}


def save_state(path: Path, done: set[int], total: int, chunk_size: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "total": total,
        "chunk_size": chunk_size,
        "done": sorted(done),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_weight(
    url: str,
    dest: Path,
    total: int,
    expected_sha: str | None,
    threads: int,
    chunk_size: int,
) -> None:
    state_path = dest.with_suffix(dest.suffix + ".parts.json")
    done = load_state(state_path, total, chunk_size)
    count = (total + chunk_size - 1) // chunk_size

    if dest.exists() and dest.stat().st_size == total:
        if expected_sha is None or sha256_file(dest) == expected_sha:
            print(f"  [ok] complete weight already present: {dest}")
            return
        print("  [warn] existing file has wrong hash; redownloading missing parts")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "ab") as fh:
        fh.truncate(total)

    remaining = [i for i in range(count) if i not in done]
    if not remaining:
        print("  all chunks already recorded; verifying")
    print(f"  chunks: {count} x {chunk_size / 1e6:.1f} MB, "
          f"already done: {len(done)}, remaining: {len(remaining)}")

    work: "queue.Queue[int]" = queue.Queue()
    for index in remaining:
        work.put(index)

    lock = threading.Lock()
    counts = {"done": len(done), "bytes": len(done) * chunk_size}
    started = time.time()
    errors: list[str] = []
    stop = threading.Event()

    def report(force: bool = False) -> None:
        now = time.time()
        elapsed = max(now - started, 1e-6)
        new_bytes = counts["bytes"] - len(done) * chunk_size
        rate = new_bytes / elapsed
        percent = 100.0 * counts["bytes"] / total
        eta = (total - counts["bytes"]) / rate if rate > 0 else None
        print(
            "JCODE_PROGRESS "
            + json.dumps(
                {
                    "current": counts["bytes"],
                    "total": total,
                    "unit": "bytes",
                    "percent": round(percent, 2),
                    "message": f"download {percent:.1f}%",
                    "eta_seconds": int(eta) if eta else None,
                }
            ),
            flush=True,
        )

    def worker(worker_id: int) -> None:
        session = requests.Session()
        session.headers.update({"Accept-Encoding": "identity"})
        while not stop.is_set():
            try:
                index = work.get_nowait()
            except queue.Empty:
                return

            start = index * chunk_size
            end = min(start + chunk_size, total) - 1
            expected = end - start + 1

            for attempt in range(1, 21):
                try:
                    response = session.get(
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(15, 90),
                    )
                    if response.status_code == 200 and start != 0:
                        raise IOError("server ignored Range (200 for non-zero offset)")
                    if response.status_code not in (200, 206):
                        raise IOError(f"HTTP {response.status_code}")

                    buffer = bytearray()
                    for piece in response.iter_content(64 * 1024):
                        buffer.extend(piece)
                    if len(buffer) != expected:
                        raise IOError(f"short read {len(buffer)}/{expected}")
                    with open(dest, "r+b") as fh:
                        fh.seek(start)
                        fh.write(buffer)
                    with lock:
                        done.add(index)
                        counts["done"] += 1
                        counts["bytes"] += expected
                        if counts["done"] % 8 == 0 or counts["done"] == count:
                            report()
                        save_state(state_path, done, total, chunk_size)
                    break
                except Exception as exc:  # noqa: BLE001
                    response = None
                    if attempt == 20:
                        errors.append(f"chunk {index}: {exc}")
                        stop.set()
                        return
                    time.sleep(min(1.5 * attempt, 15))
            work.task_done()

    thread_list = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    for thread in thread_list:
        thread.start()
    try:
        while any(t.is_alive() for t in thread_list):
            for thread in thread_list:
                thread.join(timeout=0.5)
    except KeyboardInterrupt:
        stop.set()
        for thread in thread_list:
            thread.join(timeout=2)
        raise

    if errors:
        raise SystemExit("download failed:\n  " + "\n  ".join(errors[:5]))
    if len(done) != count:
        raise SystemExit(f"download incomplete: {len(done)}/{count} chunks")

    success_path = state_path.with_suffix(".done.json")
    save_state(success_path, done, total, chunk_size)
    state_path.unlink(missing_ok=True)

    if expected_sha is not None:
        print("  verifying sha256 ...", flush=True)
        actual = sha256_file(dest)
        if actual != expected_sha:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"sha256 mismatch:\n  expected {expected_sha}\n  actual   {actual}")
        print(f"  [ok] sha256 {actual}")
    print(f"  [ok] downloaded {total / 1e6:.1f} MB in "
          f"{time.time() - started:.1f}s")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch the CLIP checkpoint for Phase 5.")
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch32"))
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--chunk-mb", type=float, default=2.0)
    p.add_argument("--skip-small", action="store_true",
                   help="only fetch the weight file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    section("1. Resolve remote metadata")
    size, sha = remote_file_metadata(args.repo_id, WEIGHT_FILE)
    print(f"  repo    : {args.repo_id}@{args.revision}")
    print(f"  file    : {WEIGHT_FILE}")
    print(f"  size    : {size / 1e6:.1f} MB")
    print(f"  sha256  : {sha}")

    url = f"https://huggingface.co/{args.repo_id}/resolve/{args.revision}/{WEIGHT_FILE}"
    dest = out_dir / WEIGHT_FILE
    chunk_size = max(int(args.chunk_mb * 1024 * 1024), 256 * 1024)

    section("2. Download weight file")
    download_weight(
        url=url,
        dest=dest,
        total=size,
        expected_sha=sha,
        threads=max(1, int(args.threads)),
        chunk_size=chunk_size,
    )

    if not args.skip_small:
        section("3. Fetch config/tokenizer files")
        for filename in SMALL_FILES:
            path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                revision=args.revision,
                local_dir=str(out_dir),
            )
            print(f"  [ok] {filename}")

    section("RESULT")
    print(f"PASS - local model ready at {out_dir}")
    print("Set configs/base.yaml: embedding.model_path: "
          f"{out_dir.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
