"""Fetch the paper's real components into ``models/`` with hashes.

The baseline paper (AgentSemSteCom) names PicX-Real, IPAdapter-plus,
OneFormer, an LLM for the public semantic key, SwinJSCC and CodeFormer.
Earlier phases substituted local models because these weights had not been
fetched. This script downloads them with the same chunked resumable
downloader used for CLIP, then writes ``models/PAPER_ASSETS.json`` with
sizes and SHA-256 values so every downstream manifest can pin them.

Usage:
    .venv\\Scripts\\python.exe scripts\\fetch_paper_assets.py
    .venv\\Scripts\\python.exe scripts\\fetch_paper_assets.py --only picx_real
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.paths  # noqa: E402,F401

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import requests  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from scripts.phase5_fetch_model import (  # noqa: E402
    download_weight,
    remote_file_metadata,
    sha256_file,
)

MODELS = PROJECT_ROOT / "models"
MANIFEST = MODELS / "PAPER_ASSETS.json"

# repo -> (local dir, allow prefixes). ``allow=None`` means all files.
HF_ASSETS = {
    "picx_real": ("picx-real", "GraydientPlatformAPI/picx-real", [
        "model_index.json",
        "scheduler/",
        "tokenizer/",
        "text_encoder/",
        "unet/",
        "vae/",
    ]),
    "ipadapter": ("ip-adapter", "h94/IP-Adapter", [
        "models/ip-adapter-plus_sd15.safetensors",
        "models/image_encoder/",
    ]),
    "oneformer": ("oneformer-ade20k-swin-large", "shi-labs/oneformer_ade20k_swin_large", [
        "config.json",
        "pytorch_model.bin",
        "preprocessor_config.json",
        "processor_config.json",
    ]),
    "qwen": ("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", None),
}

CODEFORMER = {
    "codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "detection_Resnet50_Final.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing_parsenet.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
}

SWINJSCC = {
    # Closest public checkpoint to the paper's C=48 CelebA-HQ / 10 dB model.
    # The release has C=32/64/96/128/192; C=64 is the nearest available.
    "SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C64.model":
        "1PMYjEjU-N3q9Q6xcCAu2jnGbLvEI3hiv",
}


def log(msg: str) -> None:
    print(f"[assets] {msg}", flush=True)


def allowed(name: str, prefixes) -> bool:
    if prefixes is None:
        return True
    return any(name == p or name.startswith(p) for p in prefixes)


def fetch_hf(key: str, subdir: str, repo: str, prefixes) -> dict:
    out_dir = MODELS / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    api = requests.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60)
    api.raise_for_status()
    siblings = [s for s in api.json().get("siblings", [])]
    files = []
    for sibling in siblings:
        name = sibling.get("rfilename", "")
        if not name or not allowed(name, prefixes):
            continue
        size = int(sibling.get("size") or 0)
        if size == 0 and name.endswith("/"):
            continue
        files.append(name)

    records = []
    for index, name in enumerate(sorted(files), 1):
        dest = out_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            size, remote_sha = remote_file_metadata(repo, name)
        except SystemExit:
            log(f"  skip (no metadata) {name}")
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        log(f"  [{index}/{len(files)}] {repo}/{name} {size / 1e6:.1f} MB")
        if size >= 8 * 1024 * 1024:
            download_weight(
                url=url,
                dest=dest,
                total=size,
                expected_sha=remote_sha,
                threads=8,
                chunk_size=4 * 1024 * 1024,
            )
        else:
            hf_hub_download(repo_id=repo, filename=name, local_dir=str(out_dir))
        records.append({
            "path": dest.relative_to(PROJECT_ROOT).as_posix(),
            "size": dest.stat().st_size,
            "sha256": sha256_file(dest),
        })
    return {"repo": repo, "dir": out_dir.relative_to(PROJECT_ROOT).as_posix(),
            "files": records}


def fetch_url(name: str, url: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("content-length") or 0)
    dest = out_dir / name
    log(f"  {name} {total / 1e6:.1f} MB")
    download_weight(url=url, dest=dest, total=total, expected_sha=None,
                    threads=6, chunk_size=4 * 1024 * 1024)
    return {
        "path": dest.relative_to(PROJECT_ROOT).as_posix(),
        "size": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "source": url,
    }


def fetch_gdrive(name: str, file_id: str, out_dir: Path) -> dict:
    import gdown

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name
    log(f"  {name} (Google Drive {file_id})")
    gdown.download(id=file_id, output=str(dest), quiet=False, resume=True)
    if not dest.exists():
        raise RuntimeError(f"gdown did not produce {dest}")
    return {
        "path": dest.relative_to(PROJECT_ROOT).as_posix(),
        "size": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "source": f"gdrive:{file_id}",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated asset keys")
    args = parser.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    started = time.time()
    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.setdefault("assets", {})

    for key, (subdir, repo, prefixes) in HF_ASSETS.items():
        if only and key not in only:
            continue
        log(f"== {key} ({repo}) ==")
        manifest["assets"][key] = fetch_hf(key, subdir, repo, prefixes)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not only or "codeformer" in only:
        log("== codeformer ==")
        records = [fetch_url(name, url, MODELS / "codeformer")
                   for name, url in CODEFORMER.items()]
        manifest["assets"]["codeformer"] = {"repo": "sczhou/CodeFormer", "files": records}
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not only or "swinjscc" in only:
        log("== swinjscc ==")
        records = [fetch_gdrive(name, fid, MODELS / "swinjscc")
                   for name, fid in SWINJSCC.items()]
        manifest["assets"]["swinjscc"] = {"repo": "semcomm/SwinJSCC", "files": records}
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["elapsed_s"] = round(time.time() - started, 1)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"wrote {MANIFEST.relative_to(PROJECT_ROOT)} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
