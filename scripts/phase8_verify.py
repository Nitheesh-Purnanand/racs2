"""Phase 8 — offline verification of reference artifacts and determinism."""

from __future__ import annotations

import json
import gc
import subprocess
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
from PIL import Image

from common.seed import set_seed
from evaluation.splits import CLASSES, Split, load_tokens, sha256_file
from reference.conditions import control_image, load_conditions
from reference.generate import build_pipeline, generate_reference, unload
from reference.seed import derive_seed
from semantic.cache import feature_sha256, read_cache_npz, sha256_string


def check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": name, "pass": bool(passed), "detail": detail}


def main() -> None:
    # CPU-only seeding: creating a CUDA context in the parent would take ~0.7 GB
    # of VRAM away from the per-class determinism workers on a 6 GB card.
    torch.manual_seed(42)
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = config.get("reference", {})
    manifest_path = paths.PROJECT_ROOT / cfg.get(
        "manifest_path", "experiments/results/phase8_references.json"
    )
    checks: List[Dict[str, Any]] = []
    failures: List[str] = []

    def add(name, passed, detail=""):
        checks.append(check(name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        if not passed:
            failures.append(name)

    if not manifest_path.exists():
        print(f"manifest missing: {manifest_path}")
        sys.exit(1)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    records = manifest.get("records", [])

    print("=" * 60)
    print("1. MANIFEST")
    print("=" * 60)
    add("schema_version", manifest.get("schema_version") == 1)
    split = Split.load(role="evaluation", verify=False)
    cal = Split.load(role="calibration", allow_calibration=True, verify=False)
    add("split_version", manifest.get("split_version") == split.version,
        manifest.get("split_version", ""))
    tokens, tokens_version = load_tokens()
    add("tokens_version", manifest.get("tokens_version") == tokens_version)
    add("config_hash", bool(manifest.get("config_hash")))
    counts = defaultdict(lambda: defaultdict(int))
    for rec in records:
        counts[rec["role"]][rec["class"]] += 1
    all_counts = all(counts[r][c] == 22 for r in ("calibration", "evaluation")
                     for c in CLASSES)
    add("counts_22", all_counts, json.dumps({r: dict(v) for r, v in counts.items()}))
    ids = [r["image_id"] for r in records]
    add("unique_ids", len(ids) == len(set(ids)), str(len(ids)))

    print("=" * 60)
    print("2. FILES / HASHES / SEEDS")
    print("=" * 60)
    file_ok = True
    seed_ok = True
    for rec in records:
        ref = paths.PROJECT_ROOT / rec["reference_path"]
        if not ref.exists() or sha256_file(ref) != rec["reference_sha256"]:
            file_ok = False
            failures.append(f"file:{rec['image_id']}")
        token = tokens[rec["image_id"]][
            "token" if rec["token_mode"] == "correct" else "wrong_token"
        ]
        if derive_seed(token) != rec["seed"]:
            seed_ok = False
            failures.append(f"seed:{rec['image_id']}")
    add("reference_files", file_ok, f"{len(records)} records")
    add("seed_recompute", seed_ok)

    print("=" * 60)
    print("3. CONDITION LINKAGE")
    print("=" * 60)
    cond_ok = True
    for rec in records:
        cond = load_conditions(paths.RESULTS_DIR / "public_conditions", rec["image_id"])
        if sha256_string(cond["kpub"]) != rec["kpub_sha256"]:
            cond_ok = False
        if feature_sha256(cond.get("feature")) != rec["feature_sha256"]:
            cond_ok = False
        if rec["class"] == "style" and cond["feature_kind"] != "none":
            cond_ok = False
    add("condition_linkage", cond_ok)

    print("=" * 60)
    print("4. DETERMINISM (regenerate 2/class/role)")
    print("=" * 60)
    size = int(cfg.get("output_size", 256))
    steps = int(cfg.get("steps", 20))
    guidance = float(cfg.get("guidance_scale", 7.5))
    samples = defaultdict(list)
    for rec in records:
        key = (rec["role"], rec["class"])
        if len(samples[key]) < 2:
            samples[key].append(rec)
    det_ok = True

    # Run each class in a subprocess to avoid GPU OOM on 6 GB cards.
    # Loading SD + ControlNet for one class uses ~3.4 GB; loading a second
    # class without a full process restart can OOM due to CUDA fragmentation
    # and Windows commit-charge accounting.
    for cls in CLASSES:
        todo = [r for r in records if r["class"] == cls and
                r in samples.get((r["role"], cls), [])]
        if not todo:
            continue
        # Serialise the todo list to a temp file for the subprocess
        import tempfile
        todo_file = Path(tempfile.mktemp(suffix=".json", prefix="phase8_det_"))
        with open(todo_file, "w", encoding="utf-8") as tf:
            json.dump({"class": cls, "records": todo, "size": size,
                       "steps": steps, "guidance": guidance,
                       "config": cfg}, tf)
        try:
            result = subprocess.run(
                [sys.executable, "-X", "faulthandler",
                 str(Path(__file__).resolve()), "--determinism-worker",
                 str(todo_file)],
                cwd=_ROOT, capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                # Parse any failure lines from stdout
                for line in result.stdout.strip().splitlines():
                    if line.startswith("FAIL:"):
                        det_ok = False
                        failures.append(line[5:])
                if not any(l.startswith("FAIL:") for l in result.stdout.strip().splitlines()):
                    det_ok = False
                    failures.append(f"determinism_subprocess:{cls}")
                    print(f"  subprocess for {cls} failed (rc={result.returncode})")
                    if result.stderr:
                        for line in result.stderr.strip().splitlines()[-5:]:
                            print(f"    {line}")
            else:
                for line in result.stdout.strip().splitlines():
                    if line.startswith("FAIL:"):
                        det_ok = False
                        failures.append(line[5:])
                    elif line.startswith("OK:"):
                        print(f"    {line}")
        except subprocess.TimeoutExpired:
            det_ok = False
            failures.append(f"determinism_timeout:{cls}")
        finally:
            if todo_file.exists():
                todo_file.unlink()
    add("determinism", det_ok)

    print("=" * 60)
    print("5. VISUAL / BUDGET")
    print("=" * 60)
    grid = paths.RESULTS_DIR / "phase8_reference_grid.png"
    add("grid_exists", grid.exists(), str(grid))
    peaks = [r["peak_vram_gb"] for r in records if r.get("peak_vram_gb") is not None]
    budget = float(cfg.get("max_peak_vram_gb", 5.5))
    add("vram_budget", all(p <= budget for p in peaks) if peaks else True,
        f"max={max(peaks) if peaks else 'n/a'} <= {budget}")

    result = {
        "pass": not failures,
        "schema_version": 1,
        "split_version": manifest.get("split_version"),
        "config_hash": manifest.get("config_hash"),
        "checks": checks,
        "counts": {r: dict(v) for r, v in counts.items()},
        "failures": sorted(set(failures)),
        "elapsed_s": None,
    }
    out = paths.RESULTS_DIR / "phase8_verify.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("=" * 60)
    print(f"VERIFICATION RESULT: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"checks={len(checks)} failures={len(result['failures'])}")
    print(f"written: {out}")
    sys.exit(0 if result["pass"] else 1)


def _determinism_worker(todo_path: str) -> None:
    """Subprocess worker: verify determinism for one class's records."""
    with open(todo_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cls = payload["class"]
    todo = payload["records"]
    size = int(payload["size"])
    steps = int(payload["steps"])
    guidance = float(payload["guidance"])
    cfg = payload["config"]

    tokens, _ = load_tokens()
    pipe = None
    try:
        pipe, _ = build_pipeline(cls, cfg)
        for rec in todo:
            cond = load_conditions(paths.RESULTS_DIR / "public_conditions",
                                   rec["image_id"])
            token = tokens[rec["image_id"]][
                "token" if rec["token_mode"] == "correct" else "wrong_token"
            ]
            out = generate_reference(
                pipe, prompt=cond["kpub"], control=control_image(cond, size),
                token=token, output_size=size, steps=steps,
                guidance_scale=guidance,
            )
            digest = __import__("hashlib").sha256(
                out["image"].tobytes()
            ).hexdigest()
            stored = Image.open(
                paths.PROJECT_ROOT / rec["reference_path"]
            ).convert("RGB")
            stored_digest = __import__("hashlib").sha256(
                stored.tobytes()
            ).hexdigest()
            img_ok = (digest == stored_digest)
            lat_ok = (out["latents_sha256"] == rec["latents_sha256"])
            if not img_ok:
                print(f"FAIL:determinism:{rec['image_id']}", flush=True)
            if not lat_ok:
                print(f"FAIL:latents:{rec['image_id']}", flush=True)
            if img_ok and lat_ok:
                print(f"OK: {rec['image_id']} deterministic", flush=True)
    finally:
        if pipe is not None:
            unload(pipe)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if "--determinism-worker" in sys.argv:
        idx = sys.argv.index("--determinism-worker")
        _determinism_worker(sys.argv[idx + 1])
    else:
        main()
