"""Phase 8 — offline verification of reference artifacts and determinism."""

from __future__ import annotations

import json
import gc
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
    set_seed(42, verbose=True)
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
    for cls in CLASSES:
        pipe = None
        try:
            todo = [r for r in records if r["class"] == cls and
                    r in samples.get((r["role"], cls), [])]
            if not todo:
                continue
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
                stored = Image.open(paths.PROJECT_ROOT / rec["reference_path"]).convert("RGB")
                stored_digest = __import__("hashlib").sha256(stored.tobytes()).hexdigest()
                if digest != stored_digest:
                    det_ok = False
                    failures.append(f"determinism:{rec['image_id']}")
                if out["latents_sha256"] != rec["latents_sha256"]:
                    det_ok = False
                    failures.append(f"latents:{rec['image_id']}")
        finally:
            if pipe is not None:
                unload(pipe)
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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


if __name__ == "__main__":
    main()
