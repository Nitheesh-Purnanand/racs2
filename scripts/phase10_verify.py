"""Phase 10 — verify the EDICT sweep and operating point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths

import yaml


def main() -> None:
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = config.get("edict", {})
    out_path = paths.PROJECT_ROOT / cfg.get(
        "results_path", "experiments/results/phase10_edict.json"
    )
    checks = []
    failures = []

    def add(name, passed, detail=""):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        if not passed:
            failures.append(name)

    if not out_path.exists():
        print(f"missing {out_path}")
        sys.exit(1)
    with open(out_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    add("schema_version", doc.get("schema_version") == 1)
    add("operating_point", doc.get("operating_point") == {
        "steps": int(cfg.get("steps", 20)), "mixing": float(cfg.get("mixing", 0.93))
    }, json.dumps(doc.get("operating_point")))
    op = doc.get("operating_point_result")
    add("operating_point_present", op is not None)
    if op:
        add("n_random", op.get("n_random", 0) >= int(cfg.get("num_random_latents", 20)),
            str(op.get("n_random")))
        add("n_image", op.get("n_image", 0) >= int(cfg.get("num_image_latents", 20)),
            str(op.get("n_image")))
        add("finite", bool(op.get("finite")))
        threshold = float(cfg.get("max_rel_error", 5e-3))
        add("relative_error", op.get("max_rel_error", 1.0) <= threshold,
            f"{op.get('max_rel_error'):.3e} <= {threshold:.1e}")
        add("absolute_error_recorded", op.get("max_abs_error", 0) > 0,
            f"{op.get('max_abs_error'):.3e}")

    add("resolution_recorded", doc.get("image_size") == int(cfg.get("image_size", 256))
        and doc.get("latent_size") == int(cfg.get("latent_size", 32)),
        f"{doc.get('image_size')}px / {doc.get('latent_size')} latent")
    add("model_recorded", bool(doc.get("model_id")), str(doc.get("model_id")))

    cells = doc.get("cells", [])
    add("sweep_cells", len(cells) >= 2, f"{len(cells)} cells")
    finite_high_mix = all(c.get("finite") for c in cells
                          if c.get("mixing", 0) >= 0.9)
    add("sweep_finite_mixing_ge_0.9", finite_high_mix,
        "mixing <= 0.7 is numerically unstable; 0.93/0.99 are the operating range")

    grid = paths.PROJECT_ROOT / cfg.get(
        "grid_path", "experiments/results/phase10_edict_sweep.png"
    )
    add("grid_exists", grid.exists(), str(grid))

    # No downstream module may leak into the EDICT core.
    source = (paths.PROJECT_ROOT / "steganography" / "edict.py").read_text(encoding="utf-8")
    forbidden = ["retrieval", "faiss", "awgn", "semantic_codec"]
    add("no_downstream_imports", not any(word in source.lower() for word in forbidden))

    result = {
        "pass": not failures,
        "schema_version": 1,
        "operating_point": doc.get("operating_point"),
        "checks": checks,
        "failures": failures,
    }
    out = paths.RESULTS_DIR / "phase10_verify.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nVERIFICATION RESULT: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"written: {out}")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
