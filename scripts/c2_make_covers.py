"""Contribution 2, step 1 — paper-style generative stego covers for the evaluation split.

For every evaluation image: RACS-retrieved Xref (Phase 21 Tx acquisitions),
EDICT hide at strength 0.5 with IPAdapter, saved as the cover that the
exact-recovery layer then carries the secret in. Resumable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from pipelines.agentsemstecom import Baseline  # noqa: E402
from steganography import pipeline  # noqa: E402

OUT = paths.RESULTS_DIR / "c2" / "covers"


def main() -> int:
    cfg = pipeline.load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    acq = pipeline.read_json(paths.RESULTS_DIR / "phase21" / "acquire_tx.json")
    split = json.loads((paths.METADATA_DIR / "test_split.json").read_text("utf-8"))
    tokens = {r["image_id"]: r for r in pipeline.read_json(paths.TOKENS_PATH)["records"]}
    b = Baseline(cfg)
    b.stego_cfg["edict_strength"] = 0.5
    size = int(b.stego_cfg["image_size"])
    for r in split["records"]:
        if r["role"] != "evaluation":
            continue
        iid = r["image_id"]
        out = OUT / f"{iid}.png"
        if out.exists():
            continue
        t0 = time.time()
        tok = tokens[iid]["token"]
        ref = acq[f"{iid}|{tok}"]["path"]
        h = b.hide(pipeline.load_pixels(str(_ROOT / r["path"]), size),
                   pipeline.load_pixels(ref, size), b.conditions(iid)["kpub"], tok,
                   reference_path=ref)
        pipeline.save_pixels(h["stego_pixels"], out)
        print(f"[c2] cover {iid} ({time.time() - t0:.1f}s)", flush=True)
    b.edict.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
