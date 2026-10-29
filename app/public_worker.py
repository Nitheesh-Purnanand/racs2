"""Public-side worker for the RACS app: Kpub generation and reference retrieval.

Runs in its own process (CLIP segfaults beside the diffusion stack on this
machine). Reads one JSON request per stdin line, writes one JSON reply per
stdout line:

    {"op": "kpub", "image": "<path>"}                -> {"kpub": "..."}
    {"op": "retrieve", "kpub": "...", "token": "..."} -> {"path", "image_id", "sha256", "timing"}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths  # noqa: E402,F401  (HF cache on D: before model libs)
from PIL import Image  # noqa: E402

from retrieval.racs import Retriever  # noqa: E402
from steganography import pipeline  # noqa: E402


def main() -> int:
    cfg = pipeline.load_config()
    size = int(cfg["steganography"]["image_size"])
    retriever = Retriever(cfg)
    keygen = None
    out = sys.stdout
    sys.stdout = sys.stderr  # keep library prints off the reply channel
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["op"] == "kpub":
                if keygen is None:
                    from semantic.keygen import build_keygen
                    kcfg = dict(cfg["semantic"]["keygen"], device="cpu", dtype="float32")
                    keygen = build_keygen(kcfg)
                img = Image.open(req["image"]).convert("RGB").resize((512, 512), Image.BICUBIC)
                reply = {"kpub": keygen.generate_kpub(img)["kpub"]}
            elif req["op"] == "retrieve":
                # Reference acquisition, end to end: CLIP text query, exact
                # search, rerank, token selection, then reading Xref into the
                # pixel tensor the pipeline consumes.
                t0 = time.perf_counter()
                res = retriever.acquire(req["kpub"], None, req["token"])
                t1 = time.perf_counter()
                pipeline.load_pixels(res["path"], size)
                t2 = time.perf_counter()
                reply = {"path": res["path"], "image_id": res["image_id"],
                         "sha256": res["sha256"],
                         "timing": {"retrieve_s": t1 - t0, "load_xref_s": t2 - t1,
                                    "acquisition_s": t2 - t0}}
            elif req["op"] == "retrieve_image":
                # Cover acquisition for lossless mode: CLIP image query of the
                # secret -> exact Top-K -> token picks one (sender side only).
                from retrieval.racs import select_index
                t0 = time.perf_counter()
                q = retriever.clip.image(Image.open(req["image"]))
                cands = retriever.index.search(q, retriever.top_k)
                row = cands[select_index(req["token"], len(cands), retriever.domain)][0]
                c = retriever.corpus
                reply = {"path": c["paths"][row], "image_id": c["ids"][row],
                         "sha256": c["sha256"][row],
                         "timing": {"acquisition_s": time.perf_counter() - t0}}
            else:
                reply = {"error": f"unknown op {req['op']!r}"}
        except Exception as err:  # noqa: BLE001
            reply = {"error": repr(err)}
        out.write(json.dumps(reply) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
