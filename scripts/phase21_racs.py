"""Phase 21 — RACS-Base end to end, measured beside the baseline on the same cases.

1. Xref_tx == Xref_rx for every evaluation image (a fresh receiver-side
   Retriever recomputes each reference; file and tensor SHA-256 compared).
2. Both systems run through one object and the imported baseline ``run_case``;
   only ``reference_path`` differs (Phase 8 generated vs retrieved).
3. Security at 10 dB: correct / wrong / no_token for both systems, plus RACS
   with the wrong token also selecting the wrong reference.

``--limit N`` takes the first N evaluation images per class (default 2) to keep
the run short on a 6 GB card; the full split is ``--limit 0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from common.seed import set_seed  # noqa: E402
from pipelines import agentsemstecom  # noqa: E402
from pipelines.racs_base import RACSBase  # noqa: E402
from retrieval.racs import Retriever, sha256_bytes  # noqa: E402
from steganography import pipeline  # noqa: E402

R = paths.RESULTS_DIR
METRICS = ("mse", "psnr", "ssim", "lpips")


def acquire_only(out: Path) -> int:
    """One side (Tx or Rx): retrieve Xref for every evaluation image, both tokens."""
    cfg = pipeline.load_config()
    retr = Retriever(cfg)
    split = json.loads((paths.METADATA_DIR / "test_split.json").read_text("utf-8"))
    tokens = {r["image_id"]: r for r in pipeline.read_json(paths.TOKENS_PATH)["records"]}
    res = {}
    for r in split["records"]:
        if r["role"] != "evaluation":
            continue
        iid = r["image_id"]
        with np.load(R / "public_conditions" / f"{iid}.npz", allow_pickle=True) as z:
            kpub = str(z["kpub"])
            feat = z["feature"] if "feature" in z.files else None
        for tok in (tokens[iid]["token"], tokens[iid]["wrong_token"]):
            t0 = time.perf_counter()
            o = retr.acquire(kpub, feat, tok)
            o["latency_s"] = time.perf_counter() - t0
            res[f"{iid}|{tok}"] = o
    pipeline.write_json(out, res)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--security-snr", type=float, default=10.0)
    ap.add_argument("--acquire-only", default="")
    args = ap.parse_args()
    if args.acquire_only:
        return acquire_only(Path(args.acquire_only))

    started = time.time()
    cfg = pipeline.load_config()
    set_seed(int(cfg.get("seed", 42)))
    split = json.loads((paths.METADATA_DIR / "test_split.json").read_text("utf-8"))
    evals = [r for r in split["records"] if r["role"] == "evaluation"]
    size = int(cfg["steganography"]["image_size"])
    snrs = [float(s) for s in cfg["baseline"].get("snr_db", cfg["channel"]["snr_db"])]

    # ---- 1. Tx/Rx invariant: two independent processes, whole evaluation split ----
    acq_dir = R / "phase21"
    acq_dir.mkdir(parents=True, exist_ok=True)
    for side in ("tx", "rx"):
        subprocess.check_call([sys.executable, __file__, "--acquire-only",
                               str(acq_dir / f"acquire_{side}.json")])
    tx_all = pipeline.read_json(acq_dir / "acquire_tx.json")
    rx_all = pipeline.read_json(acq_dir / "acquire_rx.json")
    tokens = {r["image_id"]: r for r in pipeline.read_json(paths.TOKENS_PATH)["records"]}
    invariant = []
    for r in evals:
        iid = r["image_id"]
        key = f"{iid}|{tokens[iid]['token']}"
        tx, rxo = tx_all[key], rx_all[key]
        file_eq = (sha256_bytes(Path(tx["path"]).read_bytes())
                   == sha256_bytes(Path(rxo["path"]).read_bytes()) == tx["sha256"])
        ten_eq = (agentsemstecom.tensor_hash(pipeline.load_pixels(tx["path"], size))
                  == agentsemstecom.tensor_hash(pipeline.load_pixels(rxo["path"], size)))
        invariant.append({"image_id": iid, "class": r["class"], "xref_id": tx["image_id"],
                          "xref_sha256": tx["sha256"], "pick": tx["pick"],
                          "file_equal": file_eq, "tensor_equal": ten_eq,
                          "retrieval_latency_s": tx["latency_s"]})
    inv_ok = all(x["file_equal"] and x["tensor_equal"] for x in invariant)
    print(f"[phase21] Xref_tx == Xref_rx: {sum(x['file_equal'] and x['tensor_equal'] for x in invariant)}"
          f"/{len(invariant)}", flush=True)
    sysobj = RACSBase(cfg, acquisitions={tuple(k.split("|")): v for k, v in tx_all.items()})

    # ---- 2. quality on the same cases, both systems ----
    p8 = {r["image_id"]: r for r in pipeline.read_json(R / "phase8_references.json")["records"]
          if r["role"] == "evaluation"}
    cases = []
    for cls in ("common", "facial", "style"):
        cases += [r for r in evals if r["class"] == cls][: args.limit or None]
    out_dir = R / "phase21"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, security = [], []
    for r in cases:
        iid, cls = r["image_id"], r["class"]
        tok = sysobj.tokens(iid)
        secret = pipeline.load_pixels(str(_ROOT / r["path"]), size)
        for system in ("agentsemstecom", "racs_base"):
            sysobj.generated = system == "agentsemstecom"
            ref_path = sysobj.reference_path("evaluation", cls, iid)
            if system == "racs_base":
                ref_lat = sysobj.acquire(iid, tok["token"])["latency_s"]
                ref_vram = 0.0
            else:
                ref_lat = float(p8[iid]["latency_s"])
                ref_vram = float(p8[iid]["peak_vram_gb"])
            t0 = time.perf_counter()
            hidden = sysobj.hide(secret, pipeline.load_pixels(ref_path, size),
                                 sysobj.conditions(iid)["kpub"], tok["token"],
                                 reference_path=ref_path)
            hide_s = time.perf_counter() - t0
            pipeline.save_pixels(hidden["stego_pixels"], out_dir / f"{system}_{iid}_stego.png")
            for snr in snrs:
                t1 = time.perf_counter()
                out = agentsemstecom.run_case(
                    baseline=sysobj, secret_px=secret, class_name=cls, image_id=iid,
                    token=tok["token"], wrong_token=tok["wrong_token"], snr_db=snr,
                    split_role="evaluation", hidden=hidden)
                pipeline.save_pixels(out["recovered"],
                                     out_dir / f"{system}_{iid}_snr{int(snr)}_recovered.png")
                rows.append({"system": system, "image_id": iid, "class": cls,
                             "snr_db": int(snr), "role": "evaluation",
                             **{m: out[m] for m in METRICS if m in out},
                             "reference_sha256": out["reference_sha256"],
                             "reference_latency_s": ref_lat, "reference_peak_vram_gb": ref_vram,
                             "total_latency_s": ref_lat + hide_s + time.perf_counter() - t1})
                print(f"[phase21] {system:15s} {iid} snr={snr:>4.0f} psnr={out['psnr']:.2f}",
                      flush=True)
            for mode in ("wrong", "no_token"):
                o = agentsemstecom.run_case(
                    baseline=sysobj, secret_px=secret, class_name=cls, image_id=iid,
                    token=tok["token"], wrong_token=tok["wrong_token"],
                    snr_db=args.security_snr, split_role="evaluation",
                    token_mode=mode, hidden=hidden)
                security.append({"system": system, "image_id": iid, "class": cls,
                                 "token_mode": mode, "psnr": o["psnr"], "ssim": o["ssim"]})
            if system == "racs_base":  # wrong token also retrieves the wrong Xref
                wrong_ref = sysobj.acquire(iid, tok["wrong_token"])["path"]
                tx = sysobj.transmit(hidden["stego_pixels"], role="evaluation",
                                     class_name=cls, snr_db=args.security_snr)
                rec = sysobj.receive(tx["output"], token_mode="wrong", token=tok["token"],
                                     wrong_token=tok["wrong_token"],
                                     prompt=sysobj.conditions(iid)["kpub"],
                                     blob={k: hidden[k] for k in
                                           ("delta", "noise", "eta", "steps", "mixing")},
                                     image_embeds=sysobj.embeds_from_path(wrong_ref))
                m = pipeline.metrics(secret, rec, sysobj.stego_cfg)
                security.append({"system": system, "image_id": iid, "class": cls,
                                 "token_mode": "wrong_with_wrong_ref", "psnr": m["psnr"],
                                 "ssim": m["ssim"], "wrong_ref_differs":
                                 Path(wrong_ref).resolve() != Path(ref_path).resolve()})
            correct = next(x for x in rows if x["system"] == system and x["image_id"] == iid
                           and x["snr_db"] == int(args.security_snr))
            security.append({"system": system, "image_id": iid, "class": cls,
                             "token_mode": "correct", "psnr": correct["psnr"],
                             "ssim": correct["ssim"]})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    racs_cfg = cfg["retrieval"]
    p17 = pipeline.read_json(R / "phase17_index_manifest.json")
    p19 = pipeline.read_json(R / "phase19_structural.json")
    import inspect
    downstream_same = (RACSBase.hide is agentsemstecom.Baseline.hide
                       and RACSBase.transmit is agentsemstecom.Baseline.transmit
                       and RACSBase.receive is agentsemstecom.Baseline.receive
                       and RACSBase.maybe_enhance is agentsemstecom.Baseline.maybe_enhance
                       and "def run_case" not in inspect.getsource(
                           sys.modules["pipelines.racs_base"]))
    table = {"schema_version": 1, "phase": 21, "role": "evaluation", "created_utc": now,
             "limit_per_class": args.limit, "snr_db": [int(s) for s in snrs],
             "rows": rows, "security": security}
    pipeline.write_json(R / "racs_table.json", table)
    manifest = {
        "schema_version": 1, "phase": 21, "created_utc": now,
        "index_sha256": p17["index_sha256"], "embeddings_sha256": p17["embeddings_sha256"],
        "corpus_version": p17["corpus_version"],
        "structural_maps_sha256": p19["structural_maps_sha256"],
        "alpha": racs_cfg["ranking"]["alpha"], "beta": racs_cfg["ranking"]["beta"],
        "top_k": racs_cfg["ranking"]["top_k"], "pool_k": racs_cfg["ranking"]["pool_k"],
        "token_domain": racs_cfg["token_selection"]["domain"],
        "split_version": split["split_version"],
        "tokens_version": pipeline.read_json(paths.TOKENS_PATH)["tokens_version"],
        "xref_invariant": invariant, "table_sha256": hashlib.sha256(
            (R / "racs_table.json").read_bytes()).hexdigest(),
        "elapsed_s": time.time() - started,
    }
    pipeline.write_json(R / "racs_manifest.json", manifest)
    checks = {"xref_tx_equals_rx_all_eval_images": inv_ok,
              "downstream_is_imported_baseline": downstream_same,
              "evaluation_split_only": all(r["role"] == "evaluation" for r in cases),
              "all_cases_measured": len(rows) == 2 * len(cases) * len(snrs)}
    ok = all(checks.values())
    pipeline.write_json(R / "phase21_verify.json",
                        {"phase": 21, "pass": ok, "checks": checks, "created_utc": now,
                         "cases": len(cases), "note": "subset run" if args.limit else "full"})
    print(f"[phase21] {'PASS' if ok else 'FAIL'} {checks} ({time.time() - started:.0f}s)")
    sysobj.edict.unload()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
