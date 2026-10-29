"""Contribution 2 — exact-recovery layer for generative steganography: evaluation.

Carrier = the paper-style generative stego (c2_make_covers.py). The secret file
is carried in it by three layers, compared on the 66 evaluation images:

    A  naive      : 2-LSB replacement of the original file (the app's first version)
    B  +compact   : A, with the secret losslessly re-encoded first (ablation)
    C  proposed   : texture-adaptive +-1 embedding of the compacted secret

Metrics: pixel-exact recovery, wrong-key rejection, payload, carrier distortion
(PSNR stego vs the same carrier without payload), and detectability by the
chi-square and SPA attacks (AUC over the 66 carrier/stego pairs; 0.5 = chance).
The paper's approximate recovery (Phase 15, same split) is reported beside it.
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from app import lossless as L  # noqa: E402
from security.lsb_detect import auc, chi_square_p, spa_rate  # noqa: E402

R = paths.RESULTS_DIR
COVERS = R / "c2" / "covers"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def roundtrip(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Image.open(io.BytesIO(buf.getvalue()))


def main() -> int:
    split = json.loads((paths.METADATA_DIR / "test_split.json").read_text("utf-8"))
    tokens = {r["image_id"]: r for r in json.loads(paths.TOKENS_PATH.read_text())["records"]}
    evals = [r for r in split["records"] if r["role"] == "evaluation"
             and (COVERS / f"{r['image_id']}.png").exists()]
    methods = {"A_naive_2lsb": [], "B_naive_compact": [], "C_adaptive_compact": [],
               "D_adaptive_compact_dither": []}
    for r in evals:
        iid, key = r["image_id"], tokens[r["image_id"]]["token"]
        wrong = tokens[iid]["wrong_token"]
        secret_bytes = (_ROOT / r["path"]).read_bytes()
        secret_px = pixels(secret_bytes)
        cover = Image.open(COVERS / f"{iid}.png").convert("RGB")
        cname, cdata = L.compact(Path(r["path"]).name, secret_bytes)
        for m in methods:
            name, data = ((Path(r["path"]).name, secret_bytes) if m == "A_naive_2lsb"
                          else (cname, cdata))
            t0 = time.perf_counter()
            adaptive = m.startswith(("C", "D"))
            noise = L.ADAPT_NOISE if m.startswith("D") else 0.0
            if adaptive:
                st = L.embed_adaptive(cover, name, data, key, noise=noise)
            else:
                st = L.embed(cover, name, data, key, keep_aspect=True)
            emb_s = time.perf_counter() - t0
            st = roundtrip(st)  # through a real PNG file, as a phone would send it
            t1 = time.perf_counter()
            _, got = (L.extract_adaptive if adaptive else L.extract)(st, key)
            ext_s = time.perf_counter() - t1
            try:
                (L.extract_adaptive if adaptive else L.extract)(st, wrong)
                rejected = False
            except ValueError:
                rejected = True
            sarr = np.asarray(st.convert("RGB"))
            # the same carrier with no payload (same upscale, same dither)
            if adaptive:
                carr = L.prepare_carrier(cover, st.size, L.carrier_seed(key), noise)
            else:
                carr = (np.asarray(cover.resize(st.size, Image.LANCZOS))
                        if cover.size != st.size else np.asarray(cover))
            methods[m].append({
                "image_id": iid, "class": r["class"],
                "pixel_exact": bool(np.array_equal(pixels(got), secret_px)),
                "file_exact": got == secret_bytes,
                "wrong_key_rejected": rejected,
                "payload_bytes": len(data), "carrier_px": st.size[0] * st.size[1],
                "bits_per_value": len(data) * 8 / sarr.size,
                "psnr_vs_carrier": psnr(sarr, carr),
                "chi_cover": chi_square_p(carr), "chi_stego": chi_square_p(sarr),
                "spa_cover": spa_rate(carr), "spa_stego": spa_rate(sarr),
                "embed_s": emb_s, "extract_s": ext_s})
        print(f"[c2] {iid} done", flush=True)

    summary = {}
    for m, rows in methods.items():
        f = lambda k: float(np.mean([x[k] for x in rows]))  # noqa: E731
        summary[m] = {
            "n": len(rows),
            "pixel_exact_rate": f("pixel_exact"), "file_exact_rate": f("file_exact"),
            "wrong_key_rejection_rate": f("wrong_key_rejected"),
            "payload_kb_mean": f("payload_bytes") / 1024,
            "carrier_mpx_mean": f("carrier_px") / 1e6,
            "bits_per_value_mean": f("bits_per_value"),
            "psnr_vs_carrier_db": f("psnr_vs_carrier"),
            "chi_square_auc": auc([x["chi_cover"] for x in rows], [x["chi_stego"] for x in rows]),
            "spa_auc": auc([x["spa_cover"] for x in rows], [x["spa_stego"] for x in rows]),
            "spa_rate_cover_mean": f("spa_cover"), "spa_rate_stego_mean": f("spa_stego"),
            "embed_s_mean": f("embed_s"), "extract_s_mean": f("extract_s")}

    paper_psnr = None
    bt = R / "baseline_table.json"
    if bt.exists():
        cells = json.loads(bt.read_text())["cells"]
        vals = [c["psnr"]["mean"] if isinstance(c["psnr"], dict) else c["psnr"] for c in cells]
        paper_psnr = float(np.mean(vals))
    payload = {"schema_version": 1, "contribution": 2,
               "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "carrier": "EDICT stego, strength 0.5, RACS-retrieved Xref, 256 px",
               "generative_recovery_psnr_db": paper_psnr, "summary": summary,
               "cases": methods}
    (R / "c2").mkdir(exist_ok=True)
    (R / "c2_exact_recovery.json").write_text(json.dumps(payload, indent=2))

    names = {"A_naive_2lsb": "A naive 2-LSB, original file",
             "B_naive_compact": "B naive 2-LSB + lossless compaction",
             "C_adaptive_compact": "C adaptive +-1 + compaction",
             "D_adaptive_compact_dither": "D **C + dithered carrier (proposed)**"}
    md = ["# Contribution 2 — Exact-recovery layer for generative steganography", "",
          f"Evaluation split, n = {len(evals)} secrets. Carrier: paper-style EDICT stego "
          "(strength 0.5) with a RACS-retrieved reference.", "",
          f"Generative recovery alone (Phase 15, approximate): **{paper_psnr:.2f} dB** mean PSNR. "
          "With the layer below the secret is recovered **exactly**.", "",
          "| layer | pixel-exact | wrong key rejected | payload KB | bits/value | PSNR vs carrier dB "
          "| chi-square AUC | SPA AUC | embed s | extract s |",
          "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |"]
    for m, s in summary.items():
        md.append(f"| {names[m]} | {s['pixel_exact_rate']:.0%} | {s['wrong_key_rejection_rate']:.0%} "
                  f"| {s['payload_kb_mean']:.0f} | {s['bits_per_value_mean']:.3f} "
                  f"| {s['psnr_vs_carrier_db']:.2f} | {s['chi_square_auc']:.3f} | {s['spa_auc']:.3f} "
                  f"| {s['embed_s_mean']:.2f} | {s['extract_s_mean']:.2f} |")
    md += ["", "AUC = probability the detector scores the stego above its own clean carrier; "
           "0.5 is chance, 1.0 is always detected. Classical structural detectors only; a "
           "trained CNN steganalyser is future work."]
    (R / "c2_exact_recovery.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
