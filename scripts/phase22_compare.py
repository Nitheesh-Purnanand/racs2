"""Phase 22 — controlled comparison: AgentSemSteCom vs RACS-Base (primary result).

Reads ``racs_table.json`` (Phase 21), where both systems were measured on the
same evaluation cases through the same code path, and writes
``phase22_comparison.{json,md}`` and ``phase22_grid.png``. Refuses
calibration data. Paper numbers are never mixed into measured rows.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from steganography import pipeline  # noqa: E402

R = paths.RESULTS_DIR
SYSTEMS = ("agentsemstecom", "racs_base")
METRICS = ("psnr", "ssim", "mse", "lpips", "reference_latency_s", "total_latency_s",
           "reference_peak_vram_gb")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    table = pipeline.read_json(R / "racs_table.json")
    verify21 = pipeline.read_json(R / "phase21_verify.json")
    rows = table["rows"]
    if table.get("role") != "evaluation" or any(r["role"] != "evaluation" for r in rows):
        raise SystemExit("phase22 refuses non-evaluation data")

    cells = defaultdict(list)
    for r in rows:
        cells[(r["system"], r["class"], r["snr_db"])].append(r)
    out_rows = []
    for (system, cls, snr), rs in sorted(cells.items()):
        out_rows.append({"system": system, "class": cls, "snr_db": snr, "n": len(rs),
                         **{m: mean([x.get(m) for x in rs]) for m in METRICS}})
    overall = {s: {m: mean([r.get(m) for r in rows if r["system"] == s]) for m in METRICS}
               for s in SYSTEMS}
    sec = defaultdict(list)
    for s in table["security"]:
        sec[(s["system"], s["token_mode"])].append(s["psnr"])
    security = {f"{s}|{m}": mean(v) for (s, m), v in sorted(sec.items())}
    gaps = {s: (security.get(f"{s}|correct") or 0) - (security.get(f"{s}|wrong") or 0)
            for s in SYSTEMS}

    # paired per-case deltas (same image, same SNR)
    paired = defaultdict(dict)
    for r in rows:
        paired[(r["image_id"], r["snr_db"])][r["system"]] = r
    d_psnr = [v["racs_base"]["psnr"] - v["agentsemstecom"]["psnr"]
              for v in paired.values() if len(v) == 2]
    d_lpips = [v["racs_base"]["lpips"] - v["agentsemstecom"]["lpips"]
               for v in paired.values() if len(v) == 2 and "lpips" in v["racs_base"]]
    b, rr = overall["agentsemstecom"], overall["racs_base"]
    speedup = (b["reference_latency_s"] / rr["reference_latency_s"]
               if rr["reference_latency_s"] else None)
    conclusion = [
        f"Reference acquisition: {fmt(b['reference_latency_s'], 2)} s (diffusion) vs "
        f"{fmt(rr['reference_latency_s'], 3)} s (retrieval), {fmt(speedup, 0)}x faster; "
        f"retrieval needs no GPU memory for Xref (baseline peak "
        f"{fmt(b['reference_peak_vram_gb'], 2)} GB).",
        f"Recovery PSNR: baseline {fmt(b['psnr'], 2)} dB vs RACS {fmt(rr['psnr'], 2)} dB "
        f"(paired mean delta {fmt(mean(d_psnr), 2)} dB over {len(d_psnr)} cases; "
        f"{'RACS better' if (mean(d_psnr) or 0) > 0 else 'RACS worse or equal'}).",
        f"SSIM {fmt(b['ssim'])} vs {fmt(rr['ssim'])}; LPIPS {fmt(b['lpips'])} vs "
        f"{fmt(rr['lpips'])} (lower is better).",
        f"Security gap (correct - wrong token, 10 dB): baseline {fmt(gaps['agentsemstecom'], 2)} dB, "
        f"RACS {fmt(gaps['racs_base'], 2)} dB.",
    ]
    if table.get("limit_per_class"):
        conclusion.append(f"Subset run: {table['limit_per_class']} evaluation image(s) per class; "
                          "treat as indicative, not final.")

    payload = {"schema_version": 1, "phase": 22,
               "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "role": "evaluation", "same_code_path": True,
               "xref_invariant_pass": verify21["checks"]["xref_tx_equals_rx_all_eval_images"],
               "rows": out_rows, "overall": overall, "security_psnr": security,
               "security_gap_db": gaps, "paired_delta_psnr_mean": mean(d_psnr),
               "paired_delta_lpips_mean": mean(d_lpips), "reference_speedup": speedup,
               "conclusion": conclusion,
               "paper_reference_approximate": {
                   "note": "AgentSemSteCom paper anchors, never used in deltas",
                   "security_gap_db": 6.46, "xunet_accuracy": 0.5091}}
    pipeline.write_json(R / "phase22_comparison.json", payload)

    md = ["# Phase 22 — AgentSemSteCom vs RACS-Base", "",
          "Same evaluation cases, tokens, codec, SNRs and downstream code; only Xref differs.", "",
          "| system | class | SNR | n | PSNR | SSIM | LPIPS | ref latency s | total s | ref VRAM GB |",
          "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |"]
    for r in out_rows:
        md.append(f"| {r['system']} | {r['class']} | {r['snr_db']} | {r['n']} | {fmt(r['psnr'], 2)} "
                  f"| {fmt(r['ssim'])} | {fmt(r['lpips'])} | {fmt(r['reference_latency_s'], 3)} "
                  f"| {fmt(r['total_latency_s'], 1)} | {fmt(r['reference_peak_vram_gb'], 2)} |")
    md += ["", "## Security (10 dB, mean PSNR)", "", "| system | token mode | PSNR |",
           "| :-- | :-- | --: |"]
    md += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {fmt(v, 2)} |" for k, v in security.items()]
    md += ["", "## Conclusion (measured)", ""] + [f"- {c}" for c in conclusion]
    md += ["", "Paper anchors (approximate, not measured here): security gap 6.46 dB, "
           "XuNet 0.5091."]
    (R / "phase22_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # grid: secret | baseline rec@5 | baseline rec@20 | RACS rec@5 | RACS rec@20
    d = R / "phase21"
    split = {r["image_id"]: r for r in
             pipeline.read_json(paths.METADATA_DIR / "test_split.json")["records"]}
    grid_rows = []
    for iid in sorted({r["image_id"] for r in rows}):
        row = [str(_ROOT / split[iid]["path"])]
        for s in SYSTEMS:
            for snr in (5, 20):
                row.append(str(d / f"{s}_{iid}_snr{snr}_recovered.png"))
        if all(Path(p).exists() for p in row):
            grid_rows.append(tuple(row))
    if grid_rows:
        pipeline.grid(grid_rows, R / "phase22_grid.png",
                      labels=("secret", "base@5dB", "base@20dB", "racs@5dB", "racs@20dB"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
