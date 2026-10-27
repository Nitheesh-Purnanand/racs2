"""Phase 12 — offline gate for the AWGN channel and recovery curve."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from communication.awgn import awgn_channel, noise_variance_for_snr_db  # noqa: E402
from steganography import pipeline  # noqa: E402


def main() -> int:
    cfg_all = pipeline.load_config()
    cfg = cfg_all["channel"]
    manifest_path = paths.PROJECT_ROOT / cfg["manifest_path"]
    verify_path = paths.PROJECT_ROOT / cfg["results_dir"] / "phase12_verify.json"
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    if not manifest_path.exists():
        pipeline.write_json(verify_path, {"schema_version": 1, "phase": 12, "pass": False,
                                          "checks": [{"check": "manifest exists",
                                                      "pass": False}]})
        print("missing phase12_awgn.json")
        return 1
    doc = pipeline.read_json(manifest_path)
    results = doc.get("results", {})
    snrs = [float(s) for s in cfg["snr_db"]]

    # Analytic variance vs measured noise variance.
    probe = torch.rand(1, 3, 256, 256) * 0.5 + 0.25
    variance_checks = []
    for idx, snr in enumerate(snrs):
        _, meta = awgn_channel(probe, snr_db=snr, seed=int(cfg["seed"]) + idx, h=1 + 0j)
        target = float(noise_variance_for_snr_db(probe, snr))
        rel = abs(meta["variance_measured"] - target) / max(target, 1e-12)
        variance_checks.append({"snr_db": snr, "target": target,
                                "measured": meta["variance_measured"], "rel_error": rel})
    add("analytic variance within 1 percent on every cell",
        all(v["rel_error"] <= 0.01 for v in variance_checks),
        f"max rel={max(v['rel_error'] for v in variance_checks):.4f}")

    # Seed replay and seed separation.
    x = torch.rand(1, 3, 16, 16)
    a, _ = awgn_channel(x, snr_db=10.0, seed=7)
    b, _ = awgn_channel(x, snr_db=10.0, seed=7)
    c, _ = awgn_channel(x, snr_db=10.0, seed=8)
    add("same seed -> identical received tensor", torch.equal(a, b))
    add("different seed -> different noise", not torch.equal(a, c))

    complete = all(results.get(str(int(s)), {}).get("n", 0) >= 20 for s in snrs)
    add("every SNR cell has at least 20 images", complete,
        str({s: results.get(str(int(s)), {}).get("n") for s in snrs}))

    means = [results.get(str(int(s)), {}).get("distribution", {}).get("psnr", {}).get("mean")
             for s in snrs]
    ordered = [results.get(str(int(s)), {}).get("distribution", {}).get("psnr", {}).get("mean")
               for s in sorted(snrs)]
    add("mean PSNR non-increasing as SNR falls",
        all(v is not None for v in ordered)
        and all(ordered[i] <= ordered[i + 1] + 1e-9 for i in range(len(ordered) - 1)),
        str([round(v, 2) if v is not None else None for v in means]))

    forbidden = ("semantic_codec", "retrieval", "faiss", "vlm", "codeformer")
    hits = pipeline.forbidden_import_hits([paths.PROJECT_ROOT / "communication" / "awgn.py"],
                                          forbidden)
    add("no codec/retrieval/VLM import in awgn.py", not hits, "; ".join(hits))

    add("phase12 manifest and curve exist",
        manifest_path.exists() and (paths.PROJECT_ROOT / cfg["curve_path"]).exists())

    passed = all(c["pass"] for c in checks)
    pipeline.write_json(verify_path, {
        "schema_version": 1, "phase": 12,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": passed, "checks": checks,
        "summary": {"passed": sum(c["pass"] for c in checks), "total": len(checks)},
        "variance": variance_checks,
        "psnr_mean_by_snr": dict(zip([str(int(s)) for s in snrs], means)),
        "monotone_psnr": doc.get("monotone_psnr"),
    })
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']} {c['detail']}")
    print(f"=== {sum(c['pass'] for c in checks)}/{len(checks)} checks passed ===")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
