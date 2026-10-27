"""Phase 11 — offline gate for the local steganography core."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from steganography import pipeline  # noqa: E402
from steganography.mask import token_delta  # noqa: E402
from reference.seed import derive_delta  # noqa: E402

import numpy as np  # noqa: E402


def main() -> int:
    cfg_all = pipeline.load_config()
    cfg = cfg_all["steganography"]
    manifest_path = paths.PROJECT_ROOT / cfg["manifest_path"]
    verify_path = paths.PROJECT_ROOT / cfg["results_dir"] / "phase11_verify.json"
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    if not manifest_path.exists():
        add("phase11_local.json exists", False, str(manifest_path))
        pipeline.write_json(verify_path, {"schema_version": 1, "phase": 11,
                                          "pass": False, "checks": checks})
        print("missing phase11_local.json")
        return 1

    doc = pipeline.read_json(manifest_path)
    conditions = doc.get("conditions", {})
    matched = conditions.get("matched", {})
    mismatched = conditions.get("mismatched", {})
    cells = matched.get("cells", [])

    add("20 images recovered, no failures", matched.get("n") == 20 and len(cells) == 20,
        f"n={matched.get('n')}")
    add("mismatched probe present", mismatched.get("n") == 20, f"n={mismatched.get('n')}")

    m_psnr = (matched.get("distribution", {}).get("psnr") or {}).get("mean")
    x_psnr = (mismatched.get("distribution", {}).get("psnr") or {}).get("mean")
    add("correct-reference mean PSNR finite", m_psnr is not None, f"mean={m_psnr}")
    gap = None if (m_psnr is None or x_psnr is None) else float(m_psnr - x_psnr)
    add("matched reference not worse than mismatched (gap > -0.25 dB)",
        gap is not None and gap > -0.25,
        f"matched={m_psnr} mismatched={x_psnr} gap={gap}")

    dist_ok = all(
        (matched.get("distribution", {}).get(k) or {}).get("n") == 20
        for k in ("mse", "psnr", "ssim", "lpips")
    )
    add("metrics are distributions with N=20", dist_ok)

    stego_psnr = (matched.get("stego_distribution", {}).get("psnr") or {}).get("mean")
    add("stego imperceptibility recorded (PSNR >= 18 dB)",
        stego_psnr is not None and stego_psnr >= 18.0, f"mean stego PSNR={stego_psnr}")

    token = (cells[0]["token"] if cells else "")
    shape = tuple(doc.get("mask_example", {}).get("shape") or (1, 4, 32, 32))
    fraction = float(cfg["mask_fraction"])
    d1 = token_delta(token, shape=shape, fraction=fraction)
    d2 = token_delta(token, shape=shape, fraction=fraction)
    add("mask is deterministic", bool(d1.equal(d2)) and d1.abs().sum().item() > 0)
    ref = np.asarray(derive_delta(token, shape, fraction))
    add("mask matches Phase 8 seed contract",
        bool(np.array_equal(d1.numpy(), ref)), f"shape={shape}")

    det = doc.get("determinism", {})
    add("hide/recover deterministic (stego hash)", det.get("stego_match") is True)
    add("hide/recover deterministic (recovered hash)", det.get("recovered_match") is True)

    words = ("communication", "codec", "retrieval", "faiss", "vlm", "semantic.codec")
    hits = pipeline.forbidden_import_hits(
        [paths.PROJECT_ROOT / "steganography" / n for n in
         ("mask.py", "hide.py", "recover.py", "pipeline.py")], words)
    add("no channel/codec/retrieval/VLM import in stego modules", not hits, "; ".join(hits))

    add("phase11 grid exists",
        (paths.PROJECT_ROOT / cfg["results_dir"] / "phase11_grid.png").exists())

    passed = all(c["pass"] for c in checks)
    pipeline.write_json(verify_path, {
        "schema_version": 1,
        "phase": 11,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": passed,
        "checks": checks,
        "summary": {"passed": sum(c["pass"] for c in checks), "total": len(checks)},
        "metrics": {"matched": matched.get("distribution"),
                    "mismatched": mismatched.get("distribution"),
                    "stego": matched.get("stego_distribution")},
    })
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']} {c['detail']}")
    print(f"=== {sum(c['pass'] for c in checks)}/{len(checks)} checks passed ===")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
