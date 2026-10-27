"""Phase 13 — offline gate for the semantic codec."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from communication.semantic_codec import build_codec, codec_roundtrip  # noqa: E402
from steganography import pipeline  # noqa: E402


def main() -> int:
    cfg_all = pipeline.load_config()
    cfg = cfg_all["codec"]
    ch = cfg_all["channel"]
    manifest_path = paths.PROJECT_ROOT / cfg["manifest_path"]
    verify_path = paths.PROJECT_ROOT / cfg["results_dir"] / "phase13_verify.json"
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    if not manifest_path.exists():
        pipeline.write_json(verify_path, {"schema_version": 1, "phase": 13, "pass": False,
                                          "checks": [{"check": "manifest exists",
                                                      "pass": False}]})
        print("missing phase13_codec.json")
        return 1
    doc = pipeline.read_json(manifest_path)
    snrs = [float(s) for s in ch["snr_db"]]
    isolated = doc.get("isolated", {})
    comparison = doc.get("comparison", {})

    from steganography.edict import EDICT

    edict = EDICT(latent_dtype=torch.float32)
    codec = build_codec(cfg, edict)
    add("checkpoint hash matches manifest", codec["sha256"] == doc.get("codec", {}).get("sha256"),
        f"{codec['sha256'][:16]}")
    add("codec is in eval mode and frozen",
        not codec["vae"].training and not any(p.requires_grad for p in codec["vae"].parameters()))

    finite = all(
        isolated.get(str(int(s)), {}).get("n") == 20
        and all(isolated[str(int(s))]["distribution"][k]["n_finite"] == 20
                for k in ("psnr", "ssim", "lpips"))
        for s in snrs
    )
    add("isolated round-trip finite on 20 images x 4 SNR cells", finite)

    ordered = [isolated.get(str(int(s)), {}).get("distribution", {}).get("psnr", {}).get("mean")
               for s in sorted(snrs)]
    add("codec reconstruction improves as SNR rises",
        all(v is not None for v in ordered)
        and all(ordered[i] <= ordered[i + 1] + 1e-9 for i in range(len(ordered) - 1)),
        str([round(v, 2) if v is not None else None for v in ordered]))

    rec = pipeline.probe_records("evaluation", "facial", 1)[0]
    secret = pipeline.load_pixels(rec["path"], cfg["input_size"])
    a = codec_roundtrip(secret, snr_db=10.0, seed=99, codec=codec, config=cfg)["output"]
    b = codec_roundtrip(secret, snr_db=10.0, seed=99, codec=codec, config=cfg)["output"]
    add("same seed -> identical codec output", torch.equal(a, b))
    add("isolated floor above 20 dB at 20 dB SNR",
        ordered[-1] is not None and ordered[-1] > 20.0,
        f"mean={ordered[-1]}")

    deltas = [comparison.get(str(int(s)), {}).get("delta_psnr") for s in snrs]
    add("raw and codec paths both measured and differ",
        all(d is not None for d in deltas) and any(abs(d) > 1e-6 for d in deltas),
        str([round(d, 3) if d is not None else None for d in deltas]))

    forbidden = ("reference", "retrieval", "faiss", "vlm", "edict")
    hits = pipeline.forbidden_import_hits(
        [paths.PROJECT_ROOT / "communication" / "semantic_codec.py"], forbidden)
    add("no reference/retrieval/VLM import in codec module", not hits, "; ".join(hits))

    add("phase13 json, curve and grid exist",
        manifest_path.exists()
        and (paths.PROJECT_ROOT / cfg["curve_path"]).exists()
        and (paths.PROJECT_ROOT / cfg["grid_path"]).exists())

    edict.unload()
    passed = all(c["pass"] for c in checks)
    pipeline.write_json(verify_path, {
        "schema_version": 1, "phase": 13,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": passed, "checks": checks,
        "summary": {"passed": sum(c["pass"] for c in checks), "total": len(checks)},
        "isolated_psnr_by_snr": dict(zip([str(int(s)) for s in snrs], ordered)),
        "delta_psnr_codec_minus_raw": dict(zip([str(int(s)) for s in snrs], deltas)),
    })
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']} {c['detail']}")
    print(f"=== {sum(c['pass'] for c in checks)}/{len(checks)} checks passed ===")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
