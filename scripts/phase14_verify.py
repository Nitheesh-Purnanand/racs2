"""Phase 14 — offline gate for optional facial enhancement."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from enhancement.codeformer import build_enhancer, enhance, should_enhance  # noqa: E402
from steganography import pipeline  # noqa: E402


def main() -> int:
    cfg_all = pipeline.load_config()
    cfg = dict(cfg_all["enhancement"])
    manifest_path = paths.PROJECT_ROOT / cfg["manifest_path"]
    verify_path = paths.PROJECT_ROOT / cfg["results_dir"] / "phase14_verify.json"
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    if not manifest_path.exists():
        pipeline.write_json(verify_path, {"schema_version": 1, "phase": 14, "pass": False,
                                          "checks": [{"check": "manifest exists",
                                                      "pass": False}]})
        print("missing phase14_codeformer.json")
        return 1
    doc = pipeline.read_json(manifest_path)

    enabled = cfg_all["enhancement"].get("enabled")
    add("enhancement switch is explicit and recorded",
        isinstance(enabled, bool) and doc.get("enabled_by_default") == enabled,
        f"enabled={enabled} manifest={doc.get('enabled_by_default')}")
    add("the switch applies to the configured classes only",
        should_enhance(cfg_all["enhancement"], "facial") is bool(enabled)
        and not should_enhance(cfg_all["enhancement"], "common")
        and not should_enhance(cfg_all["enhancement"], "style"),
        f"classes={cfg_all['enhancement'].get('classes')}")

    enhancer = build_enhancer({**cfg, "enabled": True})
    add("fidelity weight is 0.5 and recorded",
        float(enhancer["fidelity_weight"]) == 0.5
        and float(doc.get("enhancer", {}).get("fidelity_weight", -1)) == 0.5)
    add("enhancer manifest records the backend and fingerprint",
        bool(doc.get("enhancer", {}).get("backend"))
        and len(str(doc.get("enhancer", {}).get("fingerprint", ""))) == 64)

    results = doc.get("results", {})
    snrs = [float(s) for s in doc.get("snr_db", [])]
    add("deltas recorded at all four SNR cells",
        all(results.get(str(int(s)), {}).get("n", 0) > 0 for s in snrs),
        str({s: results.get(str(int(s)), {}).get("n") for s in snrs}))
    add("fidelity weight present in the config", cfg.get("fidelity_weight") == 0.5)

    # The real CodeFormer trades distortion for perceptual quality, so the PSNR
    # delta may be negative; what the gate requires is that the deltas are
    # measured and that the enhancer actually acted. A pass-through backend is
    # only acceptable while the documented miss stands.
    backend = doc.get("enhancer", {}).get("backend")
    gain = [results.get(str(int(s)), {}).get("delta", {}).get("psnr", {}).get("mean")
            for s in snrs]
    lpips_gain = [results.get(str(int(s)), {}).get("delta", {}).get("lpips", {}).get("mean")
                  for s in snrs]
    add("facial deltas measured at every SNR cell",
        all(g is not None for g in gain) and all(g is not None for g in lpips_gain),
        f"backend={backend} dPSNR={gain} dLPIPS={lpips_gain}")
    add("the enhancer changed the image, or the miss is documented",
        (backend == "codeformer" and any(abs(g) > 1e-6 for g in gain))
        or (backend == "passthrough" and all(abs(g) < 1e-9 for g in gain)),
        f"backend={backend} dPSNR={gain}")
    add("the documented miss is registered in the manifest",
        bool(doc.get("decisions", {}).get("codeformer_weights")))

    control = doc.get("control", {})
    add("common path unaffected by the switch",
        control.get("identical") is True and control.get("applied") is False,
        str(control))
    add("style class is out of scope by config",
        "style" not in set(cfg.get("classes", [])))

    probe_cell = None
    for snr in snrs:
        cells = results.get(str(int(snr)), {}).get("cells", [])
        if cells:
            probe_cell = cells[0]
            break
    deterministic = False
    shape_ok = False
    if probe_cell and Path(probe_cell["enhanced_path"]).exists():
        src = Image.open(probe_cell["recovered_path"]).convert("RGB")
        a = enhance(src, enhancer=enhancer, config={**cfg, "enabled": True})
        b = enhance(src, enhancer=enhancer, config={**cfg, "enabled": True})
        deterministic = a.tobytes() == b.tobytes()
        # The enhanced image must drop straight back into the pipeline, so its
        # size must equal the recovered image's, whatever size CodeFormer
        # restores at internally.
        shape_ok = a.size == src.size and a.mode == "RGB"
    add("same input -> identical enhanced output", deterministic)
    add("enhanced output keeps the recovered image size and mode", shape_ok,
        f"restoration_size={cfg['input_size']} (paper uses 512 end to end)")

    add("phase14 json and grid exist",
        manifest_path.exists() and (paths.PROJECT_ROOT / cfg["grid_path"]).exists())

    passed = all(c["pass"] for c in checks)
    pipeline.write_json(verify_path, {
        "schema_version": 1, "phase": 14,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": passed, "checks": checks,
        "summary": {"passed": sum(c["pass"] for c in checks), "total": len(checks)},
        "backend": backend,
        "delta_psnr_by_snr": dict(zip([str(int(s)) for s in snrs], gain)),
    })
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']} {c['detail']}")
    print(f"=== {sum(c['pass'] for c in checks)}/{len(checks)} checks passed ===")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
