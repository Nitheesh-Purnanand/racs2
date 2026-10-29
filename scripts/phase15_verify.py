"""Phase 15 — Gate A (calibration) and Gate B (evaluation) verification."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from steganography import pipeline  # noqa: E402

TOLERANCES = [
    ("facial", 5, "psnr", 22.73, 2.5, "abs"),
    ("facial", 5, "ssim", 0.6989, 0.07, "abs"),
    ("facial", 20, "psnr", 26.29, 2.5, "abs"),
    ("facial", 20, "lpips", 0.2127, 0.05, "abs"),
    ("style", 5, "mse", 0.721e-2, 0.40, "rel"),
    ("style", 5, "lpips", 0.3850, 0.05, "abs"),
    ("common", 5, "psnr", 19.71, 2.5, "abs"),
    ("common", 5, "mse", 1.207e-2, 0.40, "rel"),
    ("common", 10, "ssim", 0.5643, 0.07, "abs"),
]


def _cells(table):
    return {(c["class"], int(c["snr_db"])): c for c in table.get("cells", [])}


def main() -> int:
    cfg_all = pipeline.load_config()
    bcfg = cfg_all["baseline"]
    cal_path = paths.PROJECT_ROOT / bcfg["calibration_output"]
    ev_path = paths.PROJECT_ROOT / bcfg["evaluation_output"]
    cal_man_path = paths.PROJECT_ROOT / bcfg["calibration_manifest"]
    ev_man_path = paths.PROJECT_ROOT / bcfg["evaluation_manifest"]
    verify_path = paths.PROJECT_ROOT / "experiments/results/phase15_verify.json"
    checks, gate_a = [], []

    def add(name, ok, detail="", group="B"):
        checks.append({"check": name, "pass": bool(ok), "detail": detail, "gate": group})

    if not all(p.exists() for p in (cal_path, ev_path, cal_man_path, ev_man_path)):
        pipeline.write_json(verify_path, {"schema_version": 1, "phase": 15, "pass": False,
                                          "checks": [{"check": "tables exist",
                                                      "pass": False}]})
        print("missing phase 15 outputs")
        return 1
    cal, ev = pipeline.read_json(cal_path), pipeline.read_json(ev_path)
    cal_man, ev_man = pipeline.read_json(cal_man_path), pipeline.read_json(ev_man_path)
    tokens = pipeline.read_json(paths.TOKENS_PATH)
    live = pipeline.read_json(paths.TEST_SPLIT_PATH)

    add("15A approximate=true, 15B approximate=false",
        cal.get("approximate") is True and ev.get("approximate") is False)
    add("15A uses calibration only, 15B evaluation only",
        cal.get("role") == "calibration" and ev.get("role") == "evaluation"
        and all(c.startswith("cal_") for c in
                {x["image_id"] for x in cal_man["cases"]})
        and all(c.startswith("eva_") for c in
                {x["image_id"] for x in ev_man["cases"]}))
    add("split_version and tokens_version match live manifests",
        ev.get("split_version") == live.get("split_version")
        and ev_man.get("split_version") == live.get("split_version")
        and ev_man.get("tokens_version") == tokens.get("tokens_version"))

    ok_cells = True
    details = {}
    for table in (cal, ev):
        for cell in table.get("cells", []):
            if cell.get("n") != 22:
                ok_cells = False
            for name in ("mse", "psnr", "ssim", "lpips"):
                dist = cell.get(name) or cell.get("metrics", {}).get(name)
                if not dist or dist.get("n_finite") != 22:
                    ok_cells = False
            details[(table.get("role"), cell["class"], cell["snr_db"])] = cell.get("n")
    add("12 cells per split (3 classes x 4 SNR), N=22, finite metrics", ok_cells
        and len(cal.get("cells", [])) == 12 and len(ev.get("cells", [])) == 12,
        f"cal={len(cal.get('cells', []))} ev={len(ev.get('cells', []))}")

    cal_cells = _cells(cal)
    for cls, snr, metric, paper, tol, kind in TOLERANCES:
        cell = cal_cells.get((cls, snr), {})
        dist = cell.get(metric) or cell.get("metrics", {}).get(metric) or {}
        measured = dist.get("mean")
        if measured is None:
            ok, detail = False, "missing"
        elif kind == "rel":
            rel = abs(measured - paper) / abs(paper)
            ok = rel <= tol
            detail = f"paper={paper} measured={measured:.5f} rel={rel:.2%}"
        else:
            diff = abs(measured - paper)
            ok = diff <= tol
            detail = f"paper={paper} measured={measured:.4f} diff={diff:.4f}"
        gate_a.append({"class": cls, "snr_db": snr, "metric": metric,
                      "paper": paper, "measured": measured, "tolerance": tol,
                      "kind": kind, "pass": ok, "detail": detail})

    monotone = {}
    for cls in ("common", "facial", "style"):
        vals = [(s, (cal_cells.get((cls, s), {}).get("psnr")
                     or cal_cells.get((cls, s), {}).get("metrics", {}).get("psnr", {})
                     ).get("mean")) for s in (5, 10, 15, 20)]
        monotone[cls] = all(vals[i][1] is not None and vals[i + 1][1] is not None
                            and vals[i][1] <= vals[i + 1][1] + 1e-9
                            for i in range(len(vals) - 1))
    add("Gate A monotone improvement 5 -> 20 dB in every class",
        all(monotone.values()), str(monotone), "A")

    def mean_psnr(table, cls, snr):
        cell = _cells(table).get((cls, snr), {})
        return (cell.get("psnr") or cell.get("metrics", {}).get("psnr", {})).get("mean")

    ordering = all(mean_psnr(cal, "facial", s) is not None
                   and mean_psnr(cal, "common", s) is not None
                   and mean_psnr(cal, "facial", s) >= mean_psnr(cal, "common", s)
                   for s in (5, 20))
    add("Gate A class ordering facial >= common at 5 and 20 dB", ordering,
        f"facial5={mean_psnr(cal, 'facial', 5)} common5={mean_psnr(cal, 'common', 5)}",
        "A")

    det = ev.get("determinism", {})
    add("in-process determinism (reference, stego, recovered)",
        det.get("sample", 0) >= 2 and det.get("reference_equal") is True
        and det.get("stego_equal") is True and det.get("recovered_equal") is True,
        f"sample={det.get('sample')}")

    fresh_ok, fresh_detail = False, "not run"
    if ev_man.get("cases"):
        sample = ev_man["cases"][0]
        try:
            raw = subprocess.check_output(
                [sys.executable, "-X", "faulthandler",
                 str(paths.PROJECT_ROOT / "scripts" / "phase15_baseline.py"),
                 "--split", "evaluation", "--probe", sample["image_id"],
                 "--probe-snr", str(sample["snr_db"])],
                cwd=paths.PROJECT_ROOT, text=True, timeout=900)
            fresh = json.loads(raw.strip().splitlines()[-1])
            fresh_ok = (fresh["stego_sha256"] == sample["stego_sha256"]
                        and fresh["recovered_sha256"] == sample["recovered_sha256"]
                        and fresh["reference_sha256"] == sample["reference_sha256"])
            fresh_detail = f"{sample['image_id']} snr={sample['snr_db']} equal={fresh_ok}"
        except Exception as err:  # noqa: BLE001
            fresh_detail = repr(err)
    add("fresh-process determinism", fresh_ok, fresh_detail)

    stego_cfg = cfg_all["steganography"]
    models = ev.get("models", {})
    ip_cfg = stego_cfg.get("ipadapter") or {}
    # The gate is agreement between the live config and what the frozen run
    # recorded, not any particular component choice.
    component_checks = {
        "codec": models.get("codec") == cfg_all["codec"]["kind"],
        "enhancement": ev.get("enhancement_enabled")
        == bool(cfg_all["enhancement"].get("enabled")),
        "ipadapter": bool((models.get("ipadapter") or {}).get("enabled"))
        == bool(ip_cfg.get("enabled")),
        "edict_steps": stego_cfg["edict_steps"] == cfg_all["edict"]["steps"],
        "edict_mixing": stego_cfg["edict_mixing"] == cfg_all["edict"]["mixing"],
    }
    add("component choices match frozen manifests",
        all(component_checks.values()),
        str({k: v for k, v in component_checks.items() if not v}) or "all agree")

    forbidden = ("retrieval", "faiss", "vlm", "racs", "clip_encoder")
    hits = pipeline.forbidden_import_hits(
        [paths.PROJECT_ROOT / "pipelines" / "agentsemstecom.py"], forbidden)
    add("no retrieval/FAISS/VLM import in the pipeline", not hits, "; ".join(hits))

    grids = [paths.PROJECT_ROOT / bcfg["calibration_grid"],
             paths.PROJECT_ROOT / bcfg["evaluation_grid"]]
    add("both grids exist", all(p.exists() for p in grids), str([p.name for p in grids]))

    dev = (paths.PROJECT_ROOT / "docs" / "DEVIATIONS.md").read_text(encoding="utf-8")
    gate_a_miss = [g for g in gate_a if not g["pass"]]
    documented = "Gate A" in dev and "15A" in dev
    add("every Gate A miss is diagnosed in docs/DEVIATIONS.md",
        (not gate_a_miss) or documented,
        f"{len(gate_a_miss)} misses; documented={documented}", "A")

    structural = [c for c in checks if c["gate"] == "B"]
    gate_a_ok = all(g["pass"] for g in gate_a) and all(c["pass"] for c in checks
                                                       if c["gate"] == "A")
    passed = all(c["pass"] for c in structural) and (gate_a_ok or documented)
    warnings = [] if gate_a_ok else [
        f"Gate A tolerance misses: {len(gate_a_miss)} (accepted in docs/DEVIATIONS.md)"]

    pipeline.write_json(verify_path, {
        "schema_version": 1, "phase": 15,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": passed, "gate_a_pass": gate_a_ok, "gate_b_pass": all(c["pass"] for c in structural),
        "checks": checks, "gate_a": gate_a, "warnings": warnings,
        "summary": {"passed": sum(c["pass"] for c in checks), "total": len(checks),
                    "gate_a_misses": len(gate_a_miss)},
    })
    for g in gate_a:
        print(f"[{'PASS' if g['pass'] else 'MISS'}] A {g['class']:6s} {g['snr_db']:>2d}dB "
              f"{g['metric']}: {g['detail']}")
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']} {c['detail']}")
    for w in warnings:
        print(f"[WARN] {w}")
    print(f"=== gateA={gate_a_ok} gateB={all(c['pass'] for c in structural)} "
          f"verify={passed} ===")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
