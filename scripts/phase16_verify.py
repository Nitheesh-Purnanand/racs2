"""Phase 16 — offline gate for security and steganalysis."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from steganography import pipeline  # noqa: E402


def main() -> int:
    cfg_all = pipeline.load_config()
    cfg = cfg_all["security"]
    bcfg = cfg_all["baseline"]
    sec_path = paths.PROJECT_ROOT / cfg["output_security"]
    steg_path = paths.PROJECT_ROOT / cfg["output_steganalysis"]
    man_path = paths.PROJECT_ROOT / cfg["manifest_path"]
    verify_path = paths.PROJECT_ROOT / "experiments/results/phase16_verify.json"
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    if not sec_path.exists() or not steg_path.exists():
        pipeline.write_json(verify_path, {"schema_version": 1, "phase": 16,
                                          "pass": False,
                                          "checks": [{"check": "outputs exist",
                                                      "pass": False}]})
        print("missing phase16 outputs")
        return 1
    sec = pipeline.read_json(sec_path)
    steg = pipeline.read_json(steg_path)
    table_path = paths.PROJECT_ROOT / bcfg["evaluation_output"]

    add("baseline table hash matches",
        sec.get("baseline_table_sha256") == hashlib.sha256(table_path.read_bytes()).hexdigest()
        and steg.get("baseline_table_sha256") == sec["baseline_table_sha256"])

    add("security and steganalysis are evaluation-only and not approximate",
        sec.get("split_role") == "evaluation" and sec.get("approximate") is False
        and steg.get("split_role") == "evaluation" and steg.get("approximate") is False)

    cells = sec.get("cells", [])
    modes_ok = all(
        c.get("n") == 22 and all(c.get(m, {}).get("psnr", {}).get("n") == 22
                                 for m in ("correct", "wrong", "no_token"))
        for c in cells
    )
    add("all three token conditions present with N=22", modes_ok and len(cells) == 12,
        f"cells={len(cells)}")

    def weighted(mode):
        total = sum(c["n"] for c in cells)
        return sum(c[mode]["psnr"]["mean"] * c["n"] for c in cells) / max(1, total)

    correct, wrong, no_token = weighted("correct"), weighted("wrong"), weighted("no_token")
    add("correct-token PSNR above wrong and no-token",
        correct > wrong and correct > no_token,
        f"correct={correct:.2f} wrong={wrong:.2f} no_token={no_token:.2f}")
    add("correct-minus-wrong gap >= 4 dB in aggregate",
        (correct - wrong) >= 4.0, f"gap={correct - wrong:.2f} dB")

    man = pipeline.read_json(man_path) if man_path.exists() else {}
    records = man.get("cases", [])
    conditions_ok = (
        len(records) == 264
        and all(r.get("role") == "evaluation" for r in records)
        and all(set(r.get("conditions", {})) == {"correct", "wrong", "no_token"}
                for r in records)
        and all(all({"token_used", "reference_seed", "mask_seed", "mask_sha256"} <=
                    set(cond) for cond in r["conditions"].values())
                for r in records)
    )
    add("phase16 manifest records every receiver condition",
        man.get("split_role") == "evaluation" and man.get("approximate") is False
        and bool(man.get("split_version")) and conditions_ok,
        f"records={len(records)}")

    fallback = man.get("no_token", {})
    add("no-token fallback seed is documented and distinct",
        fallback.get("fallback_token") == "00000000"
        and isinstance(fallback.get("fallback_reference_seed"), int)
        and isinstance(fallback.get("fallback_mask_seed"), int)
        and "wrong-token" in str(fallback.get("not_the_same_as", "")),
        f"token={fallback.get('fallback_token')} "
        f"reference_seed={fallback.get('fallback_reference_seed')} "
        f"mask_seed={fallback.get('fallback_mask_seed')}")

    threshold = float(man.get("gap_threshold_db", -1))
    exceptions = man.get("gap_exceptions", [])
    documented = {(e.get("class"), e.get("snr_db")) for e in exceptions}
    expected = {(c["class"], c["snr_db"]) for c in cells
                if c["gap_correct_minus_wrong_db"] < 4.0
                or c["gap_correct_minus_no_token_db"] < 4.0}
    add("per-cell gap exceptions documented",
        threshold == 4.0 and documented == expected
        and all("reason" in e and e["reason"] for e in exceptions),
        f"exceptions={len(exceptions)}")

    split_ids = steg.get("dataset", {}).get("split_ids", {})
    train, val, test = (set(split_ids.get(s, [])) for s in ("train", "val", "test"))
    add("XuNet split is image-disjoint",
        not (train & val or train & test or val & test),
        f"train={len(train)} val={len(val)} test={len(test)}")
    add("steganalysis class balance recorded",
        steg.get("dataset", {}).get("class_balance", {}).get("cover", 0) > 0)

    ckpt = Path(steg.get("checkpoint_path", ""))
    ckpt = ckpt if ckpt.is_absolute() else paths.PROJECT_ROOT / ckpt
    sha_ok = ckpt.exists() and hashlib.sha256(ckpt.read_bytes()).hexdigest() == \
        steg.get("checkpoint_sha256")
    add("checkpoint exists and sha256 matches", sha_ok, str(ckpt))
    test_res = steg.get("results", {}).get("test", {})
    add("test accuracy reported with N",
        test_res.get("n", 0) > 0 and 0.0 <= test_res.get("accuracy", -1) <= 1.0,
        f"accuracy={test_res.get('accuracy')} n={test_res.get('n')}")
    ds = steg.get("dataset", {})
    patches = int(ds.get("n_patches_per_image", 0))
    expected_test_samples = 2 * len(test) * patches
    add("steganalysis counts and chance rate reported",
        ds.get("n_images") == 66 and patches >= 1
        and ds.get("n_samples", {}).get("test", 0) == expected_test_samples
        and ds.get("n_samples", {}).get("train", 0) == 2 * len(train) * patches
        and ds.get("n_samples", {}).get("val", 0) == 2 * len(val) * patches
        and steg.get("chance_rate") == 0.5
        and steg.get("results", {}).get("test", {}).get("chance_rate") == 0.5,
        f"test_samples={ds.get('n_samples', {}).get('test')} "
        f"patches/image={patches}")

    add("paper anchors present and labelled approximate",
        sec.get("paper_anchor", {}).get("source", "").lower().find("approximate") >= 0
        and steg.get("paper_anchor", {}).get("label") == "approximate")

    ids = {c["image_id"] for c in pipeline.read_json(
        paths.PROJECT_ROOT / bcfg["evaluation_manifest"])["cases"]}
    add("no calibration image appears", all(i.startswith("eva_") for i in ids) and len(ids) == 66,
        f"{len(ids)} evaluation images")

    add("phase16 security grid exists",
        (paths.PROJECT_ROOT / "experiments/results/phase16_security_grid.png").exists())

    passed = all(c["pass"] for c in checks)
    pipeline.write_json(verify_path, {
        "schema_version": 1, "phase": 16,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": passed, "checks": checks,
        "summary": {"passed": sum(c["pass"] for c in checks), "total": len(checks)},
        "aggregate_psnr": {"correct": correct, "wrong": wrong,
                           "no_token": no_token},
        "xuNet_test_accuracy": test_res.get("accuracy"),
    })
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']} {c['detail']}")
    print(f"=== {sum(c['pass'] for c in checks)}/{len(checks)} checks passed ===")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
