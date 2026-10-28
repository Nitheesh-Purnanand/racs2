"""Phase 9 — verify the zero-conv fine-tune artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths
import common.hf_lowmem

common.hf_lowmem.install()

from evaluation.splits import sha256_file


def main() -> None:
    training_path = paths.RESULTS_DIR / "phase9_training.json"
    checks = []
    failures = []

    def add(name, passed, detail=""):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        if not passed:
            failures.append(name)

    if not training_path.exists():
        print("missing phase9_training.json")
        sys.exit(1)
    with open(training_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    add("schema_version", doc.get("schema_version") == 1)
    add("train_role_calibration", doc.get("split_role") == "calibration",
        doc.get("split_role", ""))
    add("trainable_params", doc.get("trainable_parameters", 0) > 0,
        str(doc.get("trainable_parameters")))
    add("frozen_params", doc.get("frozen_parameters", 0) > 0,
        str(doc.get("frozen_parameters")))
    add("loss_decreased", doc.get("loss_last", 1) < doc.get("loss_first", 0),
        f"{doc.get('loss_first')} -> {doc.get('loss_last')}")
    add("loss_finite", doc.get("loss_min", -1) >= 0 and doc.get("loss_last", -1) >= 0,
        f"min={doc.get('loss_min')}")
    ckpt = paths.PROJECT_ROOT / doc.get("checkpoint_path", "")
    add("checkpoint_exists", ckpt.exists(), str(ckpt))
    if ckpt.exists():
        add("checkpoint_sha", sha256_file(ckpt) == doc.get("checkpoint_sha256"))

    probe = doc.get("probe", {})
    improved = bool(probe.get("improved"))
    decision = doc.get("decision")
    add("probe_present", probe.get("stage") == "reference_only" and probe.get("n", 0) > 0,
        f"{probe.get('n')} images")
    if not improved:
        dev = paths.DEVIATIONS_PATH
        text = dev.read_text(encoding="utf-8") if dev.exists() else ""
        add("deviation_recorded", decision == "pretrained_deviation" and "Phase 9" in text,
            "pretrained branch accepted; deviation row required")
    else:
        add("probe_improved", True,
            f"{probe.get('zero_shot_mse')} -> {probe.get('fine_tuned_mse')}")

    result = {
        "pass": not failures,
        "schema_version": 1,
        "split_version": doc.get("split_version"),
        "config_hash": doc.get("config_hash"),
        "checks": checks,
        "failures": failures,
    }
    out = paths.RESULTS_DIR / "phase9_verify.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nVERIFICATION RESULT: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"written: {out}")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
