"""
Phase 6 gate: self-test for metrics, splits, role enforcement and reporting.

Runs offline and finishes quickly. Exits non-zero if any check fails and writes
`experiments/results/phase6_selftest.json`.

Usage:
    python scripts/phase6_eval_selftest.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import numpy as np
import torch
import yaml

import common.paths  # noqa: F401
from common.paths import PROJECT_ROOT, RESULTS_DIR
from evaluation.metrics import all_metrics, distribution
from evaluation.report import build_report
from evaluation.splits import (
    Split,
    SplitError,
    RoleError,
    compute_split_version,
    compute_tokens_version,
    load_tokens,
    resolve_path,
)

CHECKS = []


def check(name, condition, detail=""):
    ok = bool(condition)
    CHECKS.append({"check": name, "pass": ok, "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
          flush=True)
    return ok


def pil_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def structural_change(tensor, block=32):
    """Rotate 90 degrees and shuffle blocks deterministically."""
    image = tensor.permute(1, 2, 0).numpy()
    rotated = np.rot90(image, k=1).copy()
    height, width, _ = rotated.shape
    blocks = []
    for top in range(0, height - block + 1, block):
        for left in range(0, width - block + 1, block):
            blocks.append(rotated[top:top + block, left:left + block])
    rng = np.random.default_rng(0)
    order = rng.permutation(len(blocks))
    out = rotated.copy()
    index = 0
    for top in range(0, height - block + 1, block):
        for left in range(0, width - block + 1, block):
            out[top:top + block, left:left + block] = blocks[order[index]]
            index += 1
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()


def main():
    config = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    eval_cfg = config["evaluation"]

    print("=== metrics ===", flush=True)
    eval_split = Split.load(role="evaluation")
    sample_record = eval_split.records("common")[0]
    from PIL import Image
    secret = Image.open(resolve_path(sample_record["path"])).convert("RGB")
    base = pil_to_tensor(secret)

    identical = all_metrics(base, base, with_lpips=True)
    check("identical mse == 0", identical["mse"] == 0.0, identical["mse"])
    check("identical psnr is inf", math.isinf(identical["psnr"]), identical["psnr"])
    check("identical ssim ~ 1", abs(identical["ssim"] - 1.0) < 1e-4, identical["ssim"])
    check("identical lpips ~ 0", identical["lpips"] <= 1e-6, identical["lpips"])

    rng = np.random.default_rng(7)
    noisy_np = np.clip(base.permute(1, 2, 0).numpy() + rng.normal(0, 0.02, base.shape[1:] + (3,)),
                       0.0, 1.0)
    noisy = torch.from_numpy(noisy_np).permute(2, 0, 1).contiguous()
    noisy_metrics = all_metrics(base, noisy, with_lpips=True)
    check("noisy psnr finite in (20,45)", 20 < noisy_metrics["psnr"] < 45,
          noisy_metrics["psnr"])
    check("noisy lpips in (0,0.5)", 0 < noisy_metrics["lpips"] < 0.5,
          noisy_metrics["lpips"])

    changed = structural_change(base)
    changed_metrics = all_metrics(base, changed, with_lpips=True)
    check("structural ssim < 0.6", changed_metrics["ssim"] < 0.6, changed_metrics["ssim"])
    check("structural lpips > 0.15", changed_metrics["lpips"] > 0.15,
          changed_metrics["lpips"])

    try:
        from skimage.metrics import structural_similarity as sk_ssim
        sk_value = sk_ssim(base.permute(1, 2, 0).numpy(),
                           noisy.permute(1, 2, 0).numpy(),
                           channel_axis=2, data_range=1.0,
                           gaussian_weights=True, sigma=1.5,
                           use_sample_covariance=False)
        delta = abs(sk_value - noisy_metrics["ssim"])
        check("project ssim matches skimage (<= 0.02)", delta <= 0.02,
              f"delta={delta:.5f}")
    except Exception as error:  # noqa: BLE001
        check("project ssim matches skimage (<= 0.02)", False, repr(error))

    dist = distribution([1, 2, 3, 4, 5])
    check("distribution mean/median",
          dist["mean"] == 3.0 and dist["median"] == 3.0 and dist["n"] == 5, dist["n"])
    dist_inf = distribution([1.0, float("inf"), 3.0])
    check("distribution handles inf",
          dist_inf["n"] == 3 and dist_inf["n_inf"] == 1 and dist_inf["mean"] == 2.0,
          f"n_inf={dist_inf['n_inf']} mean={dist_inf['mean']}")

    print("=== splits ===", flush=True)
    cal_split = Split.load(role="calibration", allow_calibration=True)
    check("both roles load", len(eval_split) > 0 and len(cal_split) > 0,
          f"eval={len(eval_split)} cal={len(cal_split)}")
    expected = eval_cfg["per_class"] * 3
    check("evaluation count matches config", len(eval_split) == expected,
          f"{len(eval_split)} vs {expected}")
    check("calibration count matches config", len(cal_split) == expected,
          f"{len(cal_split)} vs {expected}")
    check("split_version recomputes",
          eval_split.version == compute_split_version(eval_split.manifest["records"]),
          eval_split.version)
    check("calibration x evaluation disjoint", eval_split.assert_disjoint(cal_split))
    check("evaluation disjoint from retrieval corpus",
          eval_split.assert_disjoint_from_index())
    check("calibration disjoint from retrieval corpus",
          cal_split.assert_disjoint_from_index())

    per_class_ok = True
    details = []
    for role, split in (("evaluation", eval_split), ("calibration", cal_split)):
        for cls in eval_cfg["classes"]:
            count = len(split.records(cls))
            details.append(f"{role}/{cls}={count}")
            per_class_ok = per_class_ok and count >= 20
    check("each role/class has >= 20", per_class_ok, "; ".join(details))

    print("=== role enforcement ===", flush=True)
    check("evaluation split has no calibration records",
          all(r["role"] == "evaluation" for r in eval_split.records()))

    bad = Split(cal_split.records(), cal_split.manifest, role="evaluation", verify=False)
    raised = False
    try:
        bad.get(cal_split.ids()[0])
    except RoleError:
        raised = True
    check("calibration record rejected under evaluation role", raised)

    raised = False
    try:
        Split.load(role="calibration")
    except RoleError:
        raised = True
    check("calibration load requires allow_calibration", raised)

    print("=== tokens ===", flush=True)
    all_tokens, tokens_version = load_tokens()
    raw_tokens = json.loads((PROJECT_ROOT / eval_cfg["tokens_path"])
                            .read_text(encoding="utf-8"))
    recomputed = compute_tokens_version(raw_tokens["records"])
    check("tokens_version recomputes", recomputed == raw_tokens["tokens_version"],
          recomputed)

    combined = cal_split.ids() + eval_split.ids()
    check("one token per image", set(combined) <= set(all_tokens) and
          len(combined) == len(all_tokens),
          f"{len(combined)} ids, {len(all_tokens)} tokens")
    tokens_unique = len({v["token"] for v in all_tokens.values()}) == len(all_tokens)
    wrong_unique = len({v["wrong_token"] for v in all_tokens.values()}) == len(all_tokens)
    differ = all(v["token"] != v["wrong_token"] for v in all_tokens.values())
    check("tokens unique and different from wrong token",
          tokens_unique and wrong_unique and differ,
          f"unique={tokens_unique} wrong_unique={wrong_unique} differ={differ}")

    print("=== report ===", flush=True)
    synthetic = []
    for record in eval_split.records()[:3]:
        synthetic.append({
            "image_id": record["image_id"], "class": record["class"], "snr_db": 10,
            "mse": 0.01, "psnr": 20.0, "ssim": 0.8, "lpips": 0.2,
        })
    report = build_report("selftest", "evaluation", eval_split, synthetic,
                          approximate=False)
    check("evaluation report marked approximate=False",
          report["approximate"] is False)
    check("report cells well formed",
          len(report["cells"]) >= 1 and report["cells"][0]["n"] == len(synthetic),
          f"{len(report['cells'])} cells")

    cal_results = []
    for record in cal_split.records()[:2]:
        cal_results.append({
            "image_id": record["image_id"], "class": record["class"], "snr_db": 5,
            "mse": 0.02, "psnr": 17.0, "ssim": 0.7, "lpips": 0.3,
        })
    cal_report = build_report("selftest", "calibration", cal_split, cal_results,
                              approximate=True)
    check("calibration report marked approximate=True",
          cal_report["approximate"] is True)

    mixed_raised = False
    try:
        build_report("selftest", "evaluation", eval_split,
                     [{"image_id": cal_split.ids()[0], "class": "common",
                       "snr_db": 5, "mse": 0.1}], approximate=False)
    except ValueError:
        mixed_raised = True
    check("mixed-role report raises", mixed_raised)

    passed = sum(1 for item in CHECKS if item["pass"])
    failed = len(CHECKS) - passed
    summary = {
        "passed": passed,
        "failed": failed,
        "checks": CHECKS,
        "split_version": eval_split.version,
        "tokens_version": tokens_version,
        "counts": {
            "calibration": {cls: len(cal_split.records(cls)) for cls in eval_cfg["classes"]},
            "evaluation": {cls: len(eval_split.records(cls)) for cls in eval_cfg["classes"]},
        },
    }
    out = RESULTS_DIR / "phase6_selftest.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("")
    print(f"=== {passed} passed, {failed} failed ===")
    print(f"wrote {out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
