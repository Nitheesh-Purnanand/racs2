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
from common.paths import IMAGE_INDEX_PATH, PROJECT_ROOT, RESULTS_DIR
from evaluation.metrics import all_metrics, distribution
from evaluation.report import build_report
from evaluation.splits import (
    CLASSES,
    Split,
    SplitError,
    RoleError,
    compute_split_version,
    compute_tokens_version,
    load_tokens,
    resolve_path,
)

CHECKS = []


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f"duplicate YAML key: {key!r}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


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
    empty = distribution([])
    check("distribution handles empty input",
          empty["n"] == 0 and empty["mean"] is None, empty["n"])
    with_nan = distribution([1.0, float("nan"), 3.0])
    check("distribution counts NaN as missing",
          with_nan["n_missing"] == 1 and with_nan["mean"] == 2.0,
          with_nan["n_missing"])

    default_metrics = all_metrics(base, noisy)
    check("all_metrics default keys unchanged",
          set(default_metrics) == {"mse", "psnr", "ssim"}, sorted(default_metrics))

    out_of_range_raised = False
    try:
        all_metrics(base, noisy * 2.0 + 1.0, with_lpips=True)
    except ValueError:
        out_of_range_raised = True
    check("lpips rejects out-of-range input", out_of_range_raised)

    gray = base.mean(dim=0, keepdim=True).repeat(1, 1, 1)
    gray_ok = True
    try:
        gray_metrics = all_metrics(gray, gray, with_lpips=True)
        gray_ok = gray_metrics["lpips"] <= 1e-6
    except Exception:  # noqa: BLE001
        gray_ok = False
    check("lpips accepts grayscale input", gray_ok)

    batch = torch.stack([base, noisy], dim=0)
    batch_metric = all_metrics(base, noisy, with_lpips=True)["lpips"]
    from evaluation.metrics import lpips_metric
    batched = lpips_metric(batch, batch, batch_size=2)
    repeated = lpips_metric(base, noisy, batch_size=1)
    check("lpips deterministic across batch sizes",
          abs(batched) <= 1e-6 and abs(repeated - batch_metric) <= 1e-6,
          f"identical={batched:.2e} delta={abs(repeated - batch_metric):.2e}")

    print("=== splits ===", flush=True)
    cal_split = Split.load(role="calibration", allow_calibration=True)
    m_manifest = json.loads((PROJECT_ROOT / eval_cfg["splits_path"])
                            .read_text(encoding="utf-8"))
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

    from PIL import Image as PILImage
    schema_ok, schema_detail = True, []
    for record in eval_split.records() + cal_split.records():
        for field in ("source", "license", "source_ref", "width", "height", "mode"):
            if field not in record:
                schema_ok = False
                schema_detail.append(f"{record['image_id']} missing {field}")
        if record.get("width") != 512 or record.get("height") != 512:
            schema_ok = False
            schema_detail.append(f"{record['image_id']} wrong size")
        if record.get("mode") != "RGB":
            schema_ok = False
            schema_detail.append(f"{record['image_id']} wrong mode")
        with PILImage.open(resolve_path(record["path"])) as image:
            if image.size != (512, 512) or image.mode != "RGB":
                schema_ok = False
                schema_detail.append(f"{record['image_id']} pixel check failed")
    check("record schema and pixels are 512x512 RGB", schema_ok,
          "; ".join(schema_detail[:3]))

    provenance_ok, provenance_detail = True, []
    paths_ok = True
    blank_images = []
    for role, split in (("calibration", cal_split), ("evaluation", eval_split)):
        for record in split.records():
            for field in ("source", "license", "source_ref", "original_id"):
                if not str(record.get(field, "")).strip():
                    provenance_ok = False
                    provenance_detail.append(f"{record['image_id']} missing {field}")
            source = str(record.get("source", "")).lower()
            if role == "calibration" and "unistega" not in source:
                provenance_ok = False
                provenance_detail.append(f"{record['image_id']} not from UniStega")
            if role == "evaluation" and "unistega" in source:
                provenance_ok = False
                provenance_detail.append(f"{record['image_id']} leaks UniStega")
            raw_path = record["path"]
            if Path(raw_path).is_absolute() or "\\" in raw_path:
                paths_ok = False
            resolved = resolve_path(raw_path)
            if resolved.stat().st_size < 2048:
                provenance_ok = False
                provenance_detail.append(f"{record['image_id']} tiny file")
            with PILImage.open(resolved) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32)
            if float(array.std()) < (1.0 / 255.0):
                blank_images.append(record["image_id"])
    check("provenance complete and role-consistent", provenance_ok,
          "; ".join(provenance_detail[:3]))
    check("manifest paths are relative and forward-slash", paths_ok)
    check("no near-blank images", not blank_images, blank_images[:3])

    import re
    manifest_meta_ok = (
        m_manifest.get("corpus_version") == json.loads(
            IMAGE_INDEX_PATH.read_text(encoding="utf-8"))["corpus_version"]
        and m_manifest.get("counts") == {
            role: {cls: len(cal_split.records(cls)) if role == "calibration"
                   else len(eval_split.records(cls)) for cls in CLASSES}
            for role in ("calibration", "evaluation")
        }
        and str(m_manifest.get("canonical", "")).startswith("shortest_side_512")
        and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
                     str(m_manifest.get("created_utc", ""))) is not None
    )
    check("manifest metadata consistent", manifest_meta_ok)

    ids = [r["image_id"] for r in eval_split.records() + cal_split.records()]
    shas = [r["sha256"] for r in eval_split.records() + cal_split.records()]
    check("no duplicate ids or hashes", len(ids) == len(set(ids)) and len(shas) == len(set(shas)),
          f"ids={len(ids)} unique={len(set(ids))}")

    import shutil
    from evaluation.splits import write_manifests
    tmp_dir = PROJECT_ROOT / "temp" / "phase6_selftest_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = m_manifest

    bad_version = dict(raw_manifest, split_version="0" * 64)
    bad_version_path = tmp_dir / "bad_version.json"
    bad_version_path.write_text(json.dumps(bad_version), encoding="utf-8")
    raised = False
    try:
        Split.load(role="evaluation", manifest_path=bad_version_path)
    except SplitError:
        raised = True
    check("corrupted split_version rejected", raised)

    duplicate = json.loads(json.dumps(raw_manifest))
    duplicate["records"].append(dict(duplicate["records"][0]))
    duplicate["split_version"] = compute_split_version(duplicate["records"])
    duplicate_path = tmp_dir / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    raised = False
    try:
        Split.load(role="evaluation", manifest_path=duplicate_path)
    except SplitError:
        raised = True
    check("duplicate manifest record rejected", raised)

    sha_bad = json.loads(json.dumps(raw_manifest))
    sha_bad["records"][0]["sha256"] = "f" * 64
    sha_bad["split_version"] = compute_split_version(sha_bad["records"])
    sha_bad_path = tmp_dir / "sha_bad.json"
    sha_bad_path.write_text(json.dumps(sha_bad), encoding="utf-8")
    raised = False
    try:
        Split.load(role=None, manifest_path=sha_bad_path)
    except SplitError:
        raised = True
    check("sha256 mismatch rejected", raised)

    raised = False
    try:
        Split.load(role="not_a_role")
    except RoleError:
        raised = True
    check("unknown role rejected", raised)

    raised = False
    try:
        eval_split.get("does_not_exist")
    except KeyError:
        raised = True
    check("missing image_id raises KeyError", raised)

    raised = False
    try:
        eval_split.get(cal_split.ids()[0])
    except KeyError:
        raised = True
    check("other-role id is absent from split", raised)

    raised = False
    try:
        load_tokens(split=eval_split, strict=True)
    except SplitError:
        raised = True
    check("strict token load rejects other-role extras", raised)
    shutil.rmtree(tmp_dir, ignore_errors=True)

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
    token_values = {value["token"] for value in all_tokens.values()}
    wrong_values = {value["wrong_token"] for value in all_tokens.values()}
    check("token and wrong-token sets do not collide", not (token_values & wrong_values),
          f"overlap={len(token_values & wrong_values)}")

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

    approx_raised = False
    try:
        build_report("selftest", "evaluation", eval_split, synthetic,
                     approximate=True)
    except ValueError:
        approx_raised = True
    check("evaluation report rejects approximate=True", approx_raised)

    split_role_raised = False
    try:
        build_report("selftest", "calibration", eval_split, synthetic,
                     approximate=True)
    except ValueError:
        split_role_raised = True
    check("report rejects mismatched split role", split_role_raised)

    class_raised = False
    try:
        build_report("selftest", "evaluation", eval_split,
                     [{"image_id": synthetic[0]["image_id"], "class": "style",
                       "snr_db": 10, "mse": 0.1}], approximate=False)
    except ValueError:
        class_raised = True
    check("report rejects class mismatch", class_raised)

    snr_raised = False
    try:
        build_report("selftest", "evaluation", eval_split,
                     [{"image_id": synthetic[0]["image_id"], "class": synthetic[0]["class"],
                       "mse": 0.1}], approximate=False)
    except ValueError:
        snr_raised = True
    check("report rejects missing snr_db", snr_raised)

    duplicate_raised = False
    try:
        build_report("selftest", "evaluation", eval_split,
                     [synthetic[0], dict(synthetic[0])], approximate=False)
    except ValueError:
        duplicate_raised = True
    check("report rejects duplicate image+SNR", duplicate_raised)

    only_mse = build_report("selftest", "evaluation", eval_split,
                            [{"image_id": synthetic[0]["image_id"],
                              "class": synthetic[0]["class"], "snr_db": 5,
                              "mse": 0.02}], approximate=False)
    check("missing metrics are omitted, not zero-filled",
          set(only_mse["cells"][0]["metrics"]) == {"mse"},
          sorted(only_mse["cells"][0]["metrics"]))

    from evaluation.report import report_digest, write_report
    out_path = write_report(report, PROJECT_ROOT / "temp" / "phase6_report_check.json")
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    check("write_report writes valid JSON",
          reloaded["split_version"] == report["split_version"]
          and reloaded["approximate"] is False)
    check("report_digest ignores created_utc",
          report_digest(report) == report_digest(dict(report, created_utc="later")))

    math_values = [20.0, 24.0, 22.0]
    math_records = eval_split.records("common")[:3]
    math_results = [
        {"image_id": record["image_id"], "class": "common", "snr_db": 7,
         "psnr": value, "mse": 0.01}
        for record, value in zip(math_records, math_values)
    ]
    math_report = build_report("selftest", "evaluation", eval_split, math_results,
                               approximate=False)
    math_cell = math_report["cells"][0]
    math_dist = math_cell["metrics"]["psnr"]
    expected_mean = float(np.mean(math_values))
    expected_std = float(np.std(math_values, ddof=0))
    expected_p50 = float(np.percentile(math_values, 50))
    math_ok = (
        math_cell["n"] == 3
        and abs(math_dist["mean"] - expected_mean) <= 1e-9
        and abs(math_dist["std"] - expected_std) <= 1e-9
        and abs(math_dist["p50"] - expected_p50) <= 1e-9
        and math_dist["n_finite"] == 3
    )
    check("report distribution matches hand computation", math_ok,
          f"mean={math_dist['mean']} expected={expected_mean}")

    check("report metadata matches split",
          report["n_expected"] == len(eval_split)
          and report["split_counts"] == m_manifest["counts"]
          and report["corpus_version"] == m_manifest["corpus_version"]
          and report["split_version"] == m_manifest["split_version"])

    hash_a = build_report("selftest", "evaluation", eval_split, synthetic,
                          approximate=False, config={"x": 1})["config_hash"]
    hash_b = build_report("selftest", "evaluation", eval_split, synthetic,
                          approximate=False, config={"x": 1})["config_hash"]
    hash_c = build_report("selftest", "evaluation", eval_split, synthetic,
                          approximate=False, config={"x": 2})["config_hash"]
    check("config_hash stable and sensitive", hash_a == hash_b and hash_a != hash_c)

    print("=== canonicalisation ===", flush=True)
    import importlib.util
    from PIL import Image as PILImage
    builder_path = PROJECT_ROOT / "scripts" / "phase6_build_testset.py"
    module_spec = importlib.util.spec_from_file_location("p6_builder", builder_path)
    builder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(builder)

    canon_dir = PROJECT_ROOT / "temp" / "phase6_canon"
    if canon_dir.exists():
        shutil.rmtree(canon_dir)
    canon_dir.mkdir(parents=True, exist_ok=True)
    from evaluation.splits import sha256_file
    cases = {
        "wide_rgb": PILImage.new("RGB", (700, 300), (20, 40, 60)),
        "tall_gray": PILImage.new("L", (300, 700), 128),
        "rgba": PILImage.new("RGBA", (512, 300), (10, 20, 30, 77)),
    }
    deterministic = True
    size_ok = True
    for name, image in cases.items():
        source = canon_dir / f"{name}.png"
        image.save(source)
        first = canon_dir / f"{name}_a.png"
        second = canon_dir / f"{name}_b.png"
        builder.canonicalise(source, first)
        builder.canonicalise(source, second)
        deterministic = deterministic and sha256_file(first) == sha256_file(second)
        with PILImage.open(first) as output:
            size_ok = size_ok and output.size == (512, 512) and output.mode == "RGB"
    check("canonicalise is deterministic", deterministic)
    check("canonicalise outputs 512x512 RGB for RGB/gray/RGBA", size_ok)

    edge_cases = {
        "tiny": (PILImage.new("RGB", (1, 1), (1, 2, 3)), ".png"),
        "palette": (PILImage.new("P", (10, 10)), ".png"),
        "cmyk": (PILImage.new("CMYK", (20, 30), (0, 10, 20, 30)), ".tiff"),
    }
    edge_ok = True
    for name, (image, suffix) in edge_cases.items():
        source = canon_dir / f"{name}{suffix}"
        output_path = canon_dir / f"{name}_out.png"
        try:
            image.save(source)
            builder.canonicalise(source, output_path)
            with PILImage.open(output_path) as output:
                edge_ok = edge_ok and output.size == (512, 512) and output.mode == "RGB"
        except Exception:  # noqa: BLE001
            edge_ok = False
    check("canonicalise handles 1x1, palette and CMYK", edge_ok)
    shutil.rmtree(canon_dir, ignore_errors=True)

    print("=== cross-checks ===", flush=True)
    from evaluation.metrics import ssim, lpips_metric
    check("ssim is symmetric", abs(ssim(base, noisy) - ssim(noisy, base)) <= 1e-6)
    check("lpips is symmetric", abs(lpips_metric(base, noisy) - lpips_metric(noisy, base)) <= 1e-5)

    records = eval_split.records()
    copy_test = dict(records[0])
    copy_test["class"] = "style"
    check("records() returns defensive copies", eval_split.records()[0]["class"] != "style")
    manifest_copy = eval_split.manifest
    manifest_copy["split_version"] = "mutated"
    check("manifest returns a defensive copy", eval_split.version != "mutated")

    unknown_class_raised = False
    try:
        eval_split.records("nope")
    except ValueError:
        unknown_class_raised = True
    check("unknown class rejected", unknown_class_raised)

    type_raised = False
    try:
        eval_split.assert_disjoint("not a split")
    except TypeError:
        type_raised = True
    check("assert_disjoint rejects non-Split", type_raised)

    index_for_overlap = json.loads(IMAGE_INDEX_PATH.read_text(encoding="utf-8"))
    corpus_record = index_for_overlap["images"][0]
    fake = Split([{
        "image_id": "fake_overlap", "role": "evaluation", "class": "common",
        "path": corpus_record["image_path"], "sha256": corpus_record["sha256"],
    }], {"split_version": "x"}, role="evaluation", verify=False)
    overlap_raised = False
    try:
        fake.assert_disjoint_from_index()
    except SplitError:
        overlap_raised = True
    check("retrieval-corpus overlap detected", overlap_raised)

    corpus_ids = {record["source_id"] for record in index_for_overlap["images"]}
    eval_common_ids = {record["original_id"] for record in records
                       if record["class"] == "common"}
    check("evaluation common ids disjoint from corpus ids",
          not (corpus_ids & eval_common_ids),
          f"overlap={len(corpus_ids & eval_common_ids)}")

    import hashlib as _hashlib
    token_formula_ok = True
    for record in records + cal_split.records():
        value = int(_hashlib.sha256(record["image_id"].encode("utf-8")).hexdigest()[:16], 16)
        expected = f"{value % 100_000_000:08d}"
        if all_tokens[record["image_id"]]["token"] != expected:
            token_formula_ok = False
    check("tokens reproducible from image_id", token_formula_ok)

    inf_ok = True
    try:
        inf_report = build_report(
            "selftest", "evaluation", eval_split,
            [{"image_id": records[0]["image_id"], "class": records[0]["class"],
              "snr_db": 5, "psnr": float("inf")}], approximate=False)
        inf_ok = inf_report["cells"][0]["metrics"]["psnr"]["n_inf"] == 1
    except Exception:  # noqa: BLE001
        inf_ok = False
    check("report accepts infinite PSNR and counts n_inf", inf_ok)

    nonfinite_raised = False
    try:
        build_report("selftest", "evaluation", eval_split,
                     [{"image_id": records[0]["image_id"], "class": records[0]["class"],
                       "snr_db": 5, "mse": float("inf")}], approximate=False)
    except ValueError:
        nonfinite_raised = True
    check("report rejects non-finite non-PSNR metrics", nonfinite_raised)

    ordered_results = []
    for cls_name, snr_value in (("style", 20), ("common", 5), ("facial", 10)):
        sample = eval_split.records(cls_name)[0]
        ordered_results.append({
            "image_id": sample["image_id"], "class": cls_name,
            "snr_db": snr_value, "mse": 0.01,
        })
    ordered_report = build_report("selftest", "evaluation", eval_split,
                                  ordered_results, approximate=False)
    order = [(cell["class"], cell["snr_db"]) for cell in ordered_report["cells"]]
    check("cells sorted by class then SNR",
          order == sorted(order, key=lambda item: (["common", "facial", "style"].index(item[0]),
                                                   item[1])),
          order)
    check("cell n sums to result count",
          sum(cell["n"] for cell in ordered_report["cells"]) == len(ordered_results))

    nan_report = dict(ordered_report)
    nan_report["meta"] = {"bad": float("nan")}
    nan_path = PROJECT_ROOT / "temp" / "phase6_nan_report.json"
    nan_raised = False
    try:
        write_report(nan_report, nan_path)
    except ValueError:
        nan_raised = True
    leftover = nan_path.with_suffix(nan_path.suffix + ".tmp").exists()
    check("write_report rejects NaN and cleans temp", nan_raised and not leftover,
          f"raised={nan_raised} leftover={leftover}")

    from common.paths import CALIBRATION_DIR, DEVIATIONS_PATH, TEST_SET_DIR, TEST_SPLIT_PATH, TOKENS_PATH
    paths_ok = (CALIBRATION_DIR.is_dir() and TEST_SET_DIR.is_dir()
                and TEST_SPLIT_PATH.is_file() and TOKENS_PATH.is_file()
                and DEVIATIONS_PATH.is_file())
    check("paths module constants resolve", paths_ok)

    config_ok = (list(eval_cfg["classes"]) == list(CLASSES)
                 and int(eval_cfg["per_class"]) * len(CLASSES) == int(eval_cfg["total"])
                 and all(isinstance(value, int) for value in eval_cfg["snr_db"])
                 and eval_cfg["snr_db"] == sorted(eval_cfg["snr_db"]))
    check("config is internally consistent", config_ok)

    duplicate_key_free = True
    try:
        yaml.load((PROJECT_ROOT / "configs/base.yaml").read_text(encoding="utf-8"),
                  Loader=UniqueKeyLoader)
    except ValueError as error:
        duplicate_key_free = False
        check("config has no duplicate YAML keys", False, str(error))
    if duplicate_key_free:
        check("config has no duplicate YAML keys", True)

    from evaluation.splits import resolve_path as _resolve
    check("relative manifest paths resolve under project root",
          _resolve(records[0]["path"]).exists()
          and str(_resolve(records[0]["path"])).startswith(str(PROJECT_ROOT)))

    builder_guard = False
    try:
        builder.fetch_eval_class("not_a_class", {}, 1)
    except ValueError:
        builder_guard = True
    check("builder rejects unknown eval class", builder_guard)

    check("hf_image_url handles dict, string and missing",
          builder.hf_image_url({"row": {"image": {"src": "http://a"}}}) == "http://a"
          and builder.hf_image_url({"row": {"image": "http://b"}}) == "http://b"
          and builder.hf_image_url({"row": {}}) is None)

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
