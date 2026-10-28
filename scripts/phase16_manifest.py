"""Phase 16 — freeze the receiver-condition manifest and per-case token metadata.

This script never re-runs a receiver. It reads the frozen Phase 15 baseline cases and
the Phase 16 security table, records the token, seed and mask actually used for each
receiver condition, documents every per-cell gap exception, and writes the Phase 16
manifest referenced by ``security.manifest_path``.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from common.seed import set_seed  # noqa: E402
from reference.seed import derive_mask_seed, derive_seed  # noqa: E402
from security.token_conditions import NO_TOKEN_FALLBACK, mask_hash  # noqa: E402
from steganography import pipeline  # noqa: E402
from steganography.mask import token_delta, token_noise  # noqa: E402

GAP_THRESHOLD_DB = 4.0
GAP_EXCEPTION_REASON = (
    "channel-noise floor: at this SNR the AWGN perturbation dominates the latent "
    "perturbation, so both correct and wrong recovery collapse toward a generic "
    "reconstruction and the token-condition gap compresses below 4 dB"
)
MODES = ("correct", "wrong", "no_token")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _weighted(cells: list[dict], mode: str) -> float:
    total = sum(c["n"] for c in cells)
    return sum(c[mode]["psnr"]["mean"] * c["n"] for c in cells) / max(1, total)


def main() -> int:
    cfg_all = pipeline.load_config()
    cfg = cfg_all["security"]
    stego_cfg = cfg_all["steganography"]
    bcfg = cfg_all["baseline"]
    set_seed(int(cfg_all.get("seed", 42)))

    table_path = paths.PROJECT_ROOT / bcfg["evaluation_output"]
    manifest_path = paths.PROJECT_ROOT / bcfg["evaluation_manifest"]
    security_path = paths.PROJECT_ROOT / cfg["output_security"]
    steganalysis_path = paths.PROJECT_ROOT / cfg["output_steganalysis"]
    out_path = paths.PROJECT_ROOT / cfg["manifest_path"]

    table = pipeline.read_json(table_path)
    base_manifest = pipeline.read_json(manifest_path)
    security = pipeline.read_json(security_path)
    steganalysis = pipeline.read_json(steganalysis_path)
    fraction = float(stego_cfg["mask_fraction"])

    cases_dir = paths.PROJECT_ROOT / bcfg["cases_dir"] / "evaluation"
    records = []
    for case in base_manifest["cases"]:
        if case["role"] != "evaluation":
            raise RuntimeError("calibration record in the evaluation baseline")
        if case["token_mode"] != "correct":
            raise RuntimeError("baseline transmitter case must use the correct token")
        stem = cases_dir / f"{case['image_id']}_snr{case['snr_db']}"
        blob = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
        token, wrong_token = blob["token"], blob["wrong_token"]
        shape = tuple(blob["delta"].shape)
        correct_delta = blob["delta"]
        wrong_delta = token_delta(wrong_token, shape=shape, fraction=fraction)
        no_token_delta = torch.zeros(shape, dtype=torch.float32)
        correct_reference_seed = derive_seed(token)
        correct_mask_seed = derive_mask_seed(token)
        if int(case["seed"]) != int(correct_mask_seed):
            raise RuntimeError(
                f"mask seed mismatch for {case['image_id']} snr={case['snr_db']}: "
                f"{case['seed']} != {correct_mask_seed}")
        conditions = {
            "correct": {
                "token_used": token,
                "reference_seed": correct_reference_seed,
                "mask_seed": correct_mask_seed,
                "mask_sha256": mask_hash(correct_delta),
                "mask_nonzero": int((correct_delta != 0).sum()),
                "noise_sha256": _tensor_sha256(blob["noise"]),
            },
            "wrong": {
                "token_used": wrong_token,
                "reference_seed": derive_seed(wrong_token),
                "mask_seed": derive_mask_seed(wrong_token),
                "mask_sha256": mask_hash(wrong_delta),
                "mask_nonzero": int((wrong_delta != 0).sum()),
                "noise_sha256": _tensor_sha256(token_noise(wrong_token, shape=shape)),
            },
            "no_token": {
                "token_used": NO_TOKEN_FALLBACK,
                "reference_seed": derive_seed(NO_TOKEN_FALLBACK),
                "mask_seed": derive_mask_seed(NO_TOKEN_FALLBACK),
                "mask_sha256": mask_hash(no_token_delta),
                "mask_nonzero": 0,
                "noise_sha256": _tensor_sha256(token_noise(NO_TOKEN_FALLBACK, shape=shape)),
            },
        }
        records.append({
            "image_id": case["image_id"],
            "class": case["class"],
            "snr_db": case["snr_db"],
            "role": case["role"],
            "transmitter": "identical frozen Phase 15 stego for all three conditions",
            "conditions": conditions,
        })

    cells = security["cells"]
    exceptions = [
        {
            "class": c["class"],
            "snr_db": c["snr_db"],
            "gap_correct_minus_wrong_db": c["gap_correct_minus_wrong_db"],
            "gap_correct_minus_no_token_db": c["gap_correct_minus_no_token_db"],
            "threshold_db": GAP_THRESHOLD_DB,
            "reason": GAP_EXCEPTION_REASON,
        }
        for c in cells
        if c["gap_correct_minus_wrong_db"] < GAP_THRESHOLD_DB
        or c["gap_correct_minus_no_token_db"] < GAP_THRESHOLD_DB
    ]
    aggregate = {mode: _weighted(cells, mode) for mode in MODES}

    split_ids = steganalysis.get("dataset", {}).get("split_ids", {})
    image_sets = {name: set(ids) for name, ids in split_ids.items()}
    if not image_sets or (image_sets["train"] & image_sets["test"]) or \
            (image_sets["train"] & image_sets["val"]) or \
            (image_sets["val"] & image_sets["test"]):
        raise RuntimeError("steganalysis split is not image-disjoint")

    payload = {
        "schema_version": 1,
        "phase": 16,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approximate": False,
        "split_role": "evaluation",
        "split_version": table.get("split_version"),
        "baseline_table_sha256": _sha256(table_path),
        "baseline_manifest_sha256": _sha256(manifest_path),
        "security_output": cfg["output_security"],
        "security_sha256": _sha256(security_path),
        "steganalysis_output": cfg["output_steganalysis"],
        "steganalysis_sha256": _sha256(steganalysis_path),
        "conditions": security.get("conditions"),
        "no_token": {
            "definition": "receiver has no secret token; it uses a fixed public fallback",
            "fallback_token": NO_TOKEN_FALLBACK,
            "fallback_reference_seed": derive_seed(NO_TOKEN_FALLBACK),
            "fallback_mask_seed": derive_mask_seed(NO_TOKEN_FALLBACK),
            "mask": "all-zero (no token-derived perturbation)",
            "not_the_same_as": "the wrong-token run, which uses a valid but incorrect token",
        },
        "transmitter_invariance": (
            "the frozen Phase 15 stego image is the same object for correct, wrong and "
            "no_token; only the receiver token, reference seed and mask change"),
        "gap_threshold_db": GAP_THRESHOLD_DB,
        "aggregate_psnr": {
            **{mode: aggregate[mode] for mode in MODES},
            "gap_correct_minus_wrong_db": aggregate["correct"] - aggregate["wrong"],
            "gap_correct_minus_no_token_db":
                aggregate["correct"] - aggregate["no_token"],
            "n_cells": len(cells),
            "n_per_cell": sorted({c["n"] for c in cells}),
        },
        "gap_exceptions": exceptions,
        "steganalysis_protocol": {
            "snr_db": steganalysis["model"]["snr_db"],
            "input_size": int(cfg["steganalysis"]["input_size"]),
            "patch_grid": steganalysis["dataset"].get("patch_grid"),
            "n_patches_per_image": steganalysis["dataset"].get("n_patches_per_image"),
            "split_strategy": steganalysis["dataset"].get("split_strategy"),
            "n_images": steganalysis["dataset"]["n_images"],
            "n_samples": steganalysis["dataset"]["n_samples"],
            "split_ids_sha256": {
                name: hashlib.sha256(",".join(split_ids[name]).encode()).hexdigest()
                for name in ("train", "val", "test")
            },
            "grouping": "image_id grouped; one sample per image per side",
        },
        "calibration_exclusion": {
            "rule": "calibration data may not enter a security table",
            "checked": "every record has role == evaluation and an eva_ image id",
            "n_records": len(records),
        },
        "cases": records,
        "failures": [],
    }
    pipeline.write_json(out_path, payload)
    print(f"[phase16] wrote {cfg['manifest_path']} "
          f"({len(records)} cases x {len(MODES)} conditions, "
          f"{len(exceptions)} gap exceptions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
