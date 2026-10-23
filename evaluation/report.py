"""
Report aggregation for RACS2 evaluation results.

Turns per-case results into the frozen artifact schema consumed by every later
phase. The role rules are enforced here as well as in ``evaluation.splits``:

  - ``role="calibration"`` always produces ``"approximate": true``.
  - ``role="evaluation"`` always produces ``"approximate": false``.
  - A cell never mixes roles or classes.

Usage:
    from evaluation.report import build_report
    report = build_report("agentsemstecom", "evaluation", split, results,
                          approximate=False)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import math
from pathlib import Path

from evaluation.metrics import distribution
from evaluation.splits import CLASSES, ROLES, Split, sha256_json

METRIC_NAMES = ("mse", "psnr", "ssim", "lpips")


def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_hash(config: dict | None) -> str | None:
    if config is None:
        return None
    return sha256_json(config)


def build_report(system: str, role: str, split: Split, results: list,
                 *, approximate: bool, meta: dict | None = None,
                 percentiles=(5, 25, 50, 75, 95), config: dict | None = None):
    """
    Aggregates per-case results into the frozen report schema.

    Each result must contain at least ``image_id``, ``class``, ``snr_db`` and one
    or more metric values. Missing metrics are omitted from a cell rather than
    zero-filled.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if not isinstance(split, Split):
        raise TypeError("split must be an evaluation.splits.Split")
    if split.role is not None and split.role != role:
        raise ValueError(
            f"split role {split.role!r} does not match report role {role!r}"
        )
    if approximate is not (role == "calibration"):
        raise ValueError(
            "approximate must be True exactly when role is 'calibration'"
        )

    known_ids = set(split.ids())
    records = {record["image_id"]: record for record in split.records()}

    cells: dict = {}
    seen = set()
    for result in results:
        image_id = result.get("image_id")
        if image_id not in known_ids:
            raise ValueError(
                f"{image_id!r} is not in the {role} split; refusing to report it"
            )
        seen_key = (image_id, result.get("snr_db"))
        if seen_key in seen:
            raise ValueError(
                f"duplicate result for {image_id!r} at SNR {result.get('snr_db')}"
            )
        seen.add(seen_key)

        record = records[image_id]
        cls = result.get("class", record["class"])
        if cls != record["class"]:
            raise ValueError(
                f"{image_id}: class {cls!r} does not match split class {record['class']!r}"
            )
        if record["role"] != role:
            raise ValueError(
                f"{image_id}: record role {record['role']!r} != report role {role!r}"
            )
        snr = result.get("snr_db")
        if snr is None:
            raise ValueError(f"{image_id}: missing snr_db")

        key = (cls, int(snr))
        bucket = cells.setdefault(key, {"metrics": {name: [] for name in METRIC_NAMES},
                                        "extra": {}, "n": 0})
        bucket["n"] += 1
        for name in METRIC_NAMES:
            if name not in result:
                continue
            value = result[name]
            if value is not None and name != "psnr":
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(
                        f"{image_id}: {name} must be finite, got {value!r}"
                    )
            bucket["metrics"][name].append(value)
        for name in ("reference_latency_s", "total_latency_s", "peak_vram_gb"):
            if name in result:
                bucket["extra"].setdefault(name, []).append(result[name])

    out_cells = []
    for (cls, snr) in sorted(cells, key=lambda k: (CLASSES.index(k[0]), k[1])):
        bucket = cells[(cls, snr)]
        metrics_out = {}
        for name in METRIC_NAMES:
            values = bucket["metrics"][name]
            if values:
                metrics_out[name] = distribution(values, percentiles=percentiles)
        extra_out = {name: distribution(values, percentiles=percentiles)
                     for name, values in bucket["extra"].items()}
        out_cells.append({
            "class": cls,
            "snr_db": snr,
            "n": bucket["n"],
            "metrics": metrics_out,
            "extra": extra_out,
        })

    manifest = split.manifest
    return {
        "schema_version": 1,
        "system": system,
        "role": role,
        "approximate": bool(approximate),
        "split_version": split.version,
        "split_counts": manifest.get("counts"),
        "corpus_version": manifest.get("corpus_version"),
        "config_hash": config_hash(config),
        "lpips_net": (meta or {}).get("lpips_net"),
        "percentiles": list(percentiles),
        "created_utc": _now(),
        "meta": meta or {},
        "cells": out_cells,
        "n_results": len(results),
        "n_expected": len(split),
    }


def write_report(report: dict, out_path):
    """Atomically writes a report artifact."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def report_digest(report: dict) -> str:
    payload = {k: v for k, v in report.items() if k != "created_utc"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
