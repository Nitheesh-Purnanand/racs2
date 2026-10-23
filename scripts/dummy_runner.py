"""
Deterministic dummy runner for the Phase 6 evaluation harness.

It proves the `eval_report.py` driver end to end before any real pipeline exists.
The "recovery" is the secret image plus seeded Gaussian noise, so metrics are
plausible and identical every run.

Runner contract:
    run_case(record, snr_db, role) -> dict with at least
        {"secret": PIL.Image, "recovered": PIL.Image}
    and optionally stego, reference_latency_s, total_latency_s, peak_vram_gb.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

import os
import sys

_BOOTSTRAP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOOTSTRAP_ROOT not in sys.path:
    sys.path.insert(0, _BOOTSTRAP_ROOT)

from evaluation.splits import resolve_path


def _seed(image_id, snr_db):
    digest = hashlib.sha256(f"{image_id}|{snr_db}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def run_case(record, snr_db, role):
    secret = Image.open(resolve_path(record["path"])).convert("RGB")
    pixels = np.asarray(secret, dtype=np.float32) / 255.0

    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    sigma = float(np.sqrt(1.0 / (snr_linear * 255.0))) if snr_linear > 0 else 1.0
    rng = np.random.default_rng(_seed(record["image_id"], snr_db))
    noisy = np.clip(pixels + rng.normal(0.0, sigma, pixels.shape), 0.0, 1.0)

    recovered = Image.fromarray((noisy * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    return {
        "secret": secret,
        "recovered": recovered,
        "stego": secret,
        "reference_latency_s": 0.0,
        "total_latency_s": 0.01,
    }
