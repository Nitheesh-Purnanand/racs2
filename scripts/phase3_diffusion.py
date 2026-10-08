"""
Phase 3 — Diffusion Generation Cost & Output Verification for RACS2.

Runs Stable Diffusion v1.5 to generate reference images and measures the cost
(wall-clock time + peak VRAM) of the stage that RACS replaces with retrieval.

IMPORTANT: unlike the VAE stage, diffusion generation has **no input image**. The
reference is synthesised from a text prompt + a random seed. So this script also
contains an explicit VERIFICATION section that *proves* the pipeline is really
generating content and that inputs actually change the output:

  1. Determinism : same (prompt, steps, guidance, seed) -> byte-identical image.
  2. Seed effect : same steps, different seed           -> different image.
  3. Steps effect: same seed, different step count      -> different image.
  4. Guidance    : CFG off (1.0) vs on (7.5)            -> different image.
  5. Non-trivial : generated image is not a flat/constant block (pixel std > 0).

Each check compares SHA-256 hashes of the raw pixels and the mean absolute pixel
difference, so you can see *quantitatively* that the image changed.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\phase3_diffusion.py
"""

import hashlib
import json
import sys
from pathlib import Path

# Make the project root importable + redirect the HF cache to D:.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.paths  # noqa: E402,F401

import numpy as np
import yaml
import torch

from common.seed import set_seed
from common.device import get_device
from common.paths import RESULTS_DIR, PROJECT_ROOT as _ROOT
from models.sd_pipeline import load_pipeline, generate

TIMINGS_PATH = RESULTS_DIR / "phase3_timings.json"


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def image_hash(img) -> str:
    """SHA-256 of the raw RGB bytes. Identical images -> identical hash."""
    return hashlib.sha256(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def mean_abs_diff(a, b) -> float:
    """Mean absolute per-pixel difference in [0, 255] units."""
    x = np.asarray(a.convert("RGB"), dtype=np.float32)
    y = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.abs(x - y).mean())


def pixel_std(img) -> float:
    """Std-dev of pixel values. ~0 means a flat/blank image."""
    return float(np.asarray(img.convert("RGB"), dtype=np.float32).std())


def timed_generate(pipe, prompt, steps, guidance, seed):
    """Generates one image and returns (image, seconds, peak_vram_gb)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    img, secs = generate(pipe, prompt, steps, guidance, seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    return img, secs, peak_gb


def main() -> None:
    # 1-2. seed + device
    section("1-2. Seed & device")
    set_seed(42, verbose=True)
    device = get_device()
    print(f"Selected device: {device}")
    if torch.cuda.is_available():
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

    # 3. config
    section("3. Config")
    with open(_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    gen_cfg = config.get("generation", {})
    prompt = gen_cfg.get("prompt", "a photograph of an astronaut riding a horse")
    guidance = float(gen_cfg.get("guidance_scale", 7.5))
    steps_sweep = list(gen_cfg.get("steps_sweep", [20, 30, 50]))
    seeds_sweep = list(gen_cfg.get("seeds_sweep", [42, 123]))
    print(f"prompt          = {prompt!r}")
    print(f"guidance_scale  = {guidance}")
    print(f"steps_sweep     = {steps_sweep}")
    print(f"seeds_sweep     = {seeds_sweep}")

    # 4. load pipeline
    section("4. Load SD v1.5 pipeline")
    pipe = load_pipeline(device)
    print(f"Scheduler: {type(pipe.scheduler).__name__}")

    # 5. warm-up (discard; first call compiles CUDA kernels)
    section("5. Warm-up (discarded)")
    _, warm_secs, _ = timed_generate(pipe, prompt, 5, guidance, 0)
    print(f"Warm-up (5 steps): {warm_secs:.2f}s")

    # 6. timing sweep
    section("6. Timing sweep")
    records = []
    for steps in steps_sweep:
        for seed in seeds_sweep:
            img, secs, peak_gb = timed_generate(pipe, prompt, steps, guidance, seed)
            out_path = RESULTS_DIR / f"phase3_seed{seed}_steps{steps}.png"
            img.save(out_path)
            rec = {
                "prompt": prompt,
                "num_inference_steps": int(steps),
                "guidance_scale": guidance,
                "seed": int(seed),
                "seconds": round(secs, 4),
                "seconds_per_step": round(secs / steps, 4),
                "peak_vram_gb": round(peak_gb, 3),
                "image_hash": image_hash(img),
                "pixel_std": round(pixel_std(img), 3),
                "image_path": str(out_path.relative_to(_ROOT)).replace("\\", "/"),
            }
            records.append(rec)
            print(
                f"  steps={steps:3d} seed={seed:4d} | {secs:6.2f}s "
                f"({secs/steps:5.3f}s/step) | peakVRAM {peak_gb:4.2f} GB | "
                f"hash {rec['image_hash'][:12]} std {rec['pixel_std']:.1f}"
            )

    # 7. verification that inputs actually drive the output
    section("7. VERIFICATION (prove inputs change the output)")

    # 7a. Determinism: same inputs twice -> identical bytes.
    a, _, _ = timed_generate(pipe, prompt, 20, guidance, 42)
    b, _, _ = timed_generate(pipe, prompt, 20, guidance, 42)
    det_ok = image_hash(a) == image_hash(b)
    print(f"[determinism] same seed/steps twice -> identical: {det_ok}  (mad={mean_abs_diff(a,b):.4f})")

    # 7b. Seed effect.
    c, _, _ = timed_generate(pipe, prompt, 20, guidance, 123)
    seed_diff = mean_abs_diff(a, c)
    seed_ok = image_hash(a) != image_hash(c)
    print(f"[seed effect] seed 42 vs 123        -> different : {seed_ok}  (mad={seed_diff:.2f})")

    # 7c. Steps effect.
    d, _, _ = timed_generate(pipe, prompt, 50, guidance, 42)
    steps_diff = mean_abs_diff(a, d)
    steps_ok = image_hash(a) != image_hash(d)
    print(f"[steps effect] 20 vs 50 steps       -> different : {steps_ok}  (mad={steps_diff:.2f})")

    # 7d. Guidance effect (CFG off vs on).
    e, _, _ = timed_generate(pipe, prompt, 20, 1.0, 42)
    guid_diff = mean_abs_diff(a, e)
    guid_ok = image_hash(a) != image_hash(e)
    print(f"[guidance] 1.0 vs {guidance}          -> different : {guid_ok}  (mad={guid_diff:.2f})")

    # 7e. Non-trivial output.
    nontrivial = pixel_std(a) > 1.0
    print(f"[non-trivial] pixel std {pixel_std(a):.2f} > 1.0     -> {nontrivial}")

    a.save(RESULTS_DIR / "phase3_verify_base_seed42_steps20.png")
    e.save(RESULTS_DIR / "phase3_verify_nocfg_seed42_steps20.png")

    all_ok = all([det_ok, seed_ok, steps_ok, guid_ok, nontrivial])
    print(f"\nVERIFICATION RESULT: {'PASS' if all_ok else 'FAIL'}")

    # 8. summary + save json
    section("8. Summary")
    if records:
        sps = [r["seconds_per_step"] for r in records]
        print(f"{'steps':>6} {'seed':>5} {'sec':>8} {'s/step':>8} {'peakGB':>8}")
        for r in records:
            print(
                f"{r['num_inference_steps']:>6} {r['seed']:>5} "
                f"{r['seconds']:>8.2f} {r['seconds_per_step']:>8.3f} {r['peak_vram_gb']:>8.2f}"
            )
        print(f"\nmean seconds/step = {sum(sps)/len(sps):.3f}  (cost model: time ~= steps x s/step)")
        print(f"max peak VRAM     = {max(r['peak_vram_gb'] for r in records):.2f} GB")

    payload = {
        "prompt": prompt,
        "guidance_scale": guidance,
        "steps_sweep": steps_sweep,
        "seeds_sweep": seeds_sweep,
        "records": records,
        "verification": {
            "determinism_same_seed_identical": det_ok,
            "seed_changes_output": seed_ok,
            "steps_change_output": steps_ok,
            "guidance_changes_output": guid_ok,
            "output_non_trivial": nontrivial,
            "pass": all_ok,
        },
    }
    TIMINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {TIMINGS_PATH}")

    section("Phase 3 complete")


if __name__ == "__main__":
    main()
