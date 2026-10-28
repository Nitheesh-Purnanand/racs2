"""Run ONE image through the frozen AgentSemSteCom baseline and print metrics.

Two modes:

``--mode frozen`` (default)
    Applies the component settings that produced the committed Phase 15
    artifacts (SD-VAE codec, 20 EDICT steps, strength 0.2, guidance 3.0,
    IP-Adapter off, CodeFormer off). It works with today's working tree even
    while the paper-asset upgrade is unfinished.

``--mode paper``
    Uses configs/base.yaml exactly as-is. Requires the SwinJSCC checkpoint,
    CodeFormer weights and IP-Adapter weights referenced by the config.

By default the secret is the frozen split image for ``--image-id``, so the
result can be compared against the frozen Phase 15 case file. Pass ``--secret``
to use your own image; the frozen reference image, Kpub and token for
``--image-id`` are still used, which is how the baseline conditions the hide.

Examples
--------
    python scripts/verify_baseline_example.py
    python scripts/verify_baseline_example.py --snr 5
    python scripts/verify_baseline_example.py --secret data/secret_images/test_001.png
    python scripts/verify_baseline_example.py --mode paper --image-id eva_facial_0000 \
        --class facial --snr 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from evaluation.splits import Split  # noqa: E402
from pipelines.agentsemstecom import Baseline  # noqa: E402
from steganography import pipeline  # noqa: E402

FROZEN_OVERRIDES = {
    "codec.kind": "sd_vae_substitute",
    "enhancement.enabled": False,
    "steganography.ipadapter.enabled": False,
    "steganography.edict_steps": 20,
    "steganography.edict_strength": 0.2,
    "steganography.guidance_scale": 3.0,
}


def _apply(config: dict, dotted: str, value) -> None:
    node = config
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


def _frozen_overrides(config: dict) -> dict:
    for key, value in FROZEN_OVERRIDES.items():
        _apply(config, key, value)
    return config


def _frozen_case(image_id: str, snr_db: float, role: str) -> dict | None:
    path = (paths.PROJECT_ROOT / "experiments/results/baseline_cases" / role
            / f"{image_id}_snr{int(snr_db)}.json")
    if not path.exists():
        return None
    return pipeline.read_json(path)["result"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--secret", default="", help="your image; default = frozen split image")
    parser.add_argument("--image-id", default="eva_common_0000",
                        help="frozen case supplying reference, Kpub and token")
    parser.add_argument("--class", dest="class_name", default="common",
                        choices=("common", "facial", "style"))
    parser.add_argument("--split", default="evaluation", choices=("evaluation", "calibration"))
    parser.add_argument("--snr", type=float, default=10.0, help="channel SNR in dB")
    parser.add_argument("--mode", default="frozen", choices=("frozen", "paper"))
    parser.add_argument("--token-modes", default="correct,wrong,no_token")
    parser.add_argument("--out-dir", default="experiments/results/custom_example")
    args = parser.parse_args()

    started = time.time()
    cfg = pipeline.load_config()
    if args.mode == "frozen":
        cfg = _frozen_overrides(cfg)
        print("[example] mode=frozen (committed Phase 15 component settings)")
    else:
        print("[example] mode=paper (configs/base.yaml as-is)")

    split = Split.load(role=args.split, allow_calibration=args.split == "calibration")
    record = next((r for r in split.records(args.class_name)
                   if r["image_id"] == args.image_id), None)
    if record is None:
        print(f"[example] {args.image_id} is not in the {args.split} {args.class_name} split")
        return 2

    secret_path = args.secret or str(paths.PROJECT_ROOT / record["path"])
    if not Path(secret_path).exists():
        print(f"[example] secret image not found: {secret_path}")
        return 2

    baseline = Baseline(cfg)
    size = int(baseline.stego_cfg["image_size"])
    secret_px = pipeline.load_pixels(secret_path, size)
    reference_path = baseline.reference_path(args.split, args.class_name, args.image_id)
    reference_px = pipeline.load_pixels(reference_path, size)
    cond = baseline.conditions(args.image_id)
    tok = baseline.tokens(args.image_id)

    print(f"[example] secret={secret_path}")
    print(f"[example] reference={reference_path}")
    print(f"[example] Kpub={cond['kpub']!r}")
    print(f"[example] token={tok['token']} wrong_token={tok['wrong_token']} snr={args.snr:g} dB")

    t0 = time.time()
    hidden = baseline.hide(secret_px, reference_px, cond["kpub"], tok["token"])
    print(f"[example] hide: {time.time() - t0:.1f}s "
          f"(steps={hidden['steps']}, strength={hidden['strength']:g}, eta={hidden['eta']:g})")

    out_dir = paths.PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{Path(secret_path).stem}_{args.image_id}_snr{int(args.snr)}"
    if Path(secret_path).stem == args.image_id:
        stem = out_dir / f"{args.image_id}_snr{int(args.snr)}"
    pipeline.save_pixels(secret_px, stem.with_name(stem.name + "_secret.png"))
    pipeline.save_pixels(hidden["stego_pixels"], stem.with_name(stem.name + "_stego.png"))

    tx = baseline.transmit(hidden["stego_pixels"], role=args.split,
                           class_name=args.class_name, snr_db=args.snr)
    received = tx["output"]
    pipeline.save_pixels(received, stem.with_name(stem.name + "_received.png"))

    blob = {"delta": hidden["delta"], "noise": hidden["noise"], "eta": hidden["eta"],
            "steps": hidden["steps"], "mixing": hidden["mixing"]}
    rows, summary = [], {}
    for mode in [m.strip() for m in args.token_modes.split(",") if m.strip()]:
        t1 = time.time()
        recovered = baseline.receive(received, token_mode=mode, token=tok["token"],
                                     wrong_token=tok["wrong_token"], prompt=cond["kpub"],
                                     blob=blob, image_embeds=hidden.get("image_embeds"))
        enhanced = False
        if mode == "correct":
            recovered, enhanced = baseline.maybe_enhance(recovered, args.class_name)
        recovered_path = stem.with_name(stem.name + f"_{mode}_recovered.png")
        pipeline.save_pixels(recovered, recovered_path)
        metrics = pipeline.metrics(secret_px, recovered, baseline.stego_cfg)
        summary[mode] = {"metrics": metrics, "enhanced": bool(enhanced),
                         "recovered_path": str(recovered_path),
                         "elapsed_s": round(time.time() - t1, 2)}
        rows.append(str(recovered_path))
        print(f"[example] {mode:9s} psnr={metrics['psnr']:.2f} dB "
              f"ssim={metrics['ssim']:.4f}"
              + (f" lpips={metrics['lpips']:.4f}" if "lpips" in metrics else "")
              + f" ({time.time() - t1:.1f}s)")

    grid_path = stem.with_name(stem.name + "_grid.png")
    grid_row = [str(stem.with_name(stem.name + "_secret.png")),
                reference_path,
                str(stem.with_name(stem.name + "_stego.png"))] + rows
    pipeline.grid([grid_row], grid_path,
                  labels=tuple(["secret", "reference", "stego"]
                               + [m.strip() for m in args.token_modes.split(",") if m.strip()]))
    print(f"[example] grid={grid_path}")

    packet = {
        "schema_version": 1, "mode": args.mode, "source": "verify_baseline_example",
        "image_id": args.image_id, "class": args.class_name, "split": args.split,
        "snr_db": args.snr, "secret_path": secret_path,
        "secret_is_frozen_split_image": not args.secret,
        "reference_path": reference_path, "kpub": cond["kpub"],
        "token": tok["token"], "wrong_token": tok["wrong_token"],
        "steps": hidden["steps"], "strength": hidden["strength"], "eta": hidden["eta"],
        "stego_sha256": pipeline.tensor_sha256(hidden["stego_pixels"]),
        "received_sha256": pipeline.tensor_sha256(received),
        "channel": tx["meta"], "cases": summary,
        "grid_path": str(grid_path), "elapsed_s": round(time.time() - started, 1),
    }

    if not args.secret and args.mode == "frozen":
        expected = _frozen_case(args.image_id, args.snr, args.split)
        if expected:
            correct = summary["correct"]["metrics"]
            packet["frozen_case"] = {
                "expected": {k: expected[k] for k in ("mse", "psnr", "ssim", "lpips")
                             if k in expected},
                "delta": {k: correct[k] - expected[k] for k in correct if k in expected},
            }
            delta = packet["frozen_case"]["delta"]
            print(f"[example] vs frozen case: psnr delta={delta.get('psnr', 0):+.3f} dB, "
                  f"ssim delta={delta.get('ssim', 0):+.4f}")

    pipeline.write_json(stem.with_name(stem.name + ".json"), packet)
    print(f"[example] wrote {stem.with_name(stem.name + '.json')} "
          f"({time.time() - started:.1f}s total)")
    baseline.edict.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
