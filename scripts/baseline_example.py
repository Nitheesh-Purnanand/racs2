"""Run ONE image end to end through the AgentSemSteCom baseline.

This is the friendly entrypoint for inspecting one image end to end. It hides a
secret image, transmits the stego through the configured channel, recovers it
under three token conditions, saves every intermediate image, and writes a JSON
metrics packet.

Modes
-----
``frozen`` (default)
    Applies the component settings that produced the committed Phase 15
    artifacts (SD-VAE codec, 20 EDICT steps, strength 0.2, guidance 3.0,
    IP-Adapter off, CodeFormer off). Works in the current working tree even
    while the paper-asset upgrade is unfinished.

``paper``
    Uses ``configs/base.yaml`` exactly as-is. Requires the SwinJSCC,
    CodeFormer and IP-Adapter assets referenced by the config.

Examples
--------
    .venv\\Scripts\\python.exe scripts\\baseline_example.py
    .venv\\Scripts\\python.exe scripts\\baseline_example.py --list-images --class facial
    .venv\\Scripts\\python.exe scripts\\baseline_example.py --image-id eva_facial_0000 --class facial --snr 10
    .venv\\Scripts\\python.exe scripts\\baseline_example.py --image D:\\pics\\my.png --snr 15
    .venv\\Scripts\\python.exe scripts\\baseline_example.py --image my.png --image-id eva_common_0000 \\
        --prompt "a photo of a red bus" --token 12345678 --wrong-token 87654321 --snr 20

    # Fully custom, no frozen split anchor
    .venv\\Scripts\\python.exe scripts\\baseline_example.py --no-anchor \\
        --image my.png --reference my_ref.png --prompt "a photo of a red bus" \\
        --token 12345678 --wrong-token 87654321 --snr 20

Outputs go to ``experiments/results/baseline_example`` by default. The grid contains
secret | reference | stego | correct | wrong | no_token.
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
from common.seed import set_seed  # noqa: E402
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


def _apply_frozen(config: dict) -> dict:
    for key, value in FROZEN_OVERRIDES.items():
        _apply(config, key, value)
    return config


def _frozen_case(image_id: str, snr_db: float, role: str) -> dict | None:
    path = (paths.PROJECT_ROOT / "experiments/results/baseline_cases" / role
            / f"{image_id}_snr{int(snr_db)}.json")
    if not path.exists():
        return None
    return pipeline.read_json(path)["result"]


def _list_images(split: Split, class_name: str) -> None:
    records = split.records(class_name)
    print(f"{split.role} / {class_name}: {len(records)} images")
    for rec in records:
        print(f"  {rec['image_id']}  {rec['path']}")


def _open_path(path: Path) -> None:
    try:
        import os
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as err:  # noqa: BLE001
        print(f"[manual] could not open {path}: {err}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", default="",
                        help="your secret image; default is the frozen split image")
    parser.add_argument("--reference", default="",
                        help="reference image; default is the frozen reference")
    parser.add_argument("--prompt", default="",
                        help="Kpub text; default is the frozen Kpub")
    parser.add_argument("--token", default="", help="token; default is the frozen token")
    parser.add_argument("--wrong-token", default="",
                        help="wrong token; default is the frozen wrong token")
    parser.add_argument("--image-id", default="eva_common_0000",
                        help="frozen record used as the conditioning anchor")
    parser.add_argument("--no-anchor", action="store_true",
                        help="do not load a frozen anchor; requires --image, --reference, "
                             "--prompt and --token")
    parser.add_argument("--class", dest="class_name", default="common",
                        choices=("common", "facial", "style"))
    parser.add_argument("--split", default="evaluation",
                        choices=("evaluation", "calibration"))
    parser.add_argument("--snr", type=float, default=10.0, help="channel SNR in dB")
    parser.add_argument("--mode", default="frozen", choices=("frozen", "paper"))
    parser.add_argument("--token-modes", default="correct,wrong,no_token")
    parser.add_argument("--out-dir", default="experiments/results/baseline_example")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open", action="store_true", dest="open_output",
                        help="open the output grid when finished")
    parser.add_argument("--list-images", action="store_true",
                        help="list frozen image ids for --class and exit")
    parser.add_argument("--list-classes", action="store_true",
                        help="list frozen classes and exit")
    args = parser.parse_args(argv)

    split = Split.load(role=args.split, allow_calibration=args.split == "calibration")

    if args.list_classes:
        print("classes:", ", ".join(sorted({r["class"] for r in split.records()})))
        return 0
    if args.list_images:
        _list_images(split, args.class_name)
        return 0

    started = time.time()
    set_seed(args.seed)
    cfg = pipeline.load_config()
    if args.mode == "frozen":
        cfg = _apply_frozen(cfg)
        print("[manual] mode=frozen (committed Phase 15 component settings)")
    else:
        print("[manual] mode=paper (configs/base.yaml as-is)")

    record = None
    if not args.no_anchor:
        record = next((r for r in split.records(args.class_name)
                       if r["image_id"] == args.image_id), None)
        if record is None:
            print(f"[manual] {args.image_id} is not in the {args.split} "
                  f"{args.class_name} split")
            print("[manual] use --list-images to see valid ids, or --no-anchor "
                  "for a fully custom run")
            return 2

    if record is None:
        missing = [name for name, value in (
            ("--image", args.image), ("--reference", args.reference),
            ("--prompt", args.prompt), ("--token", args.token)) if not value]
        if missing:
            print(f"[manual] --no-anchor requires {', '.join(missing)}")
            return 2
        print("[manual] no frozen anchor; using fully custom image, reference, "
              "Kpub and token")

    anchor_id = args.image_id if record is not None else "custom"

    secret_path = args.image or str(paths.PROJECT_ROOT / record["path"])
    if not Path(secret_path).exists():
        print(f"[manual] secret image not found: {secret_path}")
        return 2

    baseline = Baseline(cfg)
    size = int(baseline.stego_cfg["image_size"])
    secret_px = pipeline.load_pixels(secret_path, size)

    if record is not None:
        cond = baseline.conditions(args.image_id)
        tok = baseline.tokens(args.image_id)
        reference_path = args.reference or baseline.reference_path(
            args.split, args.class_name, args.image_id)
    else:
        cond = {"kpub": "", "feature_kind": "custom"}
        tok = {"token": "", "wrong_token": ""}
        reference_path = args.reference
    if not Path(reference_path).exists():
        print(f"[manual] reference image not found: {reference_path}")
        return 2
    reference_px = pipeline.load_pixels(reference_path, size)

    prompt = args.prompt or cond["kpub"]
    token = args.token or tok["token"]
    wrong_token = args.wrong_token or tok.get("wrong_token", "")
    if not wrong_token:
        wrong_token = "00000001" if token != "00000001" else "00000002"

    print(f"[manual] secret    = {secret_path}")
    print(f"[manual] reference = {reference_path}")
    print(f"[manual] Kpub      = {prompt!r}")
    print(f"[manual] token={token} wrong_token={wrong_token} snr={args.snr:g} dB")

    t0 = time.time()
    hidden = baseline.hide(secret_px, reference_px, prompt, token)
    print(f"[manual] hide: {time.time() - t0:.1f}s "
          f"(steps={hidden['steps']}, strength={hidden['strength']:g}, "
          f"eta={hidden['eta']:g})")

    out_dir = paths.PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{Path(secret_path).stem}_{anchor_id}_snr{int(args.snr)}"
    if Path(secret_path).stem == anchor_id:
        stem = out_dir / f"{anchor_id}_snr{int(args.snr)}"

    pipeline.save_pixels(secret_px, stem.with_name(stem.name + "_secret.png"))
    pipeline.save_pixels(hidden["stego_pixels"], stem.with_name(stem.name + "_stego.png"))

    tx = baseline.transmit(hidden["stego_pixels"], role=args.split,
                           class_name=args.class_name, snr_db=args.snr)
    received = tx["output"]
    pipeline.save_pixels(received, stem.with_name(stem.name + "_received.png"))

    blob = {"delta": hidden["delta"], "noise": hidden["noise"], "eta": hidden["eta"],
            "steps": hidden["steps"], "mixing": hidden["mixing"]}
    modes = [m.strip() for m in args.token_modes.split(",") if m.strip()]
    rows, summary = [], {}
    for mode in modes:
        t1 = time.time()
        recovered = baseline.receive(received, token_mode=mode, token=token,
                                     wrong_token=wrong_token, prompt=prompt,
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
        extra = f" lpips={metrics['lpips']:.4f}" if "lpips" in metrics else ""
        print(f"[manual] {mode:9s} psnr={metrics['psnr']:.2f} dB "
              f"ssim={metrics['ssim']:.4f}{extra} ({time.time() - t1:.1f}s)")

    grid_path = stem.with_name(stem.name + "_grid.png")
    grid_row = [str(stem.with_name(stem.name + "_secret.png")),
                reference_path,
                str(stem.with_name(stem.name + "_stego.png"))] + rows
    pipeline.grid([grid_row], grid_path,
                  labels=tuple(["secret", "reference", "stego"] + modes))
    print(f"[manual] grid={grid_path}")

    packet = {
        "schema_version": 1, "mode": args.mode, "source": "baseline_example",
        "image_id": anchor_id, "class": args.class_name, "split": args.split,
        "snr_db": args.snr, "secret_path": secret_path,
        "secret_is_frozen_split_image": record is not None and not args.image,
        "no_frozen_anchor": args.no_anchor,
        "reference_path": reference_path, "kpub": prompt,
        "token": token, "wrong_token": wrong_token,
        "steps": hidden["steps"], "strength": hidden["strength"], "eta": hidden["eta"],
        "stego_sha256": pipeline.tensor_sha256(hidden["stego_pixels"]),
        "received_sha256": pipeline.tensor_sha256(received),
        "channel": tx["meta"], "cases": summary,
        "grid_path": str(grid_path), "elapsed_s": round(time.time() - started, 1),
    }
    custom = bool(args.image or args.reference or args.prompt or args.token)
    packet["custom_inputs"] = custom

    if not custom and args.mode == "frozen":
        expected = _frozen_case(args.image_id, args.snr, args.split)
        if expected:
            correct = summary.get("correct", {}).get("metrics", {})
            packet["frozen_case"] = {
                "expected": {k: expected[k] for k in ("mse", "psnr", "ssim", "lpips")
                             if k in expected},
                "delta": {k: correct[k] - expected[k] for k in correct if k in expected},
            }
            delta = packet["frozen_case"]["delta"]
            print(f"[manual] vs frozen case: psnr delta={delta.get('psnr', 0):+.3f} dB, "
                  f"ssim delta={delta.get('ssim', 0):+.4f}")

    packet_path = stem.with_name(stem.name + ".json")
    pipeline.write_json(packet_path, packet)
    print(f"[manual] wrote {packet_path} ({time.time() - started:.1f}s total)")

    if args.open_output:
        _open_path(grid_path)

    baseline.edict.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
