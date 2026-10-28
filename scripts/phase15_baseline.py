"""Phase 15 — frozen AgentSemSteCom baseline on calibration (15A) or evaluation (15B)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from common.seed import set_seed  # noqa: E402
from evaluation.report import build_report, config_hash, write_report  # noqa: E402
from evaluation.splits import CLASSES, Split, load_tokens  # noqa: E402
from pipelines.agentsemstecom import Baseline, run_case, tensor_hash  # noqa: E402
from steganography import pipeline  # noqa: E402

METRICS = ("mse", "psnr", "ssim", "lpips")


def _code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.PROJECT_ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _probe(args, cfg_all):
    split = Split.load(role=args.split, allow_calibration=args.split == "calibration")
    rec = split.get(args.probe)
    baseline = Baseline(cfg_all)
    tok = baseline.tokens(rec["image_id"])
    out = run_case(baseline=baseline, secret_px=pipeline.load_pixels(
        str(paths.PROJECT_ROOT / rec["path"]), int(baseline.stego_cfg["image_size"])),
        class_name=rec["class"], image_id=rec["image_id"], token=tok["token"],
        wrong_token=tok["wrong_token"], snr_db=float(args.probe_snr),
        split_role=args.split)
    print(json.dumps({"stego_sha256": out["stego_sha256"],
                      "recovered_sha256": out["recovered_sha256"],
                      "reference_sha256": out["reference_sha256"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--class", dest="class_name", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe", default="")
    parser.add_argument("--probe-snr", type=float, default=10.0)
    args = parser.parse_args()

    cfg_all = pipeline.load_config()
    if args.probe:
        return _probe(args, cfg_all)

    bcfg = cfg_all["baseline"]
    role = args.split
    approximate = role == "calibration"
    split = Split.load(role=role, allow_calibration=approximate)
    token_map, _tokens_version = load_tokens(split=split)
    tokens = token_map
    set_seed(int(cfg_all.get("seed", 42)))

    classes = [args.class_name] if args.class_name else list(bcfg.get("classes", CLASSES))
    snrs = [float(s) for s in bcfg.get("snr_db", cfg_all["channel"]["snr_db"])]
    cases_dir = paths.PROJECT_ROOT / bcfg["cases_dir"] / role
    cases_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    baseline = Baseline(cfg_all)
    results, manifests, failures = [], [], []

    for class_name in classes:
        recs = split.records(class_name)
        if args.limit:
            recs = recs[: args.limit]
        for rec in recs:
            image_id = rec["image_id"]
            secret_px = pipeline.load_pixels(str(paths.PROJECT_ROOT / rec["path"]),
                                             int(baseline.stego_cfg["image_size"]))
            tok = tokens[image_id]
            hidden = baseline.hide(secret_px, pipeline.load_pixels(
                baseline.reference_path(role, class_name, image_id),
                int(baseline.stego_cfg["image_size"])), baseline.conditions(image_id)["kpub"],
                tok["token"])
            stego_path = cases_dir / f"{image_id}_stego.png"
            pipeline.save_pixels(hidden["stego_pixels"], stego_path)
            for snr in snrs:
                stem = cases_dir / f"{image_id}_snr{int(snr)}"
                if args.resume and stem.with_suffix(".json").exists():
                    item = pipeline.read_json(stem.with_suffix(".json"))
                    results.append(item["result"])
                    manifests.append(item["manifest"])
                    continue
                t0 = time.time()
                try:
                    out = run_case(baseline=baseline, secret_px=secret_px,
                                   class_name=class_name, image_id=image_id,
                                   token=tok["token"], wrong_token=tok["wrong_token"],
                                   snr_db=snr, split_role=role, hidden=hidden)
                except Exception as err:  # noqa: BLE001
                    failures.append({"image_id": image_id, "snr_db": snr, "error": repr(err)})
                    print(f"[phase15] FAIL {image_id} snr={snr}: {err}", flush=True)
                    continue
                recovered_path = cases_dir / f"{image_id}_snr{int(snr)}_recovered.png"
                received_path = cases_dir / f"{image_id}_snr{int(snr)}_received.png"
                pipeline.save_pixels(out["recovered"], recovered_path)
                pipeline.save_pixels(out["received"], received_path)
                result = {k: out[k] for k in ("image_id", "class", "snr_db", "role")}
                result.update({k: out[k] for k in METRICS})
                manifest = {k: out[k] for k in (
                    "image_id", "class", "snr_db", "role", "token_mode", "kpub_sha256",
                    "feature_kind", "feature_sha256", "seed", "latents_sha256",
                    "reference_sha256", "stego_sha256", "received_sha256",
                    "recovered_sha256", "enhanced", "channel")}
                manifest.update({"reference_path": out["reference_path"],
                                 "stego_path": str(stego_path),
                                 "recovered_path": str(recovered_path),
                                 "received_path": str(received_path)})
                torch.save({"stego": out["stego"], "delta": hidden["delta"],
                            "noise": hidden["noise"], "eta": hidden["eta"],
                            "steps": hidden["steps"], "mixing": hidden["mixing"],
                            "token": tok["token"], "wrong_token": tok["wrong_token"],
                            "prompt": baseline.conditions(image_id)["kpub"],
                            "secret_path": str(paths.PROJECT_ROOT / rec["path"])},
                           stem.with_suffix(".pt"))
                pipeline.write_json(stem.with_suffix(".json"),
                                    {"result": result, "manifest": manifest})
                results.append(result)
                manifests.append(manifest)
                print(f"[phase15] {role:11s} {class_name:6s} {image_id} snr={snr:>4.0f} "
                      f"psnr={out['psnr']:.2f} ({time.time() - t0:.1f}s)", flush=True)

    report = build_report("agentsemstecom", role, split, results,
                          approximate=approximate, config=cfg_all,
                          meta={"models": {"reference": cfg_all["reference"]["model_id"],
                                           "controlnet": cfg_all["reference"]["controlnet_id"],
                                           "edict": cfg_all["edict"],
                                           "codec": baseline.codec["kind"],
                                           "enhancement": ("enabled"
                                                           if baseline.enhancer else "disabled")}})
    for cell in report["cells"]:
        for name in METRICS:
            if name in cell["metrics"]:
                cell[name] = cell["metrics"][name]

    determinism = {"sample": 0, "reference_equal": True, "stego_equal": True,
                   "recovered_equal": True}
    for class_name in (classes[:1] if args.limit else classes):
        for rec in split.records(class_name)[:2]:
            image_id = rec["image_id"]
            tok = tokens[image_id]
            out = run_case(baseline=baseline, secret_px=pipeline.load_pixels(
                str(paths.PROJECT_ROOT / rec["path"]), int(baseline.stego_cfg["image_size"])),
                class_name=class_name, image_id=image_id, token=tok["token"],
                wrong_token=tok["wrong_token"], snr_db=snrs[-1], split_role=role)
            prior = next((m for m in manifests
                          if m["image_id"] == image_id and m["snr_db"] == int(snrs[-1])), None)
            determinism["sample"] += 1
            if prior:
                determinism["reference_equal"] &= out["reference_sha256"] == prior["reference_sha256"]
                determinism["stego_equal"] &= out["stego_sha256"] == prior["stego_sha256"]
                determinism["recovered_equal"] &= out["recovered_sha256"] == prior["recovered_sha256"]

    # Grid: one image per class, secret | stego | recovered 5 dB | recovered 20 dB.
    rows = []
    for class_name in classes:
        recs = split.records(class_name)
        if not recs:
            continue
        iid = recs[0]["image_id"]
        row = [str(paths.PROJECT_ROOT / recs[0]["path"]), str(cases_dir / f"{iid}_stego.png")]
        for snr in (min(snrs), max(snrs)):
            p = cases_dir / f"{iid}_snr{int(snr)}_recovered.png"
            row.append(str(p) if p.exists() else str(cases_dir / f"{iid}_stego.png"))
        rows.append(row)
    grid_path = paths.PROJECT_ROOT / (bcfg["calibration_grid"] if approximate
                                      else bcfg["evaluation_grid"])
    if rows:
        pipeline.grid(rows, grid_path, labels=("secret", "stego", "rec@5dB", "rec@20dB"))

    out_path = paths.PROJECT_ROOT / (bcfg["calibration_output"] if approximate
                                     else bcfg["evaluation_output"])
    manifest_path = paths.PROJECT_ROOT / (bcfg["calibration_manifest"] if approximate
                                          else bcfg["evaluation_manifest"])
    table = {
        **report,
        "code_commit": _code_commit(),
        "models": report["meta"]["models"],
        "determinism": determinism,
        "failures": failures,
        "enhancement_enabled": bool(baseline.enhancer),
    }
    write_report(table, out_path)
    payload = {
        "schema_version": 1, "phase": 15, "role": role, "approximate": approximate,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_version": split.version,
        "tokens_version": pipeline.read_json(paths.TOKENS_PATH)["tokens_version"],
        "config_hash": config_hash(cfg_all), "code_commit": table["code_commit"],
        "models": table["models"], "determinism": determinism,
        "cases": manifests, "failures": failures,
        "table_path": str(out_path), "table_sha256": hashlib.sha256(
            out_path.read_bytes()).hexdigest(),
        "grid_path": str(grid_path), "elapsed_s": time.time() - started,
    }
    pipeline.write_json(manifest_path, payload)
    print(f"[phase15] wrote {out_path} and {manifest_path} "
          f"({len(results)} cases, {len(failures)} failures, "
          f"{time.time() - started:.0f}s)", flush=True)
    baseline.edict.unload()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
