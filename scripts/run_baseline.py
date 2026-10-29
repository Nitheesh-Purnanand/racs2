"""Run the AgentSemSteCom baseline (Phases 1-16) and print the results.

This is the one command for the baseline. It runs each phase script with the
project virtual environment, stops on the first failure, rebuilds the final
report from the frozen artifacts and prints the result tables to the terminal
every time it finishes.

    experiments/results/final_baseline_report.json
    experiments/results/final_baseline_report.md
    experiments/results/final_baseline_table.csv
    experiments/results/baseline_run_log.json

Modes
-----
``current`` (default)
    Uses ``configs/base.yaml`` exactly as it is on disk.

``paper``
    Same as ``current`` but refuses to start if any paper asset referenced by
    the config (PicX-Real, IPAdapter-plus, SwinJSCC, CodeFormer) is missing.

``frozen``
    Temporarily swaps in the config committed at ``--frozen-ref`` (default
    HEAD) so an older baseline can be reproduced byte-for-byte, then restores
    the working-tree config.

Examples
--------
    # Show the latest results without computing anything
    .venv\Scripts\python.exe scripts\run_baseline.py --show

    # Full baseline with the paper components, assets checked first (hours)
    .venv\Scripts\python.exe scripts\run_baseline.py --mode paper --yes

    # Re-check every gate against the existing artifacts (minutes)
    .venv\Scripts\python.exe scripts\run_baseline.py --verify-only --yes

    # Resume the two expensive phases
    .venv\Scripts\python.exe scripts\run_baseline.py --from-phase 15 --to-phase 16 --yes

    # One image end to end, with every intermediate image saved
    .venv\Scripts\python.exe scripts\manual_test.py --mode paper
"""

from __future__ import annotations

import argparse
import contextlib
import os
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
PYTHON = sys.executable

PHASE_NAMES = {
    1: "environment and tensor contract",
    2: "VAE round-trip",
    3: "diffusion timing and img2img",
    4: "retrieval corpus (banked)",
    5: "CLIP embeddings (banked)",
    6: "splits, tokens and evaluation harness",
    7: "public conditions, Kpub + Xfeat",
    8: "reference generation",
    9: "ControlNet zero-conv fine-tune",
    10: "EDICT isolated invertibility",
    11: "local hide and recover",
    12: "AWGN channel",
    13: "semantic codec",
    14: "optional CodeFormer enhancement",
    15: "baseline gate, 15A + 15B",
    16: "baseline security and XuNet",
}

BANKED_PHASES = {4, 5}

FROZEN_ARTIFACTS = [
    "baseline_table.json",
    "baseline_manifest.json",
    "phase15A_calibration_table.json",
    "phase15A_manifest.json",
    "baseline_security.json",
    "baseline_steganalysis.json",
    "phase15_verify.json",
    "phase16_verify.json",
    "phase16_manifest.json",
    "phase7_verify.json",
    "phase8_verify.json",
    "phase8_ipadapter_embeds.json",
    "public_conditions/manifest.json",
    "phase8_references.json",
]


def log(message: str) -> None:
    print(f"[run] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_commands(phase: int, resume: bool) -> list[list[str]]:
    if phase == 1:
        return [["scripts/phase1_verify.py"]]
    if phase == 2:
        return [["scripts/phase2_vae.py"]]
    if phase == 3:
        return [["scripts/phase3_diffusion.py"], ["scripts/phase3_img2img.py"]]
    if phase == 4:
        return [["scripts/phase4_build_corpus.py"], ["scripts/phase4_verify.py"]]
    if phase == 5:
        return [["scripts/phase5_fetch_model.py"], ["scripts/phase5_clip.py"],
                ["scripts/phase5_verify.py"]]
    if phase == 6:
        return [
            ["scripts/phase6_build_testset.py"],
            ["scripts/phase6_eval_selftest.py"],
            ["scripts/eval_report.py", "--system", "dummy", "--split", "evaluation",
             "--runner", "scripts.dummy_runner:run_case",
             "--out", "experiments/results/phase6_dummy_eval.json"],
        ]
    if phase == 7:
        return [["scripts/phase7_build_conditions.py"], ["scripts/phase7_verify.py"]]
    if phase == 8:
        # --force: new references mean new file hashes, so the embedding cache
        # is rebuilt rather than left holding stale entries.
        return [["scripts/phase8_reference.py"],
                ["scripts/phase8_embeds.py", "--force"],
                ["scripts/phase8_verify.py"]]
    if phase == 9:
        return [["scripts/phase9_train_controlnet.py"], ["scripts/phase9_verify.py"]]
    if phase == 10:
        return [["scripts/phase10_edict.py"], ["scripts/phase10_verify.py"]]
    if phase == 11:
        return [["scripts/phase11_local.py"], ["scripts/phase11_verify.py"]]
    if phase == 12:
        return [["scripts/phase12_awgn.py"], ["scripts/phase12_verify.py"]]
    if phase == 13:
        return [["scripts/phase13_codec.py"], ["scripts/phase13_verify.py"]]
    if phase == 14:
        return [["scripts/phase14_enhance.py"], ["scripts/phase14_verify.py"]]
    if phase == 15:
        evaluation = ["scripts/phase15_baseline.py", "--split", "evaluation"]
        if resume:
            evaluation.append("--resume")
        return [
            ["scripts/phase15_baseline.py", "--split", "calibration"],
            evaluation,
            ["scripts/phase15_verify.py"],
        ]
    if phase == 16:
        return [
            ["scripts/phase16_security.py"],
            ["scripts/phase16_steganalysis.py", "--train"],
            ["scripts/phase16_verify.py"],
        ]
    return []


def verify_commands(phase: int) -> list[list[str]]:
    if phase == 1:
        return [["scripts/phase1_verify.py"]]
    if phase == 4:
        return [["scripts/phase4_verify.py"]]
    if phase == 5:
        return [["scripts/phase5_verify.py"]]
    if phase == 6:
        return [["scripts/phase6_eval_selftest.py"]]
    if phase == 7:
        return [["scripts/phase7_verify.py"]]
    if phase == 8:
        return [["scripts/phase8_verify.py"]]
    if phase == 9:
        return [["scripts/phase9_verify.py"]]
    if phase == 10:
        return [["scripts/phase10_verify.py"]]
    if phase == 11:
        return [["scripts/phase11_verify.py"]]
    if phase == 12:
        return [["scripts/phase12_verify.py"]]
    if phase == 13:
        return [["scripts/phase13_verify.py"]]
    if phase == 14:
        return [["scripts/phase14_verify.py"]]
    if phase == 15:
        return [["scripts/phase15_verify.py"]]
    if phase == 16:
        return [["scripts/phase16_verify.py"]]
    return []


def paper_asset_paths(config: dict) -> list[Path]:
    paths = []
    codec = config.get("codec", {})
    if codec.get("kind") == "swinjscc" and codec.get("checkpoint"):
        paths.append(PROJECT_ROOT / codec["checkpoint"])
    enhancement = config.get("enhancement", {})
    if enhancement.get("enabled"):
        paths.append(PROJECT_ROOT / enhancement.get("weights_dir", "models/codeformer")
                     / "codeformer.pth")
    ipc = (config.get("steganography", {}) or {}).get("ipadapter", {}) or {}
    if ipc.get("enabled"):
        paths.append(PROJECT_ROOT / ipc.get("root", "models/ip-adapter")
                     / ipc.get("subfolder", "models") / ipc.get("weight_name", ""))
    reference = config.get("reference", {})
    if reference.get("backend") == "picx_real" and reference.get("model_id"):
        paths.append(PROJECT_ROOT / reference["model_id"])
    return paths


def missing_assets(config: dict) -> list[Path]:
    return [p for p in paper_asset_paths(config) if not p.exists()]


def load_config_bytes(raw: bytes) -> dict:
    import yaml
    return yaml.safe_load(raw.decode("utf-8"))


def config_bytes_for_mode(mode: str, frozen_ref: str) -> tuple[bytes, str]:
    if mode == "frozen":
        raw = subprocess.check_output(
            ["git", "show", f"{frozen_ref}:configs/base.yaml"],
            cwd=PROJECT_ROOT,
        )
        return raw, f"git show {frozen_ref}:configs/base.yaml"
    return CONFIG_PATH.read_bytes(), "configs/base.yaml"


@contextlib.contextmanager
def config_swap(mode: str, frozen_ref: str):
    original = CONFIG_PATH.read_bytes()
    if mode != "frozen":
        yield
        return
    frozen, source = config_bytes_for_mode(mode, frozen_ref)
    backup = PROJECT_ROOT / "configs" / ".base.yaml.racs2_backup"
    backup.write_bytes(original)
    CONFIG_PATH.write_bytes(frozen)
    log(f"using frozen config from {source}")
    try:
        yield
    finally:
        CONFIG_PATH.write_bytes(original)
        if backup.exists():
            backup.unlink()
        log("restored working-tree configs/base.yaml")


def archive_frozen(reason: str) -> Path | None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "archive" / f"{stamp}_{reason}"
    copied = []
    for rel in FROZEN_ARTIFACTS:
        src = RESULTS_DIR / rel
        if src.exists():
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel)
    if not copied:
        return None
    (out_dir / "ARCHIVE.json").write_text(json.dumps({
        "created_utc": utc_now(),
        "reason": reason,
        "files": copied,
    }, indent=2), encoding="utf-8")
    return out_dir


def run_command(cmd: list[str], dry_run: bool, retries: int = 0) -> dict:
    # -X faulthandler: large checkpoint loads on this box occasionally die with
    # an access violation (exit 0xC0000005); the handler both stabilises the
    # load and prints where it died when it does not.
    full = [PYTHON, "-X", "faulthandler"] + cmd
    printable = " ".join(shlex.quote(part) for part in full)
    attempts = max(0, retries) + 1
    started = time.time()
    last = None
    for attempt in range(1, attempts + 1):
        if dry_run:
            log(f"DRY-RUN {printable}")
            return {"cmd": printable, "returncode": 0, "elapsed_s": 0.0,
                    "dry_run": True, "attempts": attempt}
        log(printable + (f"  (attempt {attempt}/{attempts})" if attempts > 1 else ""))
        attempt_started = time.time()
        # This box runs close to its Windows commit limit; expandable segments
        # keep the CUDA allocator from reserving more than a phase needs.
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        rc = subprocess.run(full, cwd=PROJECT_ROOT, env=env).returncode
        attempt_elapsed = time.time() - attempt_started
        log(f"returncode={rc} elapsed={attempt_elapsed:.1f}s")
        last = {"cmd": printable, "returncode": rc,
                "elapsed_s": round(attempt_elapsed, 1),
                "dry_run": False, "attempts": attempt}
        if rc == 0:
            last["elapsed_s"] = round(time.time() - started, 1)
            return last
        if attempt < attempts:
            # The usual cause is transient memory pressure, so wait long enough
            # for the previous process to be fully reaped.
            wait = min(60, 15 * attempt)
            log(f"retrying in {wait}s after returncode {rc}")
            time.sleep(wait)
    return last if last is not None else {
        "cmd": printable, "returncode": 1, "elapsed_s": 0.0,
        "dry_run": False, "attempts": 0}


def _cell_metric(cell: dict, name: str):
    metric = cell.get(name) or cell.get("metrics", {}).get(name)
    if isinstance(metric, dict):
        return metric.get("mean")
    return None


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def build_report() -> dict:
    table = read_json(RESULTS_DIR / "baseline_table.json")
    manifest = read_json(RESULTS_DIR / "baseline_manifest.json")
    security = read_json(RESULTS_DIR / "baseline_security.json")
    steg = read_json(RESULTS_DIR / "baseline_steganalysis.json")
    verify15 = read_json(RESULTS_DIR / "phase15_verify.json")
    verify16 = read_json(RESULTS_DIR / "phase16_verify.json")

    status = "incomplete"
    if table and security and steg and verify15 and verify16:
        status = "pass" if (verify15.get("pass") and verify16.get("pass")) else "fail"
    elif table:
        status = "partial"

    rows = []
    if table:
        for cell in table.get("cells", []):
            rows.append({
                "class": cell.get("class"),
                "snr_db": cell.get("snr_db"),
                "n": cell.get("n"),
                "mse": _cell_metric(cell, "mse"),
                "psnr": _cell_metric(cell, "psnr"),
                "ssim": _cell_metric(cell, "ssim"),
                "lpips": _cell_metric(cell, "lpips"),
            })

    aggregate = None
    if security:
        cells = security.get("cells", [])
        total = sum(c.get("n", 0) for c in cells)
        if total:
            def weighted(mode: str) -> float:
                return sum(c[mode]["psnr"]["mean"] * c.get("n", 0) for c in cells) / total
            correct = weighted("correct")
            wrong = weighted("wrong")
            no_token = weighted("no_token")
            aggregate = {
                "correct_psnr": correct,
                "wrong_psnr": wrong,
                "no_token_psnr": no_token,
                "correct_minus_wrong_db": correct - wrong,
                "correct_minus_no_token_db": correct - no_token,
                "n_cells": len(cells),
            }

    xu = None
    if steg:
        test = steg.get("results", {}).get("test", {})
        xu = {
            "test_accuracy": test.get("accuracy"),
            "test_n": test.get("n"),
            "chance_rate": test.get("chance_rate"),
            "paper_anchor": steg.get("paper_anchor", {}),
            "checkpoint_path": steg.get("checkpoint_path"),
            "checkpoint_sha256": steg.get("checkpoint_sha256"),
        }

    provenance = {}
    if manifest:
        provenance = {
            "split_version": manifest.get("split_version"),
            "tokens_version": manifest.get("tokens_version"),
            "config_hash": manifest.get("config_hash"),
            "code_commit": manifest.get("code_commit"),
            "baseline_created_utc": manifest.get("created_utc"),
            "models": manifest.get("models"),
            "determinism": manifest.get("determinism"),
        }
    if (RESULTS_DIR / "baseline_table.json").exists():
        provenance["baseline_table_sha256"] = sha256_file(
            RESULTS_DIR / "baseline_table.json")

    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "status": status,
        "baseline": {
            "system": table.get("system") if table else None,
            "role": table.get("role") if table else None,
            "approximate": table.get("approximate") if table else None,
            "cells": rows,
        },
        "security": aggregate,
        "security_paper_anchor": security.get("paper_anchor") if security else None,
        "gate_a": (verify15.get("gate_a") if verify15 else None) or [],
        "steganalysis": xu,
        "verification": {
            "phase15": {
                "present": verify15 is not None,
                "pass": verify15.get("pass") if verify15 else None,
                "gate_a_pass": verify15.get("gate_a_pass") if verify15 else None,
                "gate_b_pass": verify15.get("gate_b_pass") if verify15 else None,
                "summary": verify15.get("summary") if verify15 else None,
            },
            "phase16": {
                "present": verify16 is not None,
                "pass": verify16.get("pass") if verify16 else None,
                "summary": verify16.get("summary") if verify16 else None,
            },
        },
        "provenance": provenance,
    }


def write_report(report: dict) -> tuple[Path, Path, Path]:
    json_path = RESULTS_DIR / "final_baseline_report.json"
    md_path = RESULTS_DIR / "final_baseline_report.md"
    csv_path = RESULTS_DIR / "final_baseline_table.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "snr_db", "n", "mse", "psnr", "ssim", "lpips"])
        for row in report["baseline"]["cells"]:
            writer.writerow([
                row["class"], row["snr_db"], row["n"],
                _fmt(row["mse"], 6), _fmt(row["psnr"], 4),
                _fmt(row["ssim"], 4), _fmt(row["lpips"], 4),
            ])

    lines = [
        "# RACS2 Final Baseline Report",
        "",
        f"- Generated: {report['generated_utc']}",
        f"- Status: **{report['status']}**",
        f"- System: {report['baseline'].get('system')}",
        f"- Role: {report['baseline'].get('role')}",
        f"- Approximate: {report['baseline'].get('approximate')}",
        "",
        "## Baseline recovery quality",
        "",
        "| class | SNR dB | n | MSE | PSNR | SSIM | LPIPS |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for row in report["baseline"]["cells"]:
        lines.append(
            f"| {row['class']} | {row['snr_db']} | {row['n']} | "
            f"{_fmt(row['mse'], 6)} | {_fmt(row['psnr'], 4)} | "
            f"{_fmt(row['ssim'], 4)} | {_fmt(row['lpips'], 4)} |"
        )
    lines += ["", "## Token-condition security", ""]
    sec = report.get("security")
    if sec:
        lines += [
            "| condition | PSNR dB |",
            "| :-- | --: |",
            f"| correct | {_fmt(sec['correct_psnr'], 4)} |",
            f"| wrong | {_fmt(sec['wrong_psnr'], 4)} |",
            f"| no_token | {_fmt(sec['no_token_psnr'], 4)} |",
            "",
            f"- correct minus wrong: {_fmt(sec['correct_minus_wrong_db'], 4)} dB",
            f"- correct minus no_token: {_fmt(sec['correct_minus_no_token_db'], 4)} dB",
        ]
    else:
        lines.append("Security table missing.")
    lines += ["", "## XuNet steganalysis", ""]
    xu = report.get("steganalysis")
    if xu:
        lines += [
            f"- Test accuracy: {_fmt(xu.get('test_accuracy'), 4)} (n={xu.get('test_n')})",
            f"- Chance rate: {_fmt(xu.get('chance_rate'), 4)}",
            f"- Paper anchor: {xu.get('paper_anchor')}",
            f"- Checkpoint: {xu.get('checkpoint_path')}",
        ]
    else:
        lines.append("Steganalysis table missing.")
    lines += ["", "## Verification gates", ""]
    v15 = report["verification"]["phase15"]
    v16 = report["verification"]["phase16"]
    lines += [
        f"- Phase 15: pass={v15.get('pass')} gate_a={v15.get('gate_a_pass')} "
        f"gate_b={v15.get('gate_b_pass')} summary={v15.get('summary')}",
        f"- Phase 16: pass={v16.get('pass')} summary={v16.get('summary')}",
        "",
        "Gate A compares calibration cells to the paper's Table II anchors with "
        "tolerances. Gate B is the authoritative structural gate. Every Gate A miss "
        "is recorded in docs/DEVIATIONS.md.",
        "",
        "## Provenance",
        "",
    ]
    for key, value in report.get("provenance", {}).items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Artifacts",
        "",
        "- experiments/results/baseline_table.json",
        "- experiments/results/baseline_manifest.json",
        "- experiments/results/baseline_security.json",
        "- experiments/results/baseline_steganalysis.json",
        "- experiments/results/final_baseline_report.json",
        "- experiments/results/final_baseline_table.csv",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, csv_path


PAPER_ANCHORS_NOTE = (
    "paper Table II anchors are calibration targets only; every Gate A miss is "
    "recorded in docs/DEVIATIONS.md"
)


def _components(config: dict) -> list:
    """One line per stage saying which component the config selected."""
    sem = config.get("semantic", {})
    feats = sem.get("features", {})
    keygen = sem.get("keygen", {})
    ref = config.get("reference", {})
    stego = config.get("steganography", {})
    ipc = stego.get("ipadapter") or {}
    codec = config.get("codec", {})
    enh = config.get("enhancement", {})
    ip = (f"on (scale {ipc.get('scale', 1.0)}, {ipc.get('weight_name', '?')})"
          if ipc.get("enabled") else "off")
    codec_name = codec.get("kind", "?")
    if codec_name == "swinjscc":
        codec_name += f" C={codec.get('channel_number', '?')}"
    llm = keygen.get("llm_model_id", "-") if keygen.get("method") == "local_llm" else "-"
    return [
        f"Kpub           : {keygen.get('method', '?')} ({llm})",
        f"Xfeat          : common={feats.get('common', {}).get('method', '?')}"
        f"  facial={feats.get('facial', {}).get('method', '?')}"
        f"  style={feats.get('style', {}).get('method', '?')}",
        f"reference      : {ref.get('backend', '?')} ({ref.get('model_id', '?')}), "
        f"{ref.get('steps', '?')} steps at {ref.get('output_size', '?')} px",
        f"EDICT          : {stego.get('edict_steps', '?')} steps, "
        f"mixing {stego.get('edict_mixing', '?')}, "
        f"strength {stego.get('edict_strength', '?')}, eta {stego.get('eta', '?')}",
        f"IPAdapter-plus : {ip}",
        f"codec          : {codec_name} (trained {codec.get('trained_snr_db', '?')} dB)",
        f"CodeFormer     : {'on' if enh.get('enabled') else 'off'} "
        f"(fidelity {enh.get('fidelity_weight', '?')}, {enh.get('classes', [])})",
    ]


def print_console_report(report: dict, config=None) -> None:
    """Print the baseline results so a run never ends in silence."""
    bar = "=" * 78
    status = str(report.get("status", "unknown")).upper()
    print()
    print(bar)
    print("RACS2 - AgentSemSteCom baseline reproduction")
    print(f"status: {status}    generated: {report.get('generated_utc')}")
    print(bar)
    if config:
        for line in _components(config):
            print("  " + line)
        print("-" * 78)

    cells = report.get("baseline", {}).get("cells", [])
    if cells:
        print(f"recovery quality - {report['baseline'].get('role')} split "
              f"(approximate={report['baseline'].get('approximate')})")
        print(f"  {'class':8s} {'SNR':>4s} {'n':>3s} {'MSE':>10s} {'PSNR dB':>9s}"
              f" {'SSIM':>7s} {'LPIPS':>7s}")
        for row in cells:
            print(f"  {str(row['class']):8s} {row['snr_db']:>4} {row['n']:>3}"
                  f" {_fmt(row['mse'], 6):>10s} {_fmt(row['psnr'], 3):>9s}"
                  f" {_fmt(row['ssim'], 4):>7s} {_fmt(row['lpips'], 4):>7s}")
    else:
        print("recovery quality : no baseline_table.json yet")
    print("-" * 78)

    sec = report.get("security")
    if sec:
        print("token-condition security (aggregate recovery PSNR)")
        print(f"  correct {_fmt(sec['correct_psnr'], 2)} dB | "
              f"wrong {_fmt(sec['wrong_psnr'], 2)} dB | "
              f"no_token {_fmt(sec['no_token_psnr'], 2)} dB")
        print(f"  gap correct-wrong {_fmt(sec['correct_minus_wrong_db'], 2)} dB, "
              f"correct-no_token {_fmt(sec['correct_minus_no_token_db'], 2)} dB")
        if report.get("security_paper_anchor"):
            print(f"  paper anchor: {report['security_paper_anchor']}")
    else:
        print("token-condition security : missing")

    xu = report.get("steganalysis")
    if xu:
        print("XuNet steganalysis")
        print(f"  test accuracy {_fmt(xu.get('test_accuracy'), 4)} "
              f"(n={xu.get('test_n')}, chance {_fmt(xu.get('chance_rate'), 4)})")
        if xu.get("paper_anchor"):
            print(f"  paper anchor: {xu['paper_anchor']}")
    else:
        print("XuNet steganalysis : missing")

    print("-" * 78)
    gate_a = report.get("gate_a") or []
    if gate_a:
        print("Gate A - calibration cells vs paper Table II (approximate, never a claim)")
        print(f"  {'class':8s} {'SNR':>4s} {'metric':>7s} {'paper':>10s}"
              f" {'measured':>10s}  verdict")
        for row in gate_a:
            verdict = "within tol" if row.get("pass") else "MISS"
            print(f"  {str(row.get('class')):8s} {row.get('snr_db'):>4} "
                  f"{str(row.get('metric')):>7s} {_fmt(row.get('paper'), 4):>10s} "
                  f"{_fmt(row.get('measured'), 4):>10s}  {verdict}")
        print(f"  note: {PAPER_ANCHORS_NOTE}")
        print("-" * 78)

    v15 = report["verification"]["phase15"]
    v16 = report["verification"]["phase16"]
    print(f"gates: phase15 pass={v15.get('pass')} (gate_b={v15.get('gate_b_pass')}, "
          f"gate_a={v15.get('gate_a_pass')}) {v15.get('summary')}")
    print(f"       phase16 pass={v16.get('pass')} {v16.get('summary')}")
    prov = report.get("provenance", {})
    if prov:
        print(f"provenance: config_hash={str(prov.get('config_hash'))[:16]} "
              f"split={str(prov.get('split_version'))[:16]} "
              f"commit={str(prov.get('code_commit'))[:12]}")
        det = prov.get("determinism")
        if det:
            print(f"determinism re-run on {det.get('sample')} cases: "
                  f"reference={det.get('reference_equal')} "
                  f"stego={det.get('stego_equal')} "
                  f"recovered={det.get('recovered_equal')}")
    print("full report: experiments/results/final_baseline_report.md")
    print(bar)
    print()


def validate_range(args) -> bool:
    if not (1 <= args.from_phase <= 16 and 1 <= args.to_phase <= 16):
        print("[run] phases must be between 1 and 16")
        return False
    if args.from_phase > args.to_phase:
        print("[run] --from-phase must be <= --to-phase")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from-phase", type=int, default=1)
    parser.add_argument("--to-phase", type=int, default=16)
    parser.add_argument("--mode", default="current",
                        choices=("current", "paper", "frozen"),
                        help="current: configs/base.yaml as-is; paper: same but "
                             "require every paper asset; frozen: the config "
                             "committed at --frozen-ref")
    parser.add_argument("--frozen-ref", default="HEAD",
                        help="git ref used by --mode frozen")
    parser.add_argument("--include-banked", action="store_true",
                        help="run Phases 4 and 5 instead of skipping them")
    parser.add_argument("--verify-only", action="store_true",
                        help="run verification scripts only")
    parser.add_argument("--report-only", action="store_true",
                        help="skip all phases and rebuild the report from artifacts")
    parser.add_argument("--show", action="store_true",
                        help="print the latest results and exit, writing nothing")
    parser.add_argument("--resume", default="auto", choices=("auto", "yes", "no"),
                        help="Phase 15B --resume policy; auto resumes only in frozen mode")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failing command")
    parser.add_argument("--retries", type=int, default=0,
                        help="retry a failing command this many times (GPU crashes/OOM)")
    parser.add_argument("--no-archive", action="store_true",
                        help="do not archive existing frozen artifacts in non-frozen modes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    if not validate_range(args):
        return 2

    if args.show or args.report_only:
        report = build_report()
        try:
            config = load_config_bytes(CONFIG_PATH.read_bytes())
        except Exception:  # noqa: BLE001
            config = None
        if args.report_only:
            for path in write_report(report):
                log(f"wrote {path}")
        print_console_report(report, config)
        return 0 if report["status"] in ("pass", "partial") else 1

    config_raw, config_source = config_bytes_for_mode(args.mode, args.frozen_ref)
    config = load_config_bytes(config_raw)
    log("components selected by " + config_source + ":")
    for line in _components(config):
        log("  " + line)
    missing = missing_assets(config)
    if missing:
        log(f"{config_source} references {len(missing)} missing paper asset(s):")
        for path in missing:
            log(f"  MISSING {path}")
        log("fetch them with: scripts\\fetch_paper_assets.py "
            "--only picx_real,ipadapter,swinjscc,codeformer")
        if args.mode == "paper":
            return 2
        log("continuing anyway; phases that need them will fail")

    resume = args.resume == "yes" or (args.resume == "auto" and args.mode == "frozen")

    phases = list(range(args.from_phase, args.to_phase + 1))
    if not args.include_banked:
        phases = [p for p in phases if p not in BANKED_PHASES]

    commands = []
    for phase in phases:
        phase_cmds = verify_commands(phase) if args.verify_only else phase_commands(
            phase, resume)
        if not phase_cmds:
            log(f"phase {phase} has no command in this mode, skipping")
            continue
        commands.append((phase, phase_cmds))

    if not commands:
        log("nothing to run")
        return 2

    total_commands = sum(len(cmds) for _, cmds in commands)
    log(f"mode={args.mode} from={args.from_phase} to={args.to_phase} "
        f"commands={total_commands} resume={resume} "
        f"verify_only={args.verify_only}")

    if not args.dry_run and not args.yes:
        answer = input("This can run for several hours. Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            log("aborted")
            return 0

    run_started = time.time()
    log_entries = []
    failed = False
    archive_path = None

    if args.mode != "frozen" and args.to_phase >= 7 and not args.verify_only:
        log("WARNING: non-frozen mode rebuilds conditions/references and can overwrite "
            "frozen tables and per-case files")
        log("use --mode frozen to reproduce the committed baseline, or register the "
            "new results as a new baseline version in docs/DEVIATIONS.md")

    if (args.mode != "frozen" and args.to_phase >= 15 and not args.no_archive
            and not args.dry_run):
        archive_path = archive_frozen("before_nonfrozen_run")
        if archive_path:
            log(f"archived existing frozen artifacts to {archive_path}")

    try:
        with config_swap(args.mode, args.frozen_ref):
            log(f"config source: {config_source}")
            for phase, phase_cmds in commands:
                log(f"=== Phase {phase}: {PHASE_NAMES.get(phase, '')} ===")
                for cmd in phase_cmds:
                    entry = run_command(cmd, args.dry_run, args.retries)
                    entry["phase"] = phase
                    log_entries.append(entry)
                    if entry["returncode"] != 0:
                        failed = True
                        if not args.keep_going:
                            log(f"stopping after failed command: {entry['cmd']}")
                            break
                if failed and not args.keep_going:
                    break
    except KeyboardInterrupt:
        log("interrupted")
        failed = True
    except Exception as err:  # noqa: BLE001
        log(f"orchestrator error: {err!r}")
        failed = True

    run_finished = time.time()
    log_payload = {
        "schema_version": 1,
        "started_utc": datetime.fromtimestamp(run_started, timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": utc_now(),
        "mode": args.mode,
        "config_source": config_source,
        "from_phase": args.from_phase,
        "to_phase": args.to_phase,
        "verify_only": args.verify_only,
        "resume": resume,
        "retries": args.retries,
        "dry_run": args.dry_run,
        "archive_path": str(archive_path) if archive_path else None,
        "elapsed_s": round(run_finished - run_started, 1),
        "failed": failed,
        "entries": log_entries,
    }
    log_path = RESULTS_DIR / "baseline_run_log.json"
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    log(f"wrote {log_path}")

    if args.dry_run:
        log("dry run complete")
        return 0

    report = build_report()
    if failed:
        report["status"] = "fail"
    report["run"] = {
        "mode": args.mode,
        "config_source": config_source,
        "from_phase": args.from_phase,
        "to_phase": args.to_phase,
        "verify_only": args.verify_only,
        "failed": failed,
        "elapsed_s": round(run_finished - run_started, 1),
        "log_path": str(log_path),
        "archive_path": str(archive_path) if archive_path else None,
    }
    paths = write_report(report)
    log(f"final status={report['status']} cells={len(report['baseline']['cells'])}")
    for path in paths:
        log(f"wrote {path}")
    print_console_report(report, config)

    if failed:
        log("one or more commands failed; inspect the run log and report")
        return 1
    if report["status"] != "pass":
        log("run finished but final gates did not pass")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
