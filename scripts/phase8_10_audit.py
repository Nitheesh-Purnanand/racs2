"""Cross-phase audit for Phases 8-10: requirements -> checks, whole-result coverage."""

from __future__ import annotations

import hashlib
import json
import py_compile
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths
import common.hf_lowmem

common.hf_lowmem.install()

import torch
import yaml

from evaluation.splits import CLASSES, Split, load_tokens, sha256_file
from reference.seed import derive_seed
from semantic.cache import feature_sha256, read_cache_npz, sha256_string
from steganography.edict import EDICT, _inv_mix, _mix, forward_step, reverse_step

CHECKS = []


def add(requirement, check, passed, detail=""):
    CHECKS.append({
        "requirement": requirement,
        "check": check,
        "pass": bool(passed),
        "detail": str(detail),
    })
    print(f"[{'PASS' if passed else 'FAIL'}] {requirement} :: {check} :: {detail}")


def audit_imports():
    files = [
        "reference/__init__.py", "reference/seed.py", "reference/conditions.py",
        "reference/generate.py", "reference/train_controlnet.py",
        "steganography/__init__.py", "steganography/edict.py",
        "scripts/phase8_reference.py", "scripts/phase8_verify.py",
        "scripts/phase9_train_controlnet.py", "scripts/phase9_verify.py",
        "scripts/phase10_edict.py", "scripts/phase10_verify.py",
    ]
    bad = []
    for rel in files:
        try:
            py_compile.compile(str(paths.PROJECT_ROOT / rel), doraise=True)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{rel}: {exc}")
    add("packaging", "all new modules compile", not bad, "; ".join(bad) or f"{len(files)} files")


def audit_phase8():
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh).get("reference", {})
    manifest_path = paths.PROJECT_ROOT / cfg.get(
        "manifest_path", "experiments/results/phase8_references.json"
    )
    if not manifest_path.exists():
        add("phase8", "manifest exists", False, str(manifest_path))
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    add("phase8", "manifest schema", manifest.get("schema_version") == 1)
    add("phase8", "record count", len(records) == 132, str(len(records)))
    counts = defaultdict(int)
    for rec in records:
        counts[(rec["role"], rec["class"])] += 1
    add("phase8", "22/class/role", all(counts[k] == 22 for k in counts) and len(counts) == 6,
        json.dumps({f"{r}/{c}": n for (r, c), n in sorted(counts.items())}))

    tokens, _ = load_tokens()
    split = {}
    for role in ("calibration", "evaluation"):
        s = Split.load(role=role, allow_calibration=(role == "calibration"), verify=False)
        split[role] = {rec["image_id"]: rec for rec in s.records()}

    file_ok = seed_ok = latents_ok = link_ok = kind_ok = budget_ok = True
    seen_paths = set()
    size = int(manifest.get("output_size", 256))
    for rec in records:
        path = paths.PROJECT_ROOT / rec["reference_path"]
        if not path.exists() or sha256_file(path) != rec["reference_sha256"]:
            file_ok = False
        if rec["reference_path"] in seen_paths:
            file_ok = False
        seen_paths.add(rec["reference_path"])
        token = tokens[rec["image_id"]]["token" if rec["token_mode"] == "correct" else "wrong_token"]
        if derive_seed(token) != rec["seed"]:
            seed_ok = False
        gen = torch.Generator(device="cuda").manual_seed(rec["seed"])
        probe = torch.randn((1, 4, size // 8, size // 8), generator=gen,
                            device="cuda", dtype=torch.float16)
        digest = hashlib.sha256(probe.detach().float().cpu().numpy().tobytes()).hexdigest()
        if digest != rec["latents_sha256"]:
            latents_ok = False
        cond = read_cache_npz(paths.RESULTS_DIR / "public_conditions" / f"{rec['image_id']}.npz")
        if (sha256_string(cond["kpub"]) != rec["kpub_sha256"]
                or feature_sha256(cond.get("feature")) != rec["feature_sha256"]):
            link_ok = False
        expected = {"common": "segmentation", "facial": "skeleton", "style": "none"}[rec["class"]]
        if cond["feature_kind"] != expected:
            kind_ok = False
        if rec.get("peak_vram_gb") is not None and rec["peak_vram_gb"] > 5.5:
            budget_ok = False
    add("phase8", "all 132 reference files + unique paths", file_ok)
    add("phase8", "all 132 seeds recompute", seed_ok)
    add("phase8", "all 132 latents hashes recompute", latents_ok)
    add("phase8", "all 132 kpub/feature hashes link to Phase 7", link_ok)
    add("phase8", "class feature kinds correct", kind_ok)
    add("phase8", "all peak VRAM <= 5.5 GB", budget_ok)
    add("phase8", "verify json pass", json.loads(
        (paths.RESULTS_DIR / "phase8_verify.json").read_text()).get("pass") is True)


def audit_phase9():
    training = json.loads((paths.RESULTS_DIR / "phase9_training.json").read_text())
    train_cfg = yaml.safe_load(
        (paths.PROJECT_ROOT / "configs" / "base.yaml").read_text()
    ).get("controlnet_train", {})
    add("phase9", "calibration-only training", training.get("split_role") == "calibration")
    add("phase9", "zero-conv parameter split", training.get("trainable_parameters", 0) > 0
        and training.get("frozen_parameters", 0) > 0,
        f"{training.get('trainable_parameters')} / {training.get('frozen_parameters')}")
    add("phase9", "loss finite and decreased",
        training.get("loss_last", 1) < training.get("loss_first", 0)
        and training.get("loss_min", -1) >= 0,
        f"{training.get('loss_first')} -> {training.get('loss_last')}")
    add("phase9", "probe improved", training.get("probe", {}).get("improved") is True,
        f"{training.get('probe', {}).get('zero_shot_mse')} -> "
        f"{training.get('probe', {}).get('fine_tuned_mse')}")
    ckpt = paths.PROJECT_ROOT / training["checkpoint_path"]
    add("phase9", "checkpoint hash", ckpt.exists()
        and sha256_file(ckpt) == training.get("checkpoint_sha256"), str(ckpt))
    from safetensors.torch import load_file
    state = load_file(str(ckpt))
    keys_ok = bool(state) and all(
        k.startswith("controlnet_down_blocks") or k.startswith("controlnet_mid_block")
        for k in state
    )
    add("phase9", "checkpoint contains only zero-conv tensors", keys_ok,
        f"{len(state)} tensors, {sum(v.numel() for v in state.values())} params")
    add("phase9", "checkpoint param count matches json",
        sum(v.numel() for v in state.values()) == training.get("trainable_parameters"))
    from reference.train_controlnet import load_finetuned_controlnet
    _, info = load_finetuned_controlnet(
        train_cfg.get("controlnet_id", "lllyasviel/control_v11p_sd15_seg"),
        ckpt, dtype=torch.bfloat16, return_info=True,
    )
    add("phase9", "delta applies to base ControlNet",
        info["missing_keys"] == 0 and info["unexpected_keys"] == 0
        and info["changed_tensors"] == info["checkpoint_tensors"],
        json.dumps(info))
    add("phase9", "verify json pass", json.loads(
        (paths.RESULTS_DIR / "phase9_verify.json").read_text()).get("pass") is True)


def audit_phase10():
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh).get("edict", {})
    doc = json.loads((paths.RESULTS_DIR / "phase10_edict.json").read_text())
    op = doc.get("operating_point_result") or {}
    add("phase10", "20 random + 20 image latents",
        op.get("n_random", 0) >= 20 and op.get("n_image", 0) >= 20,
        f"{op.get('n_random')} / {op.get('n_image')}")
    add("phase10", "operating point matches config",
        doc.get("operating_point") == {"steps": cfg.get("steps"), "mixing": cfg.get("mixing")})
    add("phase10", "relative error below measured fp16 gate",
        op.get("max_rel_error", 1) <= cfg.get("max_rel_error", 5e-3),
        f"{op.get('max_rel_error'):.3e} <= {cfg.get('max_rel_error')}")
    add("phase10", "sweep has 12 cells", len(doc.get("cells", [])) == 12,
        str(len(doc.get("cells", []))))
    add("phase10", "verify json pass", json.loads(
        (paths.RESULTS_DIR / "phase10_verify.json").read_text()).get("pass") is True)

    # Algebra: mixing and DDIM steps must be exact inverses without the model.
    torch.manual_seed(0)
    a = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    b = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    mixed = _mix([a, b], 0.93)
    back = _inv_mix(mixed, 0.93)
    mix_err = max(float((back[0] - a).abs().max()), float((back[1] - b).abs().max()))
    add("phase10", "affine mixing exact inverse", mix_err <= 1e-12, f"max_abs={mix_err:.2e}")

    from diffusers import DDIMScheduler
    sch = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                        num_train_timesteps=1000, clip_sample=False, set_alpha_to_one=False)
    sch.set_timesteps(20)
    t = sch.timesteps[5]
    sample = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    model_out = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    fwd = forward_step(sch, model_out, t, sample, 20)
    rev = reverse_step(sch, model_out, t, fwd, 20)
    step_err = float((rev - sample).abs().max())
    add("phase10", "DDIM forward/reverse exact inverse", step_err <= 1e-5,
        f"max_abs={step_err:.2e}")

    # Public edge cases fail fast rather than writing NaNs.
    dummy = object.__new__(EDICT)
    for label, kwargs in [
        ("mixing=0.5 rejected", dict(mixing=0.5, steps=20)),
        ("steps=0 rejected", dict(mixing=0.93, steps=0)),
    ]:
        try:
            EDICT.forward(dummy, torch.zeros(1, 4, 8, 8), prompt="x", **kwargs)
            add("phase10", label, False, "no exception")
        except ValueError:
            add("phase10", label, True)
        except Exception as exc:  # noqa: BLE001
            add("phase10", label, False, repr(exc))

    # Same-process determinism on a small public roundtrip.
    try:
        edict = EDICT()
        z = torch.randn(1, 4, 8, 8, device="cuda")
        r1 = edict.roundtrip(z, prompt="a photo", steps=4, mixing=0.93, guidance_scale=1.0)
        r2 = edict.roundtrip(z, prompt="a photo", steps=4, mixing=0.93, guidance_scale=1.0)
        det = max(float((r1["pair"][i] - r2["pair"][i]).abs().max()) for i in range(2))
        add("phase10", "EDICT rerun deterministic", det == 0.0, f"max_abs_diff={det:.2e}")
        edict.unload()
    except Exception as exc:  # noqa: BLE001
        add("phase10", "EDICT rerun deterministic", False, repr(exc))


def main():
    audit_imports()
    audit_phase8()
    audit_phase9()
    audit_phase10()
    failures = [c for c in CHECKS if not c["pass"]]
    result = {
        "pass": not failures,
        "schema_version": 1,
        "checks": CHECKS,
        "failures": [f"{c['requirement']}::{c['check']}" for c in failures],
    }
    out = paths.RESULTS_DIR / "phase8_10_audit.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nAUDIT RESULT: {'PASS' if result['pass'] else 'FAIL'} "
          f"({len(CHECKS) - len(failures)}/{len(CHECKS)}) -> {out}")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
