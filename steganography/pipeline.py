"""Phase 11 — shared helpers: config, probe records, VAE/EDICT latent I/O."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
import common.hf_lowmem  # noqa: E402

common.hf_lowmem.install()

from evaluation.metrics import distribution, lpips_metric, mse, psnr, ssim  # noqa: E402

SCALING = 0.18215


def load_config() -> dict:
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _index(records: List[dict], key: str = "image_id") -> Dict[str, dict]:
    return {str(rec[key]): rec for rec in records}


def probe_records(role: str, class_name: str, limit: int = 20) -> List[dict]:
    """Frozen Phase 8/7 records for one role and class, in manifest order."""
    refs = read_json(paths.RESULTS_DIR / "phase8_references.json")
    tokens = _index(read_json(paths.TOKENS_PATH)["records"])
    split = read_json(paths.TEST_SPLIT_PATH)["records"]
    ref_index = {(r["role"], r["class"], r["image_id"]): r for r in refs["records"]}
    out = []
    for rec in split:
        if rec["role"] != role or rec["class"] != class_name:
            continue
        ref = ref_index.get((role, class_name, rec["image_id"]))
        if ref is None or not Path(ref["reference_path"]).exists():
            raise FileNotFoundError(f"missing Phase 8 reference for {rec['image_id']}")
        cond_path = paths.RESULTS_DIR / "public_conditions" / f"{rec['image_id']}.npz"
        if not cond_path.exists():
            raise FileNotFoundError(f"missing Phase 7 conditions for {rec['image_id']}")
        with np.load(cond_path, allow_pickle=True) as cond:
            kpub = str(cond["kpub"])
        out.append({
            "image_id": rec["image_id"],
            "role": role,
            "class": class_name,
            "path": str(paths.PROJECT_ROOT / rec["path"]),
            "reference_path": str(paths.PROJECT_ROOT / ref["reference_path"]),
            "token": str(tokens[rec["image_id"]]["token"]),
            "wrong_token": str(tokens[rec["image_id"]]["wrong_token"]),
            "kpub": kpub,
        })
        if limit and len(out) >= int(limit):
            break
    if not out:
        raise RuntimeError(f"no probe records for role={role} class={class_name}")
    return out


def load_pixels(path: str, size: int) -> torch.Tensor:
    """RGB image -> float32 [1,3,size,size] in [0,1]."""
    image = Image.open(path).convert("RGB").resize((int(size), int(size)), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def encode(edict, pixels01: torch.Tensor) -> torch.Tensor:
    """[0,1] pixels -> scaled latent (float32)."""
    pixels = (pixels01.to(torch.float32) * 2.0 - 1.0).cpu()
    latent = edict.vae.encode(pixels).latent_dist.mode()
    return (latent * SCALING).to(edict.latent_dtype)


@torch.no_grad()
def decode(edict, latent: torch.Tensor) -> torch.Tensor:
    """scaled latent -> [0,1] pixels on CPU."""
    image = edict.vae.decode((latent.detach().float().cpu() / SCALING)).sample
    return ((image / 2.0 + 0.5).clamp(0.0, 1.0)).float().cpu()


def save_pixels(tensor01: torch.Tensor, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor01[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def metrics(a: torch.Tensor, b: torch.Tensor, cfg: dict) -> dict:
    """All four metrics; LPIPS runs on CPU so it never competes with the UNet."""
    ev = cfg.get("evaluation", {})
    out = {"mse": mse(a, b), "psnr": psnr(a, b), "ssim": ssim(a, b)}
    if ev.get("lpips_net"):
        out["lpips"] = lpips_metric(a, b, net=ev.get("lpips_net", "alex"), device="cpu")
    return out


def summarize(values: List[float]) -> dict:
    return distribution(values)


def tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(
        tensor.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def mean_pair(pair: List[torch.Tensor]) -> torch.Tensor:
    return (pair[0] + pair[1]) * 0.5 if len(pair) == 2 else pair[0]


def grid(paths_out: List[tuple], out_path: Path, labels=("secret", "reference", "stego", "recovered")) -> None:
    """Simple labelled contact sheet from PIL images."""
    from PIL import ImageDraw

    rows = len(paths_out)
    cols = len(paths_out[0])
    tiles = [[Image.open(p).convert("RGB") for p in row] for row in paths_out]
    w, h = tiles[0][0].size
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(tiles):
        for c, img in enumerate(row):
            sheet.paste(img.resize((w, h), Image.BICUBIC), (c * w, r * h))
    for c, label in enumerate(labels[:cols]):
        draw.text((c * w + 4, 2), label, fill="black")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def forbidden_import_hits(files: List[Path], words) -> List[str]:
    hits = []
    for file in files:
        for line in Path(file).read_text(encoding="utf-8").splitlines():
            low = line.strip().lower()
            if low.startswith(("import ", "from ")) and any(w in low for w in words):
                hits.append(f"{Path(file).name}: {line.strip()}")
    return hits
