"""Phase 16 — XuNet steganalysis baseline."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

XUNET_ARCH = ("SRM 5x5 high-pass init, conv3-8 5x5 + BN + Tanh + avgpool5/2, "
              "conv8-16 5x5 + BN + Tanh + avgpool5/2, conv16-32 1x1 + BN + Tanh + avgpool5/2, "
              "conv32-64 1x1 + BN + Tanh + avgpool5/2, conv64-128 1x1 + BN + Tanh + "
              "global average pool, FC 128->2")


class XuNet(nn.Module):
    """Faithful XuNet: high-pass SRM front end, five conv blocks, global pool, 2-way FC."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, padding=2, bias=False), nn.BatchNorm2d(8), nn.Tanh(),
            nn.AvgPool2d(5, 2, 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, padding=2, bias=False), nn.BatchNorm2d(16), nn.Tanh(),
            nn.AvgPool2d(5, 2, 2))
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32), nn.Tanh(),
            nn.AvgPool2d(5, 2, 2))
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.Tanh(),
            nn.AvgPool2d(5, 2, 2))
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.Tanh(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, num_classes)
        self._init_srm()

    def _init_srm(self) -> None:
        kernel = torch.zeros(5, 5)
        kernel[2, 2] = -1.0
        kernel[2, 1] = kernel[2, 3] = kernel[1, 2] = kernel[3, 2] = 0.25
        with torch.no_grad():
            self.block1[0].weight.copy_(kernel.expand(8, 3, 5, 5).contiguous())

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x).flatten(1)
        return self.fc(x)


def _load_patches(path: str, size: int, grid: int) -> np.ndarray:
    """Deterministic non-overlapping patches, shape (grid*grid, 3, size, size)."""
    arr = np.asarray(
        Image.open(path).convert("RGB").resize((size * grid, size * grid), Image.BICUBIC),
        dtype=np.float32) / 255.0
    patches = [
        arr[iy * size:(iy + 1) * size, ix * size:(ix + 1) * size].transpose(2, 0, 1)
        for iy in range(grid) for ix in range(grid)
    ]
    return np.stack(patches)


def build_dataset(baseline_manifest: dict, baseline_table: dict, config: dict) -> dict:
    """Cover/stego pairs at one SNR, split 70/15/15 by image_id (no leakage)."""
    snr = int(config.get("snr_db", 10))
    cases = [c for c in baseline_manifest["cases"]
             if c["snr_db"] == snr and c["role"] == "evaluation"]
    cases.sort(key=lambda c: c["image_id"])
    if not cases:
        raise RuntimeError(f"no baseline cases at {snr} dB")
    size = int(config["input_size"])
    grid = int(config.get("patch_grid", 1))
    n_patches = grid * grid
    covers, stegos, ids = [], [], []
    for case in cases:
        blob = torch.load(Path(case["stego_path"]).parent /
                          f"{case['image_id']}_snr{snr}.pt",
                          map_location="cpu", weights_only=False)
        covers.append(_load_patches(blob["secret_path"], size, grid))
        stegos.append(_load_patches(case["stego_path"], size, grid))
        ids.append(case["image_id"])
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
    n = len(order)
    n_train = int(round(n * float(config["train_fraction"])))
    n_val = int(round(n * float(config["val_fraction"])))
    parts = {"train": order[:n_train], "val": order[n_train:n_train + n_val],
             "test": order[n_train + n_val:]}
    out = {"snr_db": snr, "n_images": n, "class_balance": {"cover": n, "stego": n},
           "patch_grid": grid, "n_patches_per_image": n_patches,
           "split_hashes": {}, "parts": {}}
    for name, idx in parts.items():
        cover_px = np.stack([covers[i] for i in idx]).reshape(-1, 3, size, size)
        stego_px = np.stack([stegos[i] for i in idx]).reshape(-1, 3, size, size)
        xs = np.concatenate([cover_px, stego_px])
        ys = np.array([0] * (len(idx) * n_patches) + [1] * (len(idx) * n_patches),
                      dtype=np.int64)
        out["parts"][name] = (torch.from_numpy(xs), torch.from_numpy(ys))
        out["split_hashes"][name] = hashlib.sha256(
            ",".join(ids[i] for i in idx).encode()).hexdigest()
        out[name + "_ids"] = [ids[i] for i in idx]
    return out


def train_xunet(dataset: dict, config: dict, out_dir: Path, seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XuNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    batch = int(config["batch_size"])
    x_tr, y_tr = dataset["parts"]["train"]
    x_va, y_va = dataset["parts"]["val"]
    best_acc, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        perm = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(seed + epoch))
        total = 0.0
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            xb = x_tr[idx].to(device) * 2.0 - 1.0
            yb = y_tr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.inference_mode():
            pred = model(x_va.to(device) * 2.0 - 1.0).argmax(1).cpu()
        acc = float((pred == y_va).float().mean())
        history.append({"epoch": epoch, "loss": total / max(1, len(perm)), "val_accuracy": acc})
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 10:
            break
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"xunet_baseline_{seed:08x}.pt"
    torch.save({"state_dict": best_state, "arch": XUNET_ARCH,
                "config": config, "best_epoch": best_epoch}, ckpt)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    return {"model": model, "checkpoint": str(ckpt), "sha256": sha,
            "best_epoch": best_epoch, "best_val_accuracy": best_acc, "history": history}


def evaluate(model, dataset: dict, split: str = "test") -> dict:
    x, y = dataset["parts"][split]
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        pred = model(x.to(device) * 2.0 - 1.0).argmax(1).cpu()
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"split": split, "n": int(len(y)), "accuracy": accuracy,
            "precision": precision, "recall": recall,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "chance_rate": 0.5}
