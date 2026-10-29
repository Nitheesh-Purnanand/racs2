"""Phase 15 — frozen AgentSemSteCom baseline pipeline (one function, full path)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
import common.hf_lowmem  # noqa: E402

common.hf_lowmem.install()

from communication.semantic_codec import build_codec, codec_roundtrip  # noqa: E402
from enhancement.codeformer import build_enhancer, enhance, should_enhance  # noqa: E402
from steganography import pipeline  # noqa: E402
from steganography.edict import EDICT  # noqa: E402
from steganography.hide import hide  # noqa: E402
from steganography.mask import token_delta, token_noise  # noqa: E402
from steganography.recover import recover  # noqa: E402
from steganography.ipadapter import (  # noqa: E402
    build_ipadapter,
    reference_embeds,
    reference_embeds_for_path,
)


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def channel_seed(role: str, class_name: str, snr_db: float) -> int:
    digest = hashlib.sha256(f"phase12|{role}|{class_name}|{int(snr_db)}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class Baseline:
    """Loads every frozen component once; exposes hide/transmit/receive."""

    def __init__(self, cfg_all: dict):
        self.cfg_all = cfg_all
        self.stego_cfg = dict(cfg_all["steganography"])
        self.stego_cfg["evaluation"] = cfg_all.get("evaluation", {})
        self.codec_cfg = dict(cfg_all["codec"])
        self.enh_cfg = dict(cfg_all["enhancement"])
        self.edict = EDICT(latent_dtype=torch.float32)
        self.ip_cfg = dict(self.stego_cfg.get("ipadapter", {}) or {})
        self.ipadapter = None
        if self.ip_cfg.get("enabled"):
            self.ipadapter = build_ipadapter(
                self.edict.unet,
                root=self.ip_cfg.get("root", "models/ip-adapter"),
                subfolder=self.ip_cfg.get("subfolder", "models"),
                weight_name=self.ip_cfg.get(
                    "weight_name", "ip-adapter-plus_sd15.safetensors"
                ),
                image_encoder_folder=self.ip_cfg.get(
                    "image_encoder_folder", "models/image_encoder"
                ),
                scale=float(self.ip_cfg.get("scale", 1.0)),
                device=self.edict.device,
                dtype=torch.float16,
                encoder_device=self.ip_cfg.get("encoder_device", "cpu"),
                encoder_dtype=torch.float16 if str(
                    self.ip_cfg.get("encoder_dtype", "float32")
                ).lower() in ("float16", "fp16") else torch.float32,
                cache_dir=self.ip_cfg.get(
                    "cache_dir", "experiments/results/ipadapter_embeds"
                ),
            )
        self.codec = build_codec(self.codec_cfg, self.edict)
        self.enhancer = build_enhancer(self.enh_cfg) if self.enh_cfg.get("enabled") else None
        self._refs = {(r["role"], r["class"], r["image_id"]): r
                      for r in pipeline.read_json(
                          paths.RESULTS_DIR / "phase8_references.json")["records"]}
        self._tokens = {r["image_id"]: r for r in
                        pipeline.read_json(paths.TOKENS_PATH)["records"]}
        self._cond = {}

    def conditions(self, image_id: str) -> dict:
        if image_id not in self._cond:
            with np.load(paths.RESULTS_DIR / "public_conditions" / f"{image_id}.npz",
                         allow_pickle=True) as npz:
                feature = (npz["feature"] if "feature" in npz.files
                           else np.zeros(0, dtype=np.float32))
                self._cond[image_id] = {
                    "kpub": str(npz["kpub"]),
                    "feature_kind": str(npz["feature_kind"]),
                    "feature_sha256": hashlib.sha256(
                        np.ascontiguousarray(feature)).hexdigest(),
                    "kpub_sha256": hashlib.sha256(str(npz["kpub"]).encode()).hexdigest(),
                }
        return self._cond[image_id]

    def reference_path(self, role: str, class_name: str, image_id: str) -> str:
        rec = self._refs[(role, class_name, image_id)]
        return str(paths.PROJECT_ROOT / rec["reference_path"])

    def tokens(self, image_id: str) -> dict:
        return self._tokens[image_id]

    def hide(self, secret_px, reference_px, prompt, token, reference_path=None):
        # Prefer the path form: the receiver reads the same cached tensor, so
        # Tx and Rx condition on byte-identical embeddings.
        embeds = (self.embeds_from_path(reference_path) if reference_path
                  else self.embeds_from_pixels(reference_px))
        out = hide(self.edict, secret_px, reference_px, prompt, token,
                   self.stego_cfg, image_embeds=embeds)
        out["image_embeds"] = embeds
        return out

    def embeds_from_pixels(self, reference_px):
        if self.ipadapter is None:
            return None
        import numpy as np
        from PIL import Image

        arr = (reference_px[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0)
        # round, never truncate: the receiver re-reads the same PNG and both
        # ends must hand CLIP byte-identical pixels.
        image = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype("uint8"))
        return reference_embeds(
            self.ipadapter, image,
            guidance_scale=float(self.stego_cfg.get("guidance_scale", 1.0)),
        )

    def embeds_from_path(self, reference_path):
        if self.ipadapter is None:
            return None
        return reference_embeds_for_path(
            self.ipadapter, reference_path,
            guidance_scale=float(self.stego_cfg.get("guidance_scale", 1.0)),
        )

    def transmit(self, stego, *, role, class_name, snr_db):
        return codec_roundtrip(stego, snr_db=snr_db,
                               seed=channel_seed(role, class_name, snr_db),
                               codec=self.codec, config=self.codec_cfg)

    def receive(self, received, *, token_mode, token, wrong_token, prompt, blob,
                image_embeds=None):
        cfg = self.stego_cfg
        if token_mode == "correct":
            used, delta, noise = token, blob["delta"], blob["noise"]
        elif token_mode == "wrong":
            used = wrong_token
            delta = token_delta(used, shape=tuple(blob["delta"].shape),
                                fraction=float(cfg["mask_fraction"]))
            noise = token_noise(used, shape=tuple(blob["noise"].shape))
        elif token_mode == "no_token":
            used = "00000000"
            delta = torch.zeros_like(blob["delta"])
            noise = token_noise(used, shape=tuple(blob["noise"].shape))
        else:
            raise ValueError(f"unknown token_mode {token_mode!r}")
        return recover(self.edict, received, used, prompt, cfg, delta=delta,
                       eta=blob["eta"], steps=blob["steps"], mixing=blob["mixing"],
                       guidance=cfg.get("guidance_scale", 1.0), noise=noise,
                       image_embeds=image_embeds)

    def maybe_enhance(self, recovered_px, class_name):
        if self.enhancer is None or not should_enhance(self.enh_cfg, class_name):
            return recovered_px, False
        from PIL import Image
        img = Image.fromarray((recovered_px[0].permute(1, 2, 0).numpy() * 255).round()
                              .astype("uint8"))
        out = enhance(img, enhancer=self.enhancer, config=self.enh_cfg)
        arr = np.asarray(out.resize(secret_size(img), Image.BICUBIC),
                         dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0), True


def secret_size(img) -> tuple:
    return img.size


def run_case(*, baseline: Baseline, secret_px, class_name: str, image_id: str,
             token: str, wrong_token: str, snr_db: float, split_role: str,
             token_mode: str = "correct", hidden: dict | None = None) -> dict:
    """One full baseline case: VAE -> EDICT -> mask -> codec/AWGN -> inverse -> metrics."""
    cond = baseline.conditions(image_id)
    prompt = cond["kpub"]
    ref_px = pipeline.load_pixels(baseline.reference_path(split_role, class_name, image_id),
                                  int(baseline.stego_cfg["image_size"]))
    if hidden is None:
        hidden = baseline.hide(
            secret_px, ref_px, prompt, token,
            reference_path=baseline.reference_path(split_role, class_name, image_id),
        )
    stego = hidden["stego_pixels"]
    image_embeds = hidden.get("image_embeds")
    tx = baseline.transmit(stego, role=split_role, class_name=class_name, snr_db=snr_db)
    received = tx["output"]
    blob = {"delta": hidden["delta"], "noise": hidden["noise"], "eta": hidden["eta"],
            "steps": hidden["steps"], "mixing": hidden["mixing"]}
    recovered = baseline.receive(received, token_mode=token_mode, token=token,
                                 wrong_token=wrong_token, prompt=prompt, blob=blob,
                                 image_embeds=image_embeds)
    enhanced = False
    if token_mode == "correct":
        recovered, enhanced = baseline.maybe_enhance(recovered, class_name)
    metrics = pipeline.metrics(secret_px, recovered, baseline.stego_cfg)
    return {
        "image_id": image_id, "class": class_name, "snr_db": int(snr_db),
        "role": split_role, "token_mode": token_mode, "approximate": False,
        "kpub": prompt, "kpub_sha256": cond["kpub_sha256"],
        "feature_kind": cond["feature_kind"], "feature_sha256": cond["feature_sha256"],
        "seed": hidden["mask"]["mask_seed"], "latents_sha256": {
            "secret": tensor_hash(hidden["z_secret"]),
            "hidden": tensor_hash(pipeline.mean_pair(hidden["pair"])),
            "stego": tensor_hash(pipeline.mean_pair(hidden["z_stego_pair"])),
        },
        "reference_path": baseline.reference_path(split_role, class_name, image_id),
        "reference_sha256": hashlib.sha256(Path(
            baseline.reference_path(split_role, class_name, image_id)).read_bytes()).hexdigest(),
        "stego": stego, "stego_sha256": tensor_hash(stego),
        "received": received, "received_sha256": tensor_hash(received),
        "recovered": recovered, "recovered_sha256": tensor_hash(recovered),
        "enhanced": bool(enhanced),
        "channel": tx["meta"],
        **metrics,
    }
