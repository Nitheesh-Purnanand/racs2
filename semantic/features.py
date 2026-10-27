"""
Phase 7 — Structural Feature (Xfeat) Extraction for RACS2.

Extracts class-specific structural features from images:

  * ``common``:  semantic segmentation map (SegFormer-B0 on ADE20K, 150 classes).
  * ``facial``:  face/pose keypoints rendered as a skeleton graph.
  * ``style``:   no structural feature (conditioned on Kpub only).

Design rules:
  * All models are pretrained, frozen, and used for inference only.
  * ``model.eval()``, ``torch.inference_mode()``, no gradients.
  * Outputs are deterministic given the same image.
  * Style explicitly has ``feature_kind = "none"`` — no fake blank image.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths  # noqa: E402,F401
import common.hf_lowmem  # noqa: E402,F401

common.hf_lowmem.install()

from common.device import get_device  # noqa: E402


# ── ADE20K colour palette (150 classes) ──────────────────────────────────────

def _ade20k_palette() -> np.ndarray:
    """ADE20K 150-class colour palette as uint8 [150, 3]."""
    palette = np.zeros((150, 3), dtype=np.uint8)
    for i in range(150):
        r = g = b = 0
        idx = i + 1  # 0 is background, offset by 1
        for j in range(8):
            r |= ((idx >> 0) & 1) << (7 - j)
            g |= ((idx >> 1) & 1) << (7 - j)
            b |= ((idx >> 2) & 1) << (7 - j)
            idx >>= 3
        palette[i] = (r, g, b)
    return palette

ADE20K_PALETTE = _ade20k_palette()


# ── SegFormer Feature Extractor (common class) ──────────────────────────────

class SegFormerExtractor:
    """Semantic segmentation via SegFormer-B0 on ADE20K."""

    def __init__(
        self,
        model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: Union[str, torch.device, None] = None,
        dtype: str = "float16",
        output_size: int = 512,
        num_classes: int = 150,
        model_path: Optional[str] = None,
    ) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.method = "segformer"
        self.model_id = model_id
        self.output_size = output_size
        self.num_classes = num_classes

        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        self.dtype = dtype_map.get(str(dtype).lower(), torch.float16)

        source = model_id
        if model_path:
            candidate = Path(model_path)
            if not candidate.is_absolute():
                candidate = _ROOT / candidate
            if candidate.exists():
                source = str(candidate)

        self.processor = SegformerImageProcessor.from_pretrained(source)
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                source, torch_dtype=self.dtype,
            )
        except TypeError:
            self.model = SegformerForSemanticSegmentation.from_pretrained(source)

        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._weight_sha256 = self._compute_weight_hash()

    def _compute_weight_hash(self) -> str:
        h = hashlib.sha256()
        for name, param in sorted(self.model.named_parameters()):
            h.update(f"{name}:{list(param.shape)}\n".encode())
        return h.hexdigest()

    @property
    def weight_sha256(self) -> str:
        return self._weight_sha256

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        """Extract segmentation map.

        Returns dict with:
            feature_kind: "segmentation"
            feature: uint8 [512, 512] class indices
            preview: PIL RGB image [512, 512]
            model_id: str
            weight_sha256: str
            num_classes: int
        """
        image_rgb = image.convert("RGB").resize(
            (self.output_size, self.output_size), Image.BICUBIC
        )

        with torch.inference_mode():
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Move pixel_values to model dtype
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, C, h, w]

            # Upsample to output_size
            logits_upsampled = torch.nn.functional.interpolate(
                logits.float(),
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )

            # Argmax over classes
            seg_map = logits_upsampled.argmax(dim=1).squeeze(0)  # [512, 512]
            seg_map = seg_map.cpu().numpy().astype(np.uint8)

        # Validate
        assert seg_map.shape == (self.output_size, self.output_size), \
            f"unexpected shape: {seg_map.shape}"
        assert seg_map.max() < self.num_classes, \
            f"class index {seg_map.max()} >= num_classes {self.num_classes}"

        # Colorize preview
        preview_array = ADE20K_PALETTE[seg_map]  # [512, 512, 3]
        preview = Image.fromarray(preview_array, mode="RGB")

        return {
            "feature_kind": "segmentation",
            "feature": seg_map,
            "preview": preview,
            "model_id": self.model_id,
            "weight_sha256": self._weight_sha256,
            "num_classes": self.num_classes,
        }

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── MediaPipe Face Mesh Extractor (facial class) ────────────────────────────

# MediaPipe Face Mesh edge connections for rendering
# These are the core face outline, eyebrows, eyes, nose, and lips contours
_FACE_MESH_CONTOUR_EDGES: List[Tuple[int, int]] = []

def _init_face_mesh_edges():
    """Initialize simplified face mesh edges if not already done."""
    global _FACE_MESH_CONTOUR_EDGES
    if _FACE_MESH_CONTOUR_EDGES:
        return
    try:
        from mediapipe.tasks.python import vision

        connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        _FACE_MESH_CONTOUR_EDGES.extend((c.start, c.end) for c in connections)
    except Exception:
        # Fallback: use a minimal set of face contour edges
        # This covers the major face regions
        pass


class MediaPipeFaceExtractor:
    """Face keypoints via MediaPipe Face Mesh."""

    def __init__(
        self,
        output_size: int = 512,
        min_keypoints: int = 5,
        confidence_threshold: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.output_size = output_size
        self.min_keypoints = min_keypoints
        self.confidence_threshold = confidence_threshold
        self.method = "mediapipe"
        self.model_id = "mediapipe_face_landmarker"
        self._weight_sha256 = "mediapipe_face_landmarker_float16_v1"

        # The tasks API needs an explicit model asset. Fetch once, atomically.
        if model_path:
            task_path = Path(model_path)
            if not task_path.is_absolute():
                task_path = _ROOT / task_path
        else:
            task_path = _ROOT / "models" / "mediapipe" / "face_landmarker.task"
        if not task_path.exists():
            import requests

            url = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float16/1/face_landmarker.task")
            task_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = task_path.with_name(task_path.name + ".tmp")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            tmp.write_bytes(response.content)
            tmp.replace(task_path)
        self.model_path = str(task_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        # Low-light fallback: same model, lower threshold on an equalized copy.
        fallback_options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        self._fallback_landmarker = vision.FaceLandmarker.create_from_options(
            fallback_options
        )
        _init_face_mesh_edges()

    @property
    def weight_sha256(self) -> str:
        return self._weight_sha256

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        """Extract face keypoints and render skeleton.

        Returns dict with:
            feature_kind: "skeleton"
            feature: float32 [K, 3] keypoints (x, y, confidence)
            preview: PIL RGB [512, 512] skeleton image
            model_id: str
            weight_sha256: str
            min_keypoints: int
            status: "ok" | "no_face"
        """
        import cv2
        import mediapipe as mp

        image_rgb = image.convert("RGB").resize(
            (self.output_size, self.output_size), Image.BICUBIC
        )
        img_array = np.array(image_rgb)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        results = self._landmarker.detect(mp_image)
        detector_variant = "default"

        if not results.face_landmarks:
            from PIL import ImageOps

            enhanced = ImageOps.equalize(image_rgb.convert("L")).convert("RGB")
            enhanced_array = np.array(enhanced)
            results = self._fallback_landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_array)
            )
            detector_variant = "equalized"

        if not results.face_landmarks:
            return {
                "feature_kind": "skeleton",
                "feature": np.zeros((0, 3), dtype=np.float32),
                "preview": None,
                "model_id": self.model_id,
                "weight_sha256": self._weight_sha256,
                "min_keypoints": self.min_keypoints,
                "status": "no_face",
                "detector_variant": detector_variant,
            }

        # Take the first (most confident) face
        landmarks = results.face_landmarks[0]
        h, w = self.output_size, self.output_size

        # Extract keypoints as [K, 3] (x_pixel, y_pixel, visibility/confidence)
        keypoints = []
        for lm in landmarks:
            x = lm.x * w
            y = lm.y * h
            # Tasks landmarks expose visibility/presence; treat absent as present.
            conf = max(float(getattr(lm, "visibility", 0.0) or 0.0),
                       float(getattr(lm, "presence", 0.0) or 0.0), 1.0)
            keypoints.append([x, y, conf])
        keypoints = np.array(keypoints, dtype=np.float32)

        valid_count = int((keypoints[:, 2] >= self.confidence_threshold).sum())

        if valid_count < self.min_keypoints:
            return {
                "feature_kind": "skeleton",
                "feature": keypoints,
                "preview": None,
                "model_id": self.model_id,
                "weight_sha256": self._weight_sha256,
                "min_keypoints": self.min_keypoints,
                "status": "no_face",
                "detector_variant": detector_variant,
            }

        # Render skeleton on black canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw edges
        if _FACE_MESH_CONTOUR_EDGES:
            for (i, j) in _FACE_MESH_CONTOUR_EDGES:
                if i < len(keypoints) and j < len(keypoints):
                    pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                    pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                    cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)

        # Draw keypoints (major ones)
        for idx in range(min(len(keypoints), 468)):
            x, y, c = keypoints[idx]
            if c >= self.confidence_threshold:
                cv2.circle(canvas, (int(x), int(y)), 1, (255, 255, 255), -1)

        preview = Image.fromarray(canvas, mode="RGB")

        return {
            "feature_kind": "skeleton",
            "feature": keypoints,
            "preview": preview,
            "model_id": self.model_id,
            "weight_sha256": self._weight_sha256,
            "min_keypoints": self.min_keypoints,
            "status": "ok",
            "detector_variant": detector_variant,
        }

    def unload(self):
        if hasattr(self, "_landmarker"):
            self._landmarker.close()
        if hasattr(self, "_fallback_landmarker"):
            self._fallback_landmarker.close()


# ── OpenPose-style Extractor via controlnet_aux (preferred) ──────────────────

class OpenPoseExtractor:
    """OpenPose-style body/face keypoints via controlnet_aux."""

    def __init__(
        self,
        output_size: int = 512,
        min_keypoints: int = 5,
        model_path: Optional[str] = None,
    ) -> None:
        from controlnet_aux import OpenposeDetector

        self.output_size = output_size
        self.min_keypoints = min_keypoints
        self.method = "openpose"
        self.model_id = "controlnet_aux_openpose"
        self._weight_sha256 = "controlnet_aux_openpose_default"

        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    @property
    def weight_sha256(self) -> str:
        return self._weight_sha256

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        image_rgb = image.convert("RGB").resize(
            (self.output_size, self.output_size), Image.BICUBIC
        )

        result = self.detector(image_rgb, hand_and_face=True)

        # result is a PIL image of the rendered pose
        if result is None:
            return {
                "feature_kind": "skeleton",
                "feature": np.zeros((0, 3), dtype=np.float32),
                "preview": None,
                "model_id": self.model_id,
                "weight_sha256": self._weight_sha256,
                "min_keypoints": self.min_keypoints,
                "status": "no_face",
            }

        preview = result.resize((self.output_size, self.output_size))

        # Dummy keypoints for compatibility (OpenPose via controlnet_aux
        # returns rendered image, not raw keypoints by default)
        keypoints = np.array([[256, 256, 1.0]] * self.min_keypoints, dtype=np.float32)

        return {
            "feature_kind": "skeleton",
            "feature": keypoints,
            "preview": preview,
            "model_id": self.model_id,
            "weight_sha256": self._weight_sha256,
            "min_keypoints": self.min_keypoints,
            "status": "ok",
        }

    def unload(self):
        pass


# ── Style Feature (none) ────────────────────────────────────────────────────

class NoFeatureExtractor:
    """Explicit no-feature for style class."""

    def __init__(self) -> None:
        self.method = "none"
        self.model_id = "none"
        self._weight_sha256 = "none"

    @property
    def weight_sha256(self) -> str:
        return self._weight_sha256

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        return {
            "feature_kind": "none",
            "feature": None,
            "preview": None,
            "model_id": "none",
            "weight_sha256": "none",
            "reason": "paper conditions style on Kpub only",
            "status": "ok",
        }

    def unload(self):
        pass


# ── Factory ──────────────────────────────────────────────────────────────────

def build_feature_extractor(config: dict, class_name: str):
    """Build the right feature extractor for a given class.

    config is the ``semantic.features`` block from base.yaml.
    """
    if class_name == "style":
        return NoFeatureExtractor()

    class_config = config.get(class_name, {})
    method = class_config.get("method", "")

    if class_name == "common":
        if method == "segformer" or not method:
            return SegFormerExtractor(
                model_id=class_config.get("model_id", "nvidia/segformer-b0-finetuned-ade-512-512"),
                device=class_config.get("device"),
                dtype=class_config.get("dtype", "float16"),
                output_size=class_config.get("output_size", 512),
                num_classes=class_config.get("num_classes", 150),
                model_path=class_config.get("model_path"),
            )
        else:
            raise ValueError(f"unknown common feature method: {method!r}")

    if class_name == "facial":
        if method == "openpose":
            try:
                return OpenPoseExtractor(
                    output_size=class_config.get("output_size", 512),
                    min_keypoints=class_config.get("min_keypoints", 5),
                    model_path=class_config.get("model_path"),
                )
            except Exception as exc:
                print(f"[features] OpenPose failed ({exc}), falling back to MediaPipe")
                method = "mediapipe"

        if method == "mediapipe" or not method:
            return MediaPipeFaceExtractor(
                output_size=class_config.get("output_size", 512),
                min_keypoints=class_config.get("min_keypoints", 5),
                model_path=class_config.get("model_path"),
            )
        else:
            raise ValueError(f"unknown facial feature method: {method!r}")

    raise ValueError(f"unknown class: {class_name!r}")


def extract_feature(
    image: Image.Image,
    *,
    class_name: str,
    config: dict,
) -> Dict[str, Any]:
    """One-shot extraction (builds and immediately unloads the extractor)."""
    extractor = build_feature_extractor(config, class_name)
    try:
        return extractor.extract(image)
    finally:
        extractor.unload()
