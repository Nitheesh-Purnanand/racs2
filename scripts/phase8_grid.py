"""Rebuild the Phase 8 contact sheet from an existing manifest (2 samples/cell)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths

import yaml
from PIL import Image, ImageDraw


def main() -> None:
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh).get("reference", {})
    manifest_path = paths.PROJECT_ROOT / cfg.get(
        "manifest_path", "experiments/results/phase8_references.json"
    )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    cells = defaultdict(list)
    for rec in manifest["records"]:
        cells[(rec["role"], rec["class"])].append(rec)

    thumb = 180
    rows = sorted(cells)
    cols = 2
    width = 130 + cols * (thumb * 2 + 10)
    height = len(rows) * (thumb + 26) + 10
    sheet = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    source_map = {}
    for split_role in ("calibration", "evaluation"):
        split = __import__("evaluation.splits", fromlist=["Split"]).Split.load(
            role=split_role, allow_calibration=(split_role == "calibration"), verify=False
        )
        for rec in split.records():
            source_map[rec["image_id"]] = rec["path"]

    for r, key in enumerate(rows):
        role, cls = key
        y = r * (thumb + 26) + 5
        draw.text((5, y + thumb // 2), f"{role[:4]}/{cls}", fill=(255, 255, 255))
        for c, rec in enumerate(cells[key][:cols]):
            x = 130 + c * (thumb * 2 + 10)
            src = Image.open(paths.PROJECT_ROOT / source_map[rec["image_id"]]).convert("RGB")
            ref = Image.open(paths.PROJECT_ROOT / rec["reference_path"]).convert("RGB")
            sheet.paste(src.resize((thumb, thumb), Image.BICUBIC), (x, y))
            sheet.paste(ref.resize((thumb, thumb), Image.BICUBIC), (x + thumb + 2, y))
            draw.text((x, y + thumb + 2), rec["image_id"][:24], fill=(190, 190, 190))
            if cls == "style":
                draw.text((x + thumb + 4, y + 4), "Kpub only", fill=(220, 220, 0))

    out = paths.RESULTS_DIR / "phase8_reference_grid.png"
    sheet.save(out, format="PNG")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
