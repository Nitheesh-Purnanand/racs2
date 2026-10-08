"""
Phase 3 (supplement) - what does the model do when we feed *our* image in?

`phase3_diffusion.py` is text -> image: the reference `Xref` is synthesised from a
semantic key with no input image. This script answers the natural follow-up:
"what comes out if we give it one of our own images?" It runs the *same* SD v1.5
weights through `StableDiffusionImg2ImgPipeline`, i.e. the input image is encoded
to latents, noised to `strength` of the schedule, and denoised back with the
prompt guiding it.

`strength` is the interesting knob:
    0.0 -> output == input (no denoising at all)
    low  -> light restyle; composition/person preserved
    high -> prompt dominates; input becomes a loose suggestion
    1.0 -> equivalent to text2img (input is fully destroyed)

The img2img pipeline is built by *reusing the components of the text2img pipeline*
(`StableDiffusionImg2ImgPipeline(**pipe.components)`), so no second copy of the
weights is loaded - that matters on this machine's tight commit budget.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\phase3_img2img.py
    .venv\\Scripts\\python.exe scripts\\phase3_img2img.py --strengths 0.3 0.6 --images data\\secret_images\\test_001.png
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.paths  # noqa: E402,F401  (HF cache on D:)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

from common.device import get_device  # noqa: E402
from common.paths import RESULTS_DIR, PROJECT_ROOT as _ROOT  # noqa: E402
from common.seed import set_seed  # noqa: E402
from models.sd_pipeline import free_commit_gb, load_pipeline  # noqa: E402

OUT_JSON = RESULTS_DIR / "phase3_img2img.json"


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def prepare_input(path: Path, size: int) -> Image.Image:
    """Any photo -> a square RGB image the model can encode (centre-crop, no stretch)."""
    img = Image.open(path).convert("RGB")
    # centre-crop rather than squash: keeps the subject's proportions intact.
    return ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.4))


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float32)
    y = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.abs(x - y).mean())


def make_grid(panels, labels, path: Path) -> None:
    """Side-by-side strip: input | strength 0.25 | ... with captions."""
    w, h = panels[0].size
    pad, cap = 8, 22
    grid = Image.new("RGB", (len(panels) * w + (len(panels) + 1) * pad, h + cap + 2 * pad), "white")
    draw = ImageDraw.Draw(grid)
    x = pad
    for panel, label in zip(panels, labels):
        grid.paste(panel, (x, pad))
        draw.text((x + 2, pad + h + 4), label, fill="black")
        x += w + pad
    grid.save(path)


def build_img2img(pipe):
    """Reuse the text2img components instead of loading the weights twice."""
    from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline

    sig = inspect.signature(StableDiffusionImg2ImgPipeline.__init__)
    components = {
        name: pipe.components[name]
        for name, param in sig.parameters.items()
        if name != "self" and name in pipe.components
    }
    # Parameters the constructor requires positionally (e.g. safety_checker, which
    # we deliberately disable) must still be supplied, even as None.
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and name != "self":
            components.setdefault(name, None)
    img2img = StableDiffusionImg2ImgPipeline(**components)
    # Fresh scheduler so the two pipelines never share mutable timestep state.
    img2img.scheduler = DDIMScheduler.from_config(img2img.scheduler.config)
    img2img.enable_attention_slicing()
    try:
        img2img.enable_vae_slicing()
    except Exception:  # noqa: BLE001 - optional
        pass
    img2img.set_progress_bar_config(disable=True)
    return img2img


@torch.no_grad()
def run(img2img, init_image, prompt, strength, steps, guidance, seed):
    """Returns (image, seconds, peak_vram_gb)."""
    device = img2img.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = img2img(
        prompt=prompt,
        image=init_image,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    return out.images[0], secs, peak


def main() -> None:
    ap = argparse.ArgumentParser(description="SD v1.5 img2img on our own images")
    ap.add_argument("--images", nargs="+", default=[str(_ROOT / "data" / "secret_images" / "test_001.png")])
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    section("1. Setup")
    set_seed(args.seed, verbose=True)
    device = get_device()
    print(f"Selected device : {device}")
    print(f"free commit     : {free_commit_gb():.2f} GB")

    section("2. Config")
    with open(_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    gen = config.get("generation", {})
    prompt = gen.get("prompt", "a photograph of an astronaut riding a horse")
    guidance = float(gen.get("guidance_scale", 7.5))
    size = int(config.get("image_size", 512))
    print(f"prompt   = {prompt!r}")
    print(f"guidance = {guidance}   size = {size}   steps = {args.steps}   seed = {args.seed}")

    section("3. Load SD v1.5 (text2img components) + build img2img on top")
    pipe = load_pipeline(device)
    img2img = build_img2img(pipe)
    print(f"img2img device  : {img2img.device}")

    section("4. img2img sweep")
    records = []
    for image_path in args.images:
        src = Path(image_path)
        init_image = prepare_input(src, size)
        # NOTE: strength must be > 0. diffusers computes the start timestep from it,
        # and strength=0 leaves an empty timestep slice (empty latents -> reshape error).
        for strength in args.strengths:
            out, secs, peak = run(img2img, init_image, prompt, strength, args.steps, guidance, args.seed)
            mad = mean_abs_diff(init_image, out)
            tag = f"{src.stem}_s{strength:g}"
            out_path = RESULTS_DIR / f"phase3_img2img_{tag}.png"
            out.save(out_path)
            rec = {
                "input": str(src.relative_to(_ROOT)).replace("\\", "/"),
                "output": str(out_path.relative_to(_ROOT)).replace("\\", "/"),
                "prompt": prompt,
                "strength": float(strength),
                "num_inference_steps": int(args.steps),
                "guidance_scale": guidance,
                "seed": int(args.seed),
                "seconds": round(secs, 3),
                "peak_vram_gb": round(peak, 3),
                "mean_abs_diff_vs_input": round(mad, 3),
            }
            records.append(rec)
            print(
                f"  {src.name:14s} strength={strength:4.2f} | {secs:6.2f}s | "
                f"peak {peak:4.2f} GB | mean|diff| vs input {mad:6.2f}"
            )

        # visual strip for this input
        panels, labels = [init_image], ["input"]
        for strength in args.strengths:
            p = RESULTS_DIR / f"phase3_img2img_{src.stem}_s{strength:g}.png"
            if p.exists():
                panels.append(Image.open(p))
                labels.append(f"strength {strength:g}")
        grid = RESULTS_DIR / f"phase3_img2img_{src.stem}_grid.png"
        make_grid(panels, labels, grid)
        print(f"  -> grid: {grid.name}")

    section("5. Summary")
    print(f"{'input':>14} {'strength':>9} {'sec':>7} {'peakGB':>7} {'mean|diff|':>10}")
    for r in records:
        print(
            f"{Path(r['input']).name:>14} {r['strength']:>9.2f} {r['seconds']:>7.2f} "
            f"{r['peak_vram_gb']:>7.2f} {r['mean_abs_diff_vs_input']:>10.2f}"
        )

    OUT_JSON.write_text(
        json.dumps({"prompt": prompt, "guidance_scale": guidance, "seed": args.seed, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {OUT_JSON}")

    section("Phase 3 img2img supplement complete")


if __name__ == "__main__":
    main()
