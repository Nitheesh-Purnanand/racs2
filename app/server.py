"""RACS phone app: hide a secret image on one phone, recover it on another.

    .venv\\Scripts\\python.exe app\\server.py            # then open the printed URL on a phone

Sender  : upload secret + class -> stego PNG (public Kpub/class in its metadata) + private key.
Receiver: upload stego PNG + key -> recovered secret.

The stego image is sent as a *file* (e.g. WhatsApp "Document", email attachment):
messengers that recompress photos destroy both the hidden signal and the metadata.
The digital file transfer replaces the simulated SwinJSCC/AWGN channel of the
experiments; everything else is the frozen baseline + RACS retrieval code.
"""

from __future__ import annotations

import base64
import cgi
import html
import io
import json
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps, PngImagePlugin  # noqa: E402

import common.paths as paths  # noqa: E402
from pipelines.agentsemstecom import Baseline  # noqa: E402
from steganography import pipeline  # noqa: E402
from steganography.mask import token_delta, token_noise  # noqa: E402

WORK = paths.PROJECT_ROOT / "temp" / "app"
WORK.mkdir(parents=True, exist_ok=True)
META_KEY = "racs2"
CLASSES = ("common", "facial", "style")


class Worker:
    """Persistent public_worker.py process (CLIP / keygen live there)."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(_ROOT / "app" / "public_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=str(_ROOT))
        self.lock = threading.Lock()
        assert json.loads(self.proc.stdout.readline()).get("ready")

    def call(self, **req) -> dict:
        with self.lock:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            reply = json.loads(self.proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply


def to_square(src: Path, dst: Path) -> dict:
    """Edge-pad to a square (no stretching); remembers how to undo it."""
    im = Image.open(src)
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    side = max(w, h)
    left, top = (side - w) // 2, (side - h) // 2
    arr = np.pad(np.asarray(im), ((top, side - h - top), (left, side - w - left), (0, 0)),
                 mode="edge")
    Image.fromarray(arr).save(dst)
    return {"w": w, "h": h, "side": side, "left": left, "top": top}


def from_square(square_px, geom: dict, max_side: int = 1024) -> Image.Image:
    """Undo to_square on a recovered [1,3,S,S] tensor, at up to max_side px."""
    arr = (square_px[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
    side = geom["side"]
    scale = min(1.0, max_side / side)
    big = Image.fromarray(arr).resize((round(side * scale),) * 2, Image.LANCZOS)
    box = [round(v * scale) for v in (geom["left"], geom["top"],
                                      geom["left"] + geom["w"], geom["top"] + geom["h"])]
    return big.crop(box)


class Engine:
    def __init__(self, size: int = 512):
        print("[app] starting public worker (CLIP retrieval + Kpub) ...", flush=True)
        self.worker = Worker()
        print("[app] loading baseline downstream (EDICT, IPAdapter, CodeFormer) ...", flush=True)
        self.cfg = pipeline.load_config()
        self.b = Baseline(self.cfg)
        # SD v1.5's native 512 px: the experiments use 256 px for the 6 GB
        # sweep budget, but at 256 px the VAE visibly blurs a single photo.
        self.b.stego_cfg["image_size"] = int(size)
        self.size = int(size)
        self.lock = threading.Lock()

    def _blob(self, key: str) -> dict:
        c = self.b.stego_cfg
        shape = (1, 4, self.size // 8, self.size // 8)
        return {"delta": token_delta(key, shape=shape, fraction=float(c["mask_fraction"])),
                "noise": token_noise(key, shape=shape), "eta": float(c["eta"]),
                "steps": int(c["edict_steps"]), "mixing": float(c["edict_mixing"])}

    def send(self, secret_path: Path, cls: str, kpub: str = "") -> dict:
        with self.lock:
            square = secret_path.with_name(secret_path.stem + "_sq.png")
            geom = to_square(secret_path, square)
            secret_path = square
            t0 = time.perf_counter()
            kpub = kpub.strip() or self.worker.call(op="kpub", image=str(secret_path))["kpub"]
            t_kpub = time.perf_counter() - t0
            key = f"{secrets.randbelow(10 ** 8):08d}"
            ref = self.worker.call(op="retrieve", kpub=kpub, token=key)
            secret = pipeline.load_pixels(str(secret_path), self.size)
            t1 = time.perf_counter()
            hidden = self.b.hide(secret, pipeline.load_pixels(ref["path"], self.size), kpub, key,
                                 reference_path=ref["path"])
            t_hide = time.perf_counter() - t1
            blob = self._blob(key)
            assert torch.equal(blob["delta"].to(hidden["delta"].dtype), hidden["delta"].cpu()), \
                "receiver could not regenerate the token perturbation"
            out = WORK / f"stego_{uuid.uuid4().hex[:8]}.png"
            arr = (hidden["stego_pixels"][0].permute(1, 2, 0).numpy() * 255).round()
            info = PngImagePlugin.PngInfo()
            info.add_text(META_KEY, json.dumps({"v": 2, "kpub": kpub, "class": cls, "geom": geom}))
            Image.fromarray(arr.clip(0, 255).astype("uint8")).save(out, pnginfo=info)
            return {"stego": out, "square": square, "key": key, "kpub": kpub, "class": cls, "xref": ref,
                    "timing": {"kpub_s": t_kpub, **ref["timing"], "hide_s": t_hide}}

    def receive(self, stego_path: Path, key: str, kpub: str = "", cls: str = "") -> dict:
        meta = json.loads(Image.open(stego_path).info.get(META_KEY, "{}") or "{}")
        kpub = kpub.strip() or meta.get("kpub", "")
        cls = cls or meta.get("class", "common")
        if not kpub:
            raise ValueError("no public description in the file (was it sent as a compressed "
                             "photo?) — send it as a document, or type the description")
        with self.lock:
            ref = self.worker.call(op="retrieve", kpub=kpub, token=key)
            received = pipeline.load_pixels(str(stego_path), self.size)
            t0 = time.perf_counter()
            rec = self.b.receive(received, token_mode="correct", token=key, wrong_token="",
                                 prompt=kpub, blob=self._blob(key),
                                 image_embeds=self.b.embeds_from_path(ref["path"]))
            rec, enhanced = self.b.maybe_enhance(rec, cls)
            t_rec = time.perf_counter() - t0
        out = WORK / f"recovered_{uuid.uuid4().hex[:8]}.png"
        geom = meta.get("geom")
        if geom:
            from_square(rec, geom).save(out)
        else:
            pipeline.save_pixels(rec, out)
        pipeline.save_pixels(rec, out.with_name(out.stem + "_sq.png"))
        return {"recovered": out, "kpub": kpub, "class": cls, "enhanced": enhanced,
                "xref": ref, "timing": {**ref["timing"], "recover_s": t_rec}}


class LosslessEngine:
    """Exact recovery: encrypted LSB embedding in a RACS-retrieved cover."""

    def __init__(self):
        from app import lossless
        self.L = lossless
        print("[app] starting public worker (CLIP cover retrieval) ...", flush=True)
        self.worker = Worker()

    def send(self, secret_path: Path, cls: str, kpub: str = "", name: str = "") -> dict:
        key = f"{secrets.randbelow(10 ** 8):08d}"
        cover = self.worker.call(op="retrieve_image", image=str(secret_path), token=key)
        data = secret_path.read_bytes()
        t0 = time.perf_counter()
        stego = self.L.embed(Image.open(cover["path"]), name or secret_path.name, data, key)
        out = WORK / f"stego_{uuid.uuid4().hex[:8]}.png"
        stego.save(out, optimize=False)
        return {"stego": out, "key": key, "kpub": "(not needed in lossless mode)",
                "class": cls, "xref": cover, "sha256": self.L.sha256(data),
                "timing": {"cover_acquisition_s": cover["timing"]["acquisition_s"],
                           "embed_s": time.perf_counter() - t0}}

    def receive(self, stego_path: Path, key: str, kpub: str = "", cls: str = "") -> dict:
        t0 = time.perf_counter()
        name, data = self.L.extract(Image.open(stego_path), key)
        out = WORK / f"recovered_{uuid.uuid4().hex[:8]}{Path(name).suffix or '.png'}"
        out.write_bytes(data)
        return {"recovered": out, "name": name, "sha256": self.L.sha256(data),
                "enhanced": False, "xref": {"image_id": "not needed (lossless)"},
                "timing": {"extract_s": time.perf_counter() - t0}}


class HybridEngine(Engine):
    """Paper-style stego look (EDICT + RACS reference) with exact recovery.

    EDICT blends the secret toward the retrieved reference at the paper's
    strength, then the encrypted original file is embedded in the 2 LSBs of
    that stego. The receiver extracts it byte for byte; no model is needed
    on the receive side.
    """

    def __init__(self, size: int = 512, strength: float = 0.5):
        super().__init__(size)
        from app import lossless
        self.L = lossless
        self.b.stego_cfg["edict_strength"] = float(strength)

    def send(self, secret_path: Path, cls: str, kpub: str = "", name: str = "") -> dict:
        data = secret_path.read_bytes()
        r = super().send(secret_path, cls, kpub)
        t0 = time.perf_counter()
        look = Image.open(r["stego"])
        # crop the padding back off so the stego keeps the photo's shape
        g = json.loads(look.info[META_KEY])["geom"]
        s = look.size[0] / g["side"]
        look = look.crop([round(v * s) for v in (g["left"], g["top"], g["left"] + g["w"],
                                                 g["top"] + g["h"])])
        stego = self.L.embed(look, name or secret_path.name, data, r["key"], keep_aspect=True)
        stego.save(r["stego"])
        r["timing"]["embed_s"] = time.perf_counter() - t0
        r["sha256"] = self.L.sha256(data)
        return r

    def receive(self, stego_path: Path, key: str, kpub: str = "", cls: str = "") -> dict:
        return LosslessEngine.receive(self, stego_path, key)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mut:#5b6470;--acc:#2f6fed;--bd:#dde1e6}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1a1f25;--fg:#e8eaed;--mut:#9aa3ad;--acc:#6b9bff;--bd:#2c333b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:560px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2rem 0 1rem}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px;margin-bottom:16px}
h2{font-size:1.1rem;margin:0 0 .6rem}label{display:block;margin:.6rem 0 .2rem;color:var(--mut);font-size:.9rem}
input,select,button{width:100%;font:inherit;padding:10px;border-radius:8px;border:1px solid var(--bd);background:var(--bg);color:var(--fg)}
button,.btn{background:var(--acc);color:#fff;border:0;margin-top:12px;font-weight:600;display:block;text-align:center;text-decoration:none;padding:12px;border-radius:8px}
img{width:100%;border-radius:8px;border:1px solid var(--bd)}.key{font:700 1.8rem ui-monospace,monospace;letter-spacing:.15em;text-align:center}
.mut{color:var(--mut);font-size:.9rem}.err{color:#d93025}table{width:100%;font-size:.9rem}td:last-child{text-align:right}
"""


def page(body: str) -> bytes:
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>RACS Stego</title>"
            f"<style>{CSS}</style></head><body><main><h1>RACS secret image messenger</h1>"
            f"{body}<p><a href='/'>&larr; back</a></p></main></body></html>").encode()


HOME = """
<div class=card><h2>1 · Send a secret image</h2>
<form method=post action=/send enctype=multipart/form-data>
<label>Secret image</label><input type=file name=image accept="image/*" required>
<label>Image type</label><select name=cls><option value=common>Scene / object</option>
<option value=facial>Face / portrait</option><option value=style>Artwork / style</option></select>
<label>Public description (optional — auto-generated if empty)</label><input name=kpub>
<button>Hide it</button></form>
<p class=mut>You get a cover image to send and a private key. Send the image <b>as a file/document</b>
and the key separately (tell it, or another app).</p></div>
<div class=card><h2>2 · Receive</h2>
<form method=post action=/receive enctype=multipart/form-data>
<label>Received image (PNG file)</label><input type=file name=image accept="image/png" required>
<label>Private key</label><input name=key inputmode=numeric pattern="[0-9]{8}" required>
<button>Reveal secret</button></form></div>
"""


MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp", ".heic": "image/heic"}


def img_tag(path: Path) -> str:
    return f"<img src='data:{MIME.get(path.suffix.lower(), 'image/png')};base64,{base64.b64encode(path.read_bytes()).decode()}'>"


def timing_table(t: dict) -> str:
    names = {"kpub_s": "Public description (Kpub)", "retrieve_s": "Retrieve Xref (CLIP+search+rerank+token)",
             "load_xref_s": "Load Xref image", "acquisition_s": "<b>Reference acquisition total</b>",
             "hide_s": "Hide (EDICT + IPAdapter)", "recover_s": "Recover (inversion + enhance)",
             "cover_acquisition_s": "<b>Cover retrieval (CLIP + search + token)</b>",
             "embed_s": "Encrypt + embed", "extract_s": "Extract + decrypt + verify"}
    return "<table>" + "".join(f"<tr><td>{names.get(k, k)}</td><td>{v:.3f} s</td></tr>"
                               for k, v in t.items()) + "</table>"


ENGINE: Engine | None = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/file/"):
            p = WORK / Path(self.path[6:]).name
            if p.exists() and p.suffix.lower() in MIME:
                return self._send(p.read_bytes(), MIME[p.suffix.lower()],
                                  {"Content-Disposition": f"attachment; filename={p.name}"})
            self.send_error(404)
            return
        self._send(page(HOME))

    def do_POST(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={"REQUEST_METHOD": "POST",
                                         "CONTENT_TYPE": self.headers["Content-Type"]})
        try:
            item = form["image"]
            up = WORK / f"upload_{uuid.uuid4().hex[:8]}{Path(item.filename or 'x.png').suffix or '.png'}"
            up.write_bytes(item.file.read())
            if self.path == "/send":
                cls = form.getfirst("cls", "common")
                if isinstance(ENGINE, (LosslessEngine, HybridEngine)):
                    r = ENGINE.send(up, cls, name=Path(item.filename or up.name).name)
                else:
                    r = ENGINE.send(up, cls if cls in CLASSES else "common",
                                    form.getfirst("kpub", ""))
                body = (f"<div class=card><h2>Send this image</h2>{img_tag(r['stego'])}"
                        f"<a class=btn href='/file/{r['stego'].name}' download>Download stego PNG</a>"
                        f"<p class=mut>Share it as a file/document, not as a photo.</p></div>"
                        f"<div class=card><h2>Private key</h2><div class=key>{r['key']}</div>"
                        f"<p class=mut>Give this to the receiver separately. Without it the image "
                        f"reveals nothing.</p></div><div class=card><h2>Details</h2>"
                        f"<p class=mut>Public description: {html.escape(r['kpub'])}<br>"
                        f"Retrieved reference: {r['xref']['image_id']}"
                        + (f"<br>Secret SHA-256: <code>{r['sha256'][:16]}…</code>"
                           if 'sha256' in r else "") + "</p>"
                        f"{timing_table(r['timing'])}</div>")
            elif self.path == "/receive":
                key = form.getfirst("key", "").strip()
                if not (key.isdigit() and len(key) == 8):
                    raise ValueError("the key is 8 digits")
                r = ENGINE.receive(up, key)
                body = (f"<div class=card><h2>Recovered secret</h2>{img_tag(r['recovered'])}"
                        f"<a class=btn href='/file/{r['recovered'].name}' download>Download</a></div>"
                        f"<div class=card><h2>Details</h2><p class=mut>Reference: "
                        f"{r['xref']['image_id']}{' · face enhanced (CodeFormer)' if r['enhanced'] else ''}"
                        + (f"<br>File: {html.escape(r['name'])} · exact copy, SHA-256 "
                           f"<code>{r['sha256'][:16]}…</code> (matches the sender's)"
                           if 'sha256' in r else "")
                        + f"</p>{timing_table(r['timing'])}</div>")
            else:
                self.send_error(404)
                return
        except Exception as err:  # noqa: BLE001
            body = f"<div class=card><p class=err>Error: {html.escape(str(err))}</p></div>"
        self._send(page(body))


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    global ENGINE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", choices=("hybrid", "lossless", "diffusion"), default="hybrid",
                    help="hybrid: paper-style stego look + exact recovery; lossless: exact "
                         "recovery in a retrieved cover; diffusion: the paper's EDICT pipeline")
    ap.add_argument("--strength", type=float, default=0.5, help="hybrid: EDICT strength")
    ap.add_argument("--size", type=int, default=512, help="working resolution (256 or 512)")
    ap.add_argument("--selftest", default="", help="secret image path: send+receive once, no server")
    args = ap.parse_args()
    ENGINE = (HybridEngine(args.size, args.strength) if args.mode == "hybrid"
              else LosslessEngine() if args.mode == "lossless" else Engine(args.size))
    if args.selftest and isinstance(ENGINE, (LosslessEngine, HybridEngine)):
        src = Path(args.selftest)
        s = ENGINE.send(src, "common")
        r = ENGINE.receive(s["stego"], s["key"])
        print(f"[selftest] exact copy: {r['recovered'].read_bytes() == src.read_bytes()} "
              f"cover={s['xref']['image_id']} timing={json.dumps(s['timing'])}")
        try:
            ENGINE.receive(s["stego"], f"{(int(s['key']) + 1) % 10 ** 8:08d}")
            print("[selftest] wrong key NOT rejected")
        except ValueError as err:
            print(f"[selftest] wrong key rejected: {err}")
        return 0
    if args.selftest:
        s = ENGINE.send(Path(args.selftest), "common")
        r = ENGINE.receive(s["stego"], s["key"])
        w = ENGINE.receive(s["stego"], f"{(int(s['key']) + 1) % 10 ** 8:08d}")
        secret = pipeline.load_pixels(str(s["square"]), ENGINE.size)
        for name, p in (("correct key", r["recovered"]), ("wrong key", w["recovered"])):
            sq = p.with_name(p.stem + "_sq.png")
            m = pipeline.metrics(secret, pipeline.load_pixels(str(sq), ENGINE.size), ENGINE.b.stego_cfg)
            print(f"[selftest] {name}: PSNR {m['psnr']:.2f} dB  SSIM {m['ssim']:.3f}  -> {p}")
        print(f"[selftest] stego {s['stego']}  key {s['key']}  timing {json.dumps(s['timing'])}")
        return 0
    srv = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[app] ready: open http://{lan_ip()}:{args.port} on a phone on the same Wi-Fi "
          f"(this PC: http://127.0.0.1:{args.port})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
