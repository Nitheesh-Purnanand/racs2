"""Lossless secret-image hiding for the phone app (exact, byte-for-byte recovery).

The secret *file* is encrypted with a key-derived stream, authenticated with
HMAC-SHA256, and written into the 2 least-significant bits of a cover image at
key-permuted positions. The receiver needs only the stego PNG and the key; no
model runs on the receive side, and a wrong key is detected (MAC), not guessed.

Cover = a RACS-retrieved corpus photo (CLIP Top-K + token selection), upscaled
just enough to hold the payload.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

MAGIC = b"RACS"
BITS = 2                     # LSBs used per channel value
HEADER = 8                   # magic + payload length, encrypted
TAG = 32                     # HMAC-SHA256
KDF_ITER = 200_000           # slows brute force of the short numeric key


def _keys(key: str) -> tuple[bytes, bytes, int]:
    master = hashlib.pbkdf2_hmac("sha256", key.encode(), b"racs2-lossless-v1", KDF_ITER)
    enc = hashlib.sha256(master + b"enc").digest()
    mac = hashlib.sha256(master + b"mac").digest()
    seed = int.from_bytes(hashlib.sha256(master + b"perm").digest()[:8], "big")
    return enc, mac, seed


def _stream(enc_key: bytes, n: int) -> np.ndarray:
    return np.frombuffer(hashlib.shake_256(enc_key).digest(n), dtype=np.uint8)


def _positions(seed: int, total: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(total)


def _to_symbols(data: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return (bits[0::2] << 1) | bits[1::2]


def _from_symbols(sym: np.ndarray) -> bytes:
    bits = np.empty(sym.size * 2, dtype=np.uint8)
    bits[0::2], bits[1::2] = sym >> 1, sym & 1
    return np.packbits(bits).tobytes()


def capacity_side(n_bytes: int, minimum: int = 512) -> int:
    """Smallest square side (multiple of 64) whose 2-LSB capacity holds n_bytes."""
    side = math.ceil(math.sqrt(n_bytes * 8 / (3 * BITS)))
    return max(minimum, -(-side // 64) * 64)


def pack(name: str, data: bytes) -> bytes:
    nb = name.encode()[:200]
    return struct.pack(">B", len(nb)) + nb + data


def unpack(payload: bytes) -> tuple[str, bytes]:
    n = payload[0]
    return payload[1:1 + n].decode(errors="replace"), payload[1 + n:]


def capacity_size(n_bytes: int, w: int, h: int) -> tuple[int, int]:
    """Smallest size with the cover's aspect ratio (never below it) that holds n_bytes."""
    need = n_bytes * 8 / (3 * BITS)
    scale = max(1.0, math.sqrt(need / (w * h)))
    while True:
        size = (math.ceil(w * scale), math.ceil(h * scale))
        if size[0] * size[1] * 3 * BITS >= n_bytes * 8:
            return size
        scale *= 1.01


def embed(cover: Image.Image, secret_name: str, secret_bytes: bytes, key: str,
          keep_aspect: bool = False) -> Image.Image:
    body = pack(secret_name, secret_bytes)
    enc_key, mac_key, seed = _keys(key)
    plain = MAGIC + struct.pack(">I", len(body)) + body
    cipher = (np.frombuffer(plain, np.uint8) ^ _stream(enc_key, len(plain))).tobytes()
    blob = cipher + hmac.new(mac_key, cipher, "sha256").digest()
    if keep_aspect:
        size = capacity_size(len(blob), *cover.size)
    else:
        size = (capacity_side(len(blob)),) * 2
    img = cover.convert("RGB")
    arr = np.asarray(img if img.size == size else img.resize(size, Image.LANCZOS)).copy()
    flat = arr.reshape(-1)
    sym = _to_symbols(blob)
    if sym.size > flat.size:
        raise ValueError("secret too large for cover")
    pos = _positions(seed, flat.size)[: sym.size]
    flat[pos] = (flat[pos] & 0xFC) | sym
    return Image.fromarray(arr)


def extract(stego: Image.Image, key: str) -> tuple[str, bytes]:
    flat = np.asarray(stego.convert("RGB")).reshape(-1)
    enc_key, mac_key, seed = _keys(key)
    pos = _positions(seed, flat.size)
    head_sym = (flat[pos[: HEADER * 4]] & 3)
    head = np.frombuffer(_from_symbols(head_sym), np.uint8) ^ _stream(enc_key, HEADER)
    if head[:4].tobytes() != MAGIC:
        raise ValueError("wrong key, or the image was modified/recompressed")
    n = struct.unpack(">I", head[4:8].tobytes())[0]
    total = HEADER + n + TAG
    if total * 4 > flat.size:
        raise ValueError("wrong key, or the image was modified/recompressed")
    raw = _from_symbols(flat[pos[: total * 4]] & 3)
    cipher, tag = raw[: HEADER + n], raw[HEADER + n:]
    if not hmac.compare_digest(tag, hmac.new(mac_key, cipher, "sha256").digest()):
        raise ValueError("integrity check failed: wrong key or modified image")
    plain = (np.frombuffer(cipher, np.uint8) ^ _stream(enc_key, len(cipher))).tobytes()
    return unpack(plain[HEADER:])


# ---------------------------------------------------------------------------
# Contribution 2: texture-adaptive +-1 embedding (receiver-recomputable)
# ---------------------------------------------------------------------------
ADAPT_FRACTION = 0.5        # share of channel values eligible (most textured)
ADAPT_RATE = 0.25           # target payload bits per channel value (carrier grows to fit)
SHIFT = 3                   # texture on the top 5 bits; +-1 never crosses an 8-boundary
ADAPT_NOISE = 1.0           # sensor-like dither on the upscaled carrier (sigma, grey levels)


def carrier_seed(key: str) -> int:
    return _keys(key)[2] ^ 0x5EED


def prepare_carrier(img: Image.Image, size: tuple, seed: int,
                    noise: float = ADAPT_NOISE) -> np.ndarray:
    """Upscale the generative stego to the needed size, then add camera-like noise.

    An upscaled 256 px image has an almost noise-free LSB plane, which makes any
    embedding stand out; real photos never look like that. The dither belongs to
    the cover source, so a payload-free carrier is made exactly the same way.
    """
    img = img.convert("RGB")
    arr = np.asarray(img if img.size == tuple(size) else img.resize(size, Image.LANCZOS))
    if noise > 0:
        n = np.random.default_rng(seed).normal(0.0, noise, arr.shape)
        arr = np.clip(np.rint(arr + n), 0, 255)
    return arr.astype(np.uint8).copy()


def _texture(arr: np.ndarray) -> np.ndarray:
    """Local activity of the top 5 bits; the +-1 embedding below never changes it."""
    a = (arr >> SHIFT).astype(np.int16)
    t = np.zeros_like(a)
    t[:, :-1] += np.abs(a[:, 1:] - a[:, :-1])
    t[:, 1:] += np.abs(a[:, 1:] - a[:, :-1])
    t[:-1] += np.abs(a[1:] - a[:-1])
    t[1:] += np.abs(a[1:] - a[:-1])
    return t.reshape(-1)


def _adaptive_positions(arr: np.ndarray, seed: int) -> np.ndarray:
    tex = _texture(arr)
    n = int(tex.size * ADAPT_FRACTION)
    rank = np.random.default_rng(seed).permutation(tex.size)   # key-driven tie-break
    score = (int(tex.max()) - tex.astype(np.int64)) * tex.size + rank  # unique keys
    sel = np.sort(np.argpartition(score, n - 1)[:n])                   # the n most textured
    return sel[np.random.default_rng(seed + 1).permutation(n)]


def _seal(name: str, data: bytes, key: str) -> tuple[bytes, int]:
    body = pack(name, data)
    enc_key, mac_key, seed = _keys(key)
    plain = MAGIC + struct.pack(">I", len(body)) + body
    cipher = (np.frombuffer(plain, np.uint8) ^ _stream(enc_key, len(plain))).tobytes()
    return cipher + hmac.new(mac_key, cipher, "sha256").digest(), seed


def _open(read_bytes, key: str) -> tuple[str, bytes]:
    enc_key, mac_key, _ = _keys(key)
    head = np.frombuffer(read_bytes(HEADER), np.uint8) ^ _stream(enc_key, HEADER)
    if head[:4].tobytes() != MAGIC:
        raise ValueError("wrong key, or the image was modified/recompressed")
    n = struct.unpack(">I", head[4:8].tobytes())[0]
    raw = read_bytes(HEADER + n + TAG)
    if raw is None:
        raise ValueError("wrong key, or the image was modified/recompressed")
    cipher, tag = raw[: HEADER + n], raw[HEADER + n:]
    if not hmac.compare_digest(tag, hmac.new(mac_key, cipher, "sha256").digest()):
        raise ValueError("integrity check failed: wrong key or modified image")
    plain = (np.frombuffer(cipher, np.uint8) ^ _stream(enc_key, len(cipher))).tobytes()
    return unpack(plain[HEADER:])


def compact(name: str, data: bytes) -> tuple[str, bytes]:
    """Lossless WebP if it is smaller: pixel-identical secret, fewer bits to hide."""
    import io
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        buf = io.BytesIO()
        im.save(buf, format="WEBP", lossless=True, quality=100, method=6, exact=True)
        if buf.tell() < len(data):
            return Path(name).stem + ".webp", buf.getvalue()
    except Exception:  # noqa: BLE001  (not an image PIL can re-encode)
        pass
    return name, data


def embed_adaptive(cover: Image.Image, secret_name: str, secret_bytes: bytes, key: str,
                   keep_aspect: bool = True, noise: float = ADAPT_NOISE) -> Image.Image:
    blob, seed = _seal(secret_name, secret_bytes, key)
    bits = np.unpackbits(np.frombuffer(blob, np.uint8))
    img = cover.convert("RGB")
    w, h = img.size
    scale = 1.0
    while True:  # 1 bit per eligible value; eligible = ADAPT_FRACTION of values
        size = ((math.ceil(w * scale), math.ceil(h * scale)) if keep_aspect
                else (max(512, math.ceil(max(w, h) * scale)),) * 2)
        if size[0] * size[1] * 3 * ADAPT_RATE >= bits.size:
            break
        scale = max(scale * 1.02, math.sqrt(bits.size / (w * h * 3 * ADAPT_RATE)))
    arr = prepare_carrier(img, size, carrier_seed(key), noise)
    flat = arr.reshape(-1)
    pos = _adaptive_positions(arr, seed)[: bits.size]
    v = flat[pos].astype(np.int16)
    need = (v & 1) != bits
    low = v & ((1 << SHIFT) - 1)
    top = (1 << SHIFT) - 1
    # LSB matching: random +-1, except at the edges of an 8-value block where
    # only one direction keeps the top bits (and so the receiver's texture map)
    rnd = np.random.default_rng(seed ^ 0xA5A5).integers(0, 2, size=v.size) * 2 - 1
    step = np.where(low == 0, 1, np.where(low == top, -1, rnd))
    v = np.where(need, v + step, v)
    flat[pos] = v.astype(np.uint8)
    return Image.fromarray(arr)


def extract_adaptive(stego: Image.Image, key: str) -> tuple[str, bytes]:
    arr = np.asarray(stego.convert("RGB"))
    flat = arr.reshape(-1)
    _, _, seed = _keys(key)
    pos = _adaptive_positions(arr, seed)

    def read(nbytes):
        if nbytes * 8 > pos.size:
            return None
        return np.packbits(flat[pos[: nbytes * 8]] & 1).tobytes()

    return _open(read, key)


def extract_any(stego: Image.Image, key: str) -> tuple[str, bytes]:
    try:
        return extract_adaptive(stego, key)
    except ValueError:
        return extract(stego, key)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":  # quick self-test: exact round trip + wrong key rejected
    import sys

    src = Path(sys.argv[1])
    cover = Image.open(sys.argv[2])
    data = src.read_bytes()
    st = embed(cover, src.name, data, "12345678")
    out = Path("temp/app/lossless_selftest.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    st.save(out)
    name, got = extract(Image.open(out), "12345678")
    print("exact:", got == data, name, len(data), "bytes; stego", st.size)
    try:
        extract(Image.open(out), "12345679")
        print("wrong key NOT rejected")
    except ValueError as e:
        print("wrong key rejected:", e)
