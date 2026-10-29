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


def embed(cover: Image.Image, secret_name: str, secret_bytes: bytes, key: str) -> Image.Image:
    body = pack(secret_name, secret_bytes)
    enc_key, mac_key, seed = _keys(key)
    plain = MAGIC + struct.pack(">I", len(body)) + body
    cipher = (np.frombuffer(plain, np.uint8) ^ _stream(enc_key, len(plain))).tobytes()
    blob = cipher + hmac.new(mac_key, cipher, "sha256").digest()
    side = capacity_side(len(blob))
    arr = np.asarray(cover.convert("RGB").resize((side, side), Image.LANCZOS)).copy()
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
