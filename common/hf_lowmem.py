"""
Low-memory safetensors loading for Windows (RACS2).

Why this exists
---------------
On Windows, HuggingFace ``safetensors`` opens checkpoints with a *copy-on-write*
file mapping. Windows charges the **entire file** against the system commit
limit (RAM + page file) for such a mapping, even though nothing is ever written
to it. For Stable Diffusion v1.5 the UNet alone is ~3.2 GB, so loading it needs
3.2 GB of free commit.

This machine is a 16 GB box whose ``C:`` drive is full, so the page file is
capped at ~16 GB and the commit limit sits at ~32 GB. With the IDE/browser
running, free commit is often under ~1.5 GB, and the model load dies with::

    OSError: The paging file is too small for this operation to complete. (os error 1455)
    MemoryError

``transformers`` already works around this (it forces ``backend="pread"`` on
Windows, see ``transformers/modeling_utils.py``), but ``diffusers`` calls
``safetensors.torch.load_file`` with the default ``mmap`` backend.

What this module does
---------------------
It replaces ``safetensors.safe_open`` / ``safetensors.torch.load_file`` with an
implementation that maps the file **read-only** (``mmap.ACCESS_READ``). A
read-only, file-backed mapping costs no commit charge, and every tensor is
handed out as a zero-copy view into that mapping. The bytes still come straight
from the OS page cache, so this is not slower in any meaningful way for a
load-once workload.

Install it *before* importing ``transformers``/``diffusers``::

    import common.hf_lowmem
    common.hf_lowmem.install()

    from diffusers import StableDiffusionPipeline  # now commit-friendly

Notes / caveats
---------------
* Returned tensors share memory with the file mapping and are **read-only**.
  Model loaders only read from a state dict, so this is safe in practice.
  Do not mutate tensors obtained from these loaders.
* Mappings are kept alive for the process lifetime (a few file handles) so that
  lazily-materialised views stay valid after the ``with safe_open(...)`` block
  exits. That trade-off is deliberate: correctness over reclamation of a
  handful of handles.
* Set ``RACS2_LOWMEM_SAFETENSORS=0`` to disable the shim entirely.
* Any failure while opening through the shim falls back to the real
  ``safetensors`` implementation, so this cannot make things worse.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = ["install", "is_installed", "read_safetensors_readonly", "access_readonly"]

# safetensors dtype string -> torch dtype
_STR_TO_TORCH: Dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# Mappings (and their file handles) must outlive the tensors that view them.
_KEEPALIVE: List["_RoFile"] = []
_KEEPALIVE_LOCK = threading.Lock()

_ORIGINALS: Dict[str, Any] = {}
_INSTALLED = False
_INSTALL_LOCK = threading.Lock()


def _torch_dtype(dtype_str: str) -> torch.dtype:
    try:
        return _STR_TO_TORCH[dtype_str]
    except KeyError as err:  # pragma: no cover - exotic dtype
        raise ValueError(f"Unsupported safetensors dtype: {dtype_str!r}") from err


def _read_header(path: str) -> Tuple[Dict[str, Any], int]:
    """Reads the safetensors JSON header. Returns (header, data_start_offset)."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise ValueError(f"{path} has an implausible header length ({header_len})")
        header = json.loads(fh.read(header_len))
    if not isinstance(header, dict):
        raise ValueError(f"{path} has a malformed safetensors header")
    return header, 8 + header_len


class _Slice:
    """
    Stand-in for ``safetensors.Slice``.

    ``transformers`` calls ``.get_dtype()`` / ``.get_shape()`` on it and later
    materialises it with ``tensor[...]``. A view into the read-only mapping
    satisfies both, because ``view[...]`` just returns the view again.
    """

    __slots__ = ("_tensor", "_dtype_str")

    def __init__(self, tensor: torch.Tensor, dtype_str: str) -> None:
        self._tensor = tensor
        self._dtype_str = dtype_str

    def get_dtype(self) -> str:
        return self._dtype_str

    def get_shape(self) -> List[int]:
        return list(self._tensor.shape)

    def __getitem__(self, key: Any) -> torch.Tensor:
        return self._tensor[key]

    def __len__(self) -> int:
        return len(self._tensor)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Slice(shape={list(self._tensor.shape)}, dtype={self._dtype_str!r})"


class _RoFile:
    """A ``safe_open``-compatible handle backed by a read-only file mapping."""

    def __init__(self, path: Any, device: Any = "cpu") -> None:
        self.path = os.fspath(path)
        self.device = torch.device(device) if device not in (None, "") else torch.device("cpu")
        self._header, self._data_start = _read_header(self.path)
        self._fh = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        self._index: Dict[str, Tuple[str, Tuple[int, ...], int, int]] = {}
        for name, info in self._header.items():
            if name == "__metadata__":
                continue
            offsets = info["data_offsets"]
            self._index[name] = (
                info["dtype"],
                tuple(info["shape"]),
                int(offsets[0]),
                int(offsets[1]) - int(offsets[0]),
            )
        with _KEEPALIVE_LOCK:
            _KEEPALIVE.append(self)

    # -- safetensors API ---------------------------------------------------
    def keys(self) -> List[str]:
        return list(self._index.keys())

    def metadata(self) -> Optional[Dict[str, str]]:
        return self._header.get("__metadata__")

    def get_tensor_shape(self, name: str) -> List[int]:
        return list(self._index[name][1])

    def _view(self, name: str) -> torch.Tensor:
        dtype_str, shape, offset, nbytes = self._index[name]
        dtype = _torch_dtype(dtype_str)
        elem = torch.tensor([], dtype=dtype).element_size()
        count = nbytes // elem
        with warnings.catch_warnings():
            # The buffer is genuinely read-only; nothing in the load path writes.
            warnings.simplefilter("ignore", UserWarning)
            tensor = torch.frombuffer(
                self._mm,
                dtype=dtype,
                count=count,
                offset=self._data_start + offset,
            )
        return tensor.reshape(shape)

    def get_tensor(self, name: str, device: Any = None) -> torch.Tensor:
        view = self._view(name)
        target = self.device if device is None else torch.device(device)
        if target.type == "cpu":
            return view
        return view.to(target)

    def get_slice(self, name: str) -> _Slice:
        return _Slice(self._view(name), self._index[name][0])

    # -- context-manager protocol -----------------------------------------
    def __enter__(self) -> "_RoFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Deliberately do not unmap: lazily materialised views handed out before
        # this point must stay valid. Mapping/unmapping is cheap anyway.
        return False

    def close(self) -> None:  # pragma: no cover - safety valve
        with _KEEPALIVE_LOCK:
            if self in _KEEPALIVE:
                _KEEPALIVE.remove(self)
        try:
            self._mm.close()
        finally:
            self._fh.close()


def read_safetensors_readonly(path: Any, device: Any = "cpu") -> Dict[str, torch.Tensor]:
    """Convenience helper: read every tensor of a .safetensors file, commit-free."""
    handle = _RoFile(path, device=device)
    return {name: handle.get_tensor(name) for name in handle.keys()}


def _safe_open(filename: Any, framework: str = "pt", device: Any = "cpu", **kwargs: Any) -> Any:
    try:
        return _RoFile(filename, device=device)
    except Exception:  # noqa: BLE001 - fall back to the real thing
        return _ORIGINALS["safe_open"](filename, framework=framework, device=device, **kwargs)


def _load_file(filename: Any, device: Any = "cpu", **kwargs: Any) -> Dict[str, torch.Tensor]:
    try:
        return read_safetensors_readonly(filename, device=device)
    except Exception:  # noqa: BLE001 - fall back to the real thing
        return _ORIGINALS["torch_load_file"](filename, device=device, **kwargs)


def install(force: bool = False) -> bool:
    """
    Patches safetensors to use read-only mappings. Idempotent.

    Returns True if the shim is active.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED and not force:
            return True
        if os.environ.get("RACS2_LOWMEM_SAFETENSORS", "1").strip().lower() in ("0", "false", "no", "off"):
            return False
        try:
            import safetensors
            import safetensors.torch as safetensors_torch
        except ImportError:  # pragma: no cover - safetensors is required in practice
            return False

        _ORIGINALS.setdefault("safe_open", safetensors.safe_open)
        _ORIGINALS.setdefault("torch_load_file", safetensors_torch.load_file)

        safetensors.safe_open = _safe_open
        safetensors_torch.safe_open = _safe_open
        safetensors_torch.load_file = _load_file
        _INSTALLED = True
        return True


def is_installed() -> bool:
    return _INSTALLED


def access_readonly() -> bool:
    """Backwards-friendly alias used in docs/tests."""
    return is_installed()


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import glob
    import sys
    import time

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "huggingface", "hub")
    files = glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True)
    if not files:
        print("No cached .safetensors files found.")
        sys.exit(0)
    target = max(files, key=os.path.getsize)
    print(f"Largest cached checkpoint: {target} ({os.path.getsize(target)/1024**3:.2f} GB)")
    install()
    t0 = time.time()
    tensors = read_safetensors_readonly(target)
    total = sum(t.numel() for t in tensors.values())
    print(f"Read {len(tensors)} tensors / {total/1e6:.1f}M params in {time.time()-t0:.2f}s")
    sample = next(iter(tensors))
    print(f"Sample {sample}: {tuple(tensors[sample].shape)} {tensors[sample].dtype}")
