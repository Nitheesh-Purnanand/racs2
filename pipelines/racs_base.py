"""Phase 21 — RACS-Base: the frozen baseline with only reference acquisition swapped.

``RACSBase`` subclasses :class:`pipelines.agentsemstecom.Baseline` and overrides
exactly one method, ``reference_path``. Everything downstream (EDICT hide, token
perturbation, IPAdapter, SwinJSCC + AWGN, inversion, CodeFormer, metrics) is the
imported baseline code, so a difference in results is attributable to how
``Xref`` was obtained.
"""

from __future__ import annotations

import time

import numpy as np

from pipelines.agentsemstecom import Baseline, run_case  # noqa: F401  (re-exported)
import common.paths as paths
from retrieval.racs import Retriever


class RACSBase(Baseline):
    """Baseline whose Xref is retrieved from the corpus instead of generated."""

    def __init__(self, cfg_all: dict, retriever: Retriever | None = None,
                 acquisitions: dict | None = None):
        super().__init__(cfg_all)
        # ``acquisitions`` maps (image_id, token) -> Retriever.acquire output,
        # computed in a separate process: the CLIP text tower segfaults when
        # it shares a process with the diffusion stack on this machine.
        self._acq: dict = dict(acquisitions or {})
        self.retriever = retriever or (None if acquisitions else Retriever(cfg_all))
        #: True routes reference_path to the Phase 8 generated Xref, so the
        #: baseline arm runs through this very object and code path.
        self.generated = False

    def feature(self, image_id: str):
        with np.load(paths.RESULTS_DIR / "public_conditions" / f"{image_id}.npz",
                     allow_pickle=True) as z:
            return z["feature"] if "feature" in z.files else None

    def acquire(self, image_id: str, token: str) -> dict:
        """Public inputs only: Kpub, Xfeat and the shared token."""
        key = (image_id, token)
        if key not in self._acq:
            t0 = time.perf_counter()
            out = self.retriever.acquire(self.conditions(image_id)["kpub"],
                                         self.feature(image_id), token)
            out["latency_s"] = time.perf_counter() - t0
            self._acq[key] = out
        return self._acq[key]

    def reference_path(self, role: str, class_name: str, image_id: str) -> str:
        if self.generated:
            return super().reference_path(role, class_name, image_id)
        return self.acquire(image_id, self.tokens(image_id)["token"])["path"]
