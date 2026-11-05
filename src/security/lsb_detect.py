"""Classical LSB steganalysis: Westfeld chi-square and Sample Pair Analysis (SPA).

Both are the standard structural detectors for LSB *replacement*; Contribution 2
measures how the texture-adaptive +-1 layer fares against them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def chi_square_p(arr: np.ndarray) -> float:
    """Westfeld-Pfitzmann: p close to 1 means pairs of values (2k, 2k+1) are equalised."""
    h = np.bincount(arr.reshape(-1), minlength=256).astype(np.float64)
    even, odd = h[0::2], h[1::2]
    exp = (even + odd) / 2.0
    keep = exp > 4
    stat = float((((even - exp) ** 2)[keep] / exp[keep]).sum())
    return float(1.0 - chi2.cdf(stat, df=max(int(keep.sum()) - 1, 1)))


def spa_rate(arr: np.ndarray) -> float:
    """Dumitrescu-Wu-Wang sample pair analysis: estimated LSB-replacement rate.

    Mean over the three channels, horizontal and vertical pairs.
    """
    est = []
    for c in range(arr.shape[2]):
        ch = arr[:, :, c].astype(np.int32)
        for u, v in ((ch[:, :-1], ch[:, 1:]), (ch[:-1, :], ch[1:, :])):
            u, v = u.reshape(-1), v.reshape(-1)
            ve = (v % 2) == 0
            x = np.count_nonzero((ve & (u < v)) | (~ve & (u > v)))
            y = np.count_nonzero((ve & (u > v)) | (~ve & (u < v)))
            k = np.count_nonzero(np.round(v / 2) == np.round(u / 2))
            if k == 0:
                continue
            qa, qb, qc = 2.0 * k, 2.0 * (2 * x - u.size), float(y - x)
            disc = max(qb * qb - 4 * qa * qc, 0.0)
            est.append(min((-qb + np.sqrt(disc)) / (2 * qa), (-qb - np.sqrt(disc)) / (2 * qa)))
    return float(np.clip(np.mean(est), 0.0, 1.0))


def auc(neg: list, pos: list) -> float:
    """Probability a stego scores higher than a cover (0.5 = undetectable)."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)
