# Phase 20: Structure-Aware Reranking

> **Goal**: Rank the Phase 17 Top-K candidates by a frozen combination of semantic and structural
> similarity, choose one weighting on calibration data, and record whether it helps, does
> nothing, or hurts.
> **Why**: Pure CLIP retrieval is the baseline design, not the final one. Phase 19 proved the two
> signals can disagree; this phase decides how much to trust each and freezes the decision.

> **Precondition**: Phases 17, 18 and 19 are frozen and their verify scripts pass.

> **Data rule**: tune `alpha`/`beta` and K on the **calibration** split only. The evaluation split
> is opened for the first time in Phase 21. Do not look at evaluation PSNR here.

> **No VLM, no generation, no editing.** This phase is ranking only.

---

## Step 0: Pre-flight and the conditioning sanity check

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase17_verify.py
.venv\Scripts\python.exe scripts\phase18_verify.py
.venv\Scripts\python.exe scripts\phase19_verify.py
```

Then answer one question before any A/B design:

> **Does the downstream hiding path actually consume the reference pixels?**

Run this probe with the exact Phase 15 downstream configuration you intend to use in Phase 21:

```python
# same secret, same prompt, same token, two different reference images
h_a = baseline.hide(secret_px, ref_a, prompt, token)
h_b = baseline.hide(secret_px, ref_b, prompt, token)
print(tensor_hash(h_a["stego_pixels"]) == tensor_hash(h_b["stego_pixels"]))
```

Interpretation:

- `False` (hashes differ): reference pixels influence the stego. The A/B can measure a real
  reference-quality effect. Continue.
- `True` (hashes equal): the downstream ignores reference pixels for this configuration. In the
  frozen baseline this happens because IP-Adapter is disabled and EDICT is text-conditioned
  only (`steganography/hide.py` uses `reference_pixels` only through optional image embeds).
  Then reranking **cannot** change recovery quality, and any claimed improvement would be an
  artifact. Stop and choose one:

  1. **Preferred**: enable the image-conditioned path (`steganography.ipadapter.enabled: true`
     with the pinned IP-Adapter weights), verify reference sensitivity, and rerun the baseline
     under the same conditioning before Phase 21. This is a baseline component correction, not a
     RACS change; record it in `docs/DEVIATIONS.md` and re-freeze Phases 15/16 as a versioned
     `baseline v2`.
  2. **Acceptable negative result**: keep the frozen baseline and report Phase 20 as an offline
     ranking study only. Phase 21 then has no pixel-level quality claim, and Phase 22 must say so.
     Do not silently compare an image-conditioned RACS against a text-only baseline.

Record the probe output (two hashes and the boolean) in `phase20_rerank.json`. This one check
prevents the most likely invalid conclusion in the whole project.

---

## Step 1: Config

Append under `retrieval:`:

```yaml
retrieval:
  # ... existing keys stay unchanged ...
  ranking:
    alpha: 0.5                  # semantic weight, frozen after the sweep
    beta: 0.5                   # structural weight, alpha + beta == 1
    top_k: 10                   # must equal faiss.default_top_k and token_selection.top_k
    semantic_transform: "cosine_to_unit"     # (cos + 1) / 2
    structural_transform: "identity"         # Phase 19 primary already in [0,1]
    tie_break: "semantic_then_row"
    tune_split: "calibration"
    sweep_alpha: [0.0, 0.25, 0.5, 0.75, 1.0]
    freeze: false               # set true after the gate
```

`alpha + beta = 1` is mandatory. Do not use unnormalised weights; they make the sweep
incomparable.

---

## Step 2: Structure

```text
retrieval/
  rerank.py
scripts/
  phase20_rerank.py     # sweep + local A/B + grid
  phase20_verify.py
experiments/results/
  phase20_ranking_sweep.json
  phase20_ranking_ab.json
  phase20_rerank_grid.png
  phase20_verify.json
```

---

## Step 3: Implement `retrieval/rerank.py`

```python
def rank_candidates(
    *,
    candidates: Sequence[dict],          # Phase 17 output, best-first
    query_embedding: np.ndarray,         # float32, L2-normalised (unused if score present)
    query_structural: np.ndarray,        # Phase 19 map for the query image
    structural_maps: np.ndarray,         # corpus cache aligned to image_ids
    image_ids: Sequence[str],
    alpha: float,
    beta: float,
    semantic_transform: str = "cosine_to_unit",
    structural_transform: str = "identity",
) -> list[dict]:
    """Return candidates re-scored and re-sorted with the composite score."""
```

Rules:

- Validate `alpha >= 0`, `beta >= 0`, `abs(alpha + beta - 1) <= 1e-9`.
- Semantic term:
  - use the candidate's Phase 17 `score` (cosine, because embeddings are L2-normalised).
  - `cosine_to_unit`: `s = (score + 1) / 2`, clipped to [0,1].
  - The transform is fixed, not per-list min-max. Changing K must not change an individual
    candidate's score.
- Structural term:
  - `structural_similarity(query_structural, map[image_id])["primary"]`, already in [0,1].
  - If the structural kind is disabled for a class, use `beta = 0` for that class and record it.
- Composite: `composite = alpha * semantic + beta * structural`.
- Sort by `(-composite, -semantic, row)` with a stable sort. The tie-break is frozen so Tx and Rx
  agree even on exact ties.
- Return every input candidate with `rank`, `image_id`, `score` (original CLIP score),
  `semantic`, `structural`, `composite`, `row`. Do not drop candidates.
- Keep the original Phase 17 rank as `retrieval_rank` for diagnostics.

Phase 21 uses exactly one function call:

```python
selected = token_select(token, rank_candidates(...))
```

There must be no second ranking implementation in the pipeline.

---

## Step 4: Implement `scripts/phase20_rerank.py`

### 4a. Offline sweep (calibration only)

```text
1. set_seed(42)
2. load calibration split records (role=calibration); load the frozen index and caches
3. for each calibration image (66 total):
     a. CLIP-embed the secret image (float32, frozen Phase 5 preprocessing)
     b. retrieve Top-K with K = [1, 5, 10, 20, 50]
     c. compute structural map for the secret
     d. for each alpha in sweep_alpha:
          rerank candidates, apply token selection to the ordered list
          record: selected id, retrieval rank, semantic, structural, composite
4. per (K, alpha) aggregate:
     - selected_structural_mean
     - selected_semantic_mean
     - top1_rank_change_rate
     - selected_changed_vs_clip_rate
     - mean_retrieval_rank_of_selected
5. choose alpha by the pre-registered rule below
6. save phase20_ranking_sweep.json and phase20_rerank_grid.png
```

Pre-registered choice rule (write it before looking at numbers):

```text
Maximise mean selected structural similarity, subject to
mean selected semantic cosine >= 0.95 * mean semantic cosine at alpha = 1.0
(CLIP-only), with K fixed at the value used by Phase 21.
If no alpha satisfies the constraint, choose alpha = 1.0 (CLIP-only) and report reranking as
not adopted. A negative result is valid.
```

K selection: prefer `K = 10` if the sweep does not show a clear, large gain. Larger K gives the
token more choices but also more room for a structurally wrong candidate. Record the decision.

Also report the trivial cases: `alpha = 0` (structure only) and `alpha = 1` (CLIP only), so the
reader can see both endpoints.

### 4b. Local A/B (only if the Step 0 sensitivity probe passed)

For a deterministic probe of `ab_images_per_class` calibration images (default 11, seed 42):

```text
CLIP-only reference   = token-selected candidate at alpha = 1.0
reranked reference    = token-selected candidate at the chosen alpha
for each reference:
  hidden = baseline.hide(secret_px, ref_px, prompt, token)
  recovered = baseline.receive(
      hidden["stego_pixels"], token_mode="correct", token=token,
      wrong_token=wrong_token, prompt=prompt,
      blob={"delta": hidden["delta"], "noise": hidden["noise"],
            "eta": hidden["eta"], "steps": hidden["steps"],
            "mixing": hidden["mixing"]},
      image_embeds=hidden.get("image_embeds"),
  )                                            # local, no channel
  metrics = mse/psnr/ssim/lpips(secret, recovered)
```

Both arms use the same secret, prompt, token, downstream config and seed. The only difference is
the reference image. Report paired deltas per image and the mean/median, not just the mean.

If the sensitivity probe failed, skip 4b, set `"ab_skipped_reason": "reference pixels not
consumed by downstream"`, and do not report a quality claim.

### Output schema (`phase20_ranking_sweep.json`)

```json
{
  "schema_version": 1,
  "phase": 20,
  "tune_split": "calibration",
  "split_version": "...",
  "index_sha256": "...",
  "structural_maps_sha256": "...",
  "top_k": 10,
  "sweep_alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
  "constraint": "selected_semantic >= 0.95 * clip_only",
  "cells": [
    {"k": 10, "alpha": 0.5, "beta": 0.5,
     "selected_structural_mean": 0.0, "selected_semantic_mean": 0.0,
     "clip_only_semantic_mean": 0.0, "selected_changed_rate": 0.0,
     "top1_rank_change_rate": 0.0, "mean_retrieval_rank_of_selected": 0.0}
  ],
  "chosen": {"k": 10, "alpha": 0.5, "beta": 0.5, "reason": "..."},
  "adopted": true,
  "reference_sensitivity": {"ref_a_sha256": "...", "ref_b_sha256": "...", "sensitive": true},
  "grid_path": "experiments/results/phase20_rerank_grid.png"
}
```

`phase20_ranking_ab.json` carries the paired local metrics, or the skip reason.

---

## Step 5: Implement `scripts/phase20_verify.py`

Checks:

```text
[ ] alpha >= 0, beta >= 0, alpha + beta == 1 within 1e-9
[ ] config ranking.top_k == faiss.default_top_k == token_selection.top_k
[ ] sweep contains alpha 0.0 and 1.0 endpoints and every configured value
[ ] every cell used K candidates and returned exactly K
[ ] chosen alpha obeys the pre-registered constraint; if not, alpha == 1.0 and adopted == false
[ ] tie-break produces a deterministic order on a synthetic tie
[ ] composite score recomputes exactly from semantic, structural, alpha, beta
[ ] no evaluation-split record appears in the sweep, A/B, or grid
[ ] phases 17/18/19 artifact hashes match
[ ] reference-sensitivity probe result is recorded
[ ] if sensitive is false: adopted == false, ab skipped, no quality claim in the JSON
[ ] phase20_rerank_grid.png exists
```

Exit non-zero on failure. A reranking gain computed on evaluation data invalidates the phase.

---

## Step 6: Freeze the choice

After the gate:

1. Set `retrieval.ranking.alpha`, `beta`, `top_k`, `freeze: true` in `configs/base.yaml`.
2. Record the config hash, chosen values, tuning split, sweep artifact hashes and the
   sensitivity result in `phase20_ranking_sweep.json`.
3. Update `docs/DEVIATIONS.md` if the structural kind is not the paper's (for example edges
   instead of segmentation/landmarks, or no IP-Adapter conditioning).
4. Do not change alpha/beta again without rerunning Phases 20-22 and saying so.

---

## Step 7: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase20_rerank.py --sweep
.venv\Scripts\python.exe scripts\phase20_rerank.py --ab
.venv\Scripts\python.exe scripts\phase20_verify.py
```

Expected summary (positive case):

```text
tune split        : calibration, 66 images
K                 : 10
chosen alpha/beta : reported with the constraint check
selected changed  : reported percent
local A/B         : paired PSNR delta reported, positive or negative
sensitivity       : true (reference pixels affect stego)
verify            : PASS
```

Expected summary (honest negative case):

```text
sensitivity       : false
adopted           : false, alpha = 1.0
A/B               : skipped with a reason
result            : reranking not adopted; offline rank study only
verify            : PASS
```

Both outcomes pass the gate. Inventing a gain does not.

---

## Checklist

```text
[ ] composite score = alpha * semantic + beta * structural, alpha + beta = 1
[ ] semantic transform fixed, not per-list min-max
[ ] structural score from the frozen Phase 19 cache only
[ ] stable tie-break (composite, semantic, row)
[ ] alpha/beta/K tuned on calibration only
[ ] pre-registered choice rule written before seeing results
[ ] reference-sensitivity probe run and recorded
[ ] local A/B paired by image, same secret/prompt/token/downstream
[ ] if insensitive, adopted = false and no quality claim
[ ] chosen weights frozen in config with freeze: true
[ ] DEVIATIONS updated for the structural kind/conditioning
[ ] no evaluation split touched
[ ] phase20_ranking_sweep.json, phase20_ranking_ab.json, phase20_verify.json written
[ ] no VLM, no editing, no generation added
```

**When this passes, the ranking protocol is frozen.** Phase 21 may use it but may not retune it.

---

## The Gate Question (answer in your own words)

> *How do you decide whether structural reranking helps, without tuning on the test set, and why
> is "it does not help" a valid Phase 20 result?*

Answer with your artifacts:

- The sweep runs on calibration queries only. The choice rule is fixed in advance: improve
  structural similarity while keeping semantic similarity within 5 percent of CLIP-only.
- A paired local A/B compares the two selected references through the same downstream, so the
  delta is attributable to the reference, not to the ranker's score.
- If the downstream ignores reference pixels, a quality delta is impossible by construction.
  Reporting that is more valuable than manufacturing a number.
- A negative result still leaves a measured rank study: how often the selection changed, how much
  structure improved and how much semantics was lost.
- Freezing alpha/beta/K before Phase 21 is what makes the later comparison credible.

---

## Report back

- Sweep table for every K and alpha, with the constraint column
- Chosen alpha/beta/K and the exact rule that selected them
- Reference-sensitivity probe result and both hashes
- Paired local A/B deltas (or the documented skip reason)
- Percent of selections that changed vs CLIP-only
- Grid path and what it shows
- DEVIATIONS changes
- Your answer to the gate question

---

## Notes and risks

- **The most likely failure is a no-op downstream.** Run the sensitivity probe first; everything
  else is secondary.
- **Do not tune alpha on evaluation data.** That would make Phase 22 circular.
- **Keep K fixed across Tx/Rx and across systems.** A different K changes both the token
  selection and the candidate pool.
- **Structural similarity is not quality.** A structurally similar but semantically wrong
  reference can be worse. The semantic constraint exists for that reason.
- **Score transforms matter.** Per-list min-max makes a candidate's score depend on K; use the
  fixed transforms in config.
- **A flat sweep is informative.** If every alpha gives the same selection, say so and adopt
  CLIP-only rather than presenting a cosmetic weight.
- **No VLM in this phase.** Deterministic scoring only; RACS-Agent is a later phase.
