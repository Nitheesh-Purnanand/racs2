# Phase 6: Evaluation Harness & Class-Balanced Test Set

> **Goal**: Build the measurement rig that every later phase reports through, and freeze two
> disjoint, role-tagged image sets: a read-only UniStega **calibration** set used only to
> check approximate reproduction of the paper, and our own **evaluation** set used for every
> claim.
> **Why**: Phase 15 must be able to answer "did we land near Table II?" without contaminating
> the data we later use to compare AgentSemSteCom against RACS. If the metrics, classes, SNRs,
> or split roles are improvised later, the baseline and the comparison stop being
> comparable.

> **Ordering rule**: the paper's dataset is a **calibration anchor only**. It never enters an
> evaluation run, the RACS retrieval corpus, the security matrix, or any headline table.

---

## Step 0: Pre-flight checks (do these first)

No model download in this phase, but LPIPS pulls torchvision weights (~500 MB across
AlexNet/VGG depending on the backbone). Confirm disk, connectivity, and that Phase 5 is
still intact:

```powershell
cd d:\racs2

# space on D: (C: is full; keep caches on D:)
Get-PSDrive D | Select-Object Used,Free

# HuggingFace reachable? (expect HTTP 200)
try { (Invoke-WebRequest -UseBasicParsing https://huggingface.co -TimeoutSec 20).StatusCode } catch { "UNREACHABLE: $($_.Exception.Message)" }

# Phase 5 artifacts present and untouched
Get-ChildItem experiments\results\phase5_corpus_embeddings.* | Select-Object Name,Length
.venv\Scripts\python.exe -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Current state: Phases 1-5 are complete. `data/retrieval_images/` holds 1,000 PNGs,
`data/metadata/image_index.json` is valid, and `phase5_corpus_embeddings.npz` holds
1,000 x 512 float32 vectors. **There is no test set, no LPIPS, and no role enforcement yet.**
That is what this phase produces.

---

## Step 1: Dependencies

```powershell
$env:TMP = "D:\racs2\temp"; $env:TEMP = "D:\racs2\temp"
.venv\Scripts\python.exe -m pip install --no-cache-dir lpips scipy scikit-image
```

Verify:

```powershell
.venv\Scripts\python.exe -c "import lpips, scipy, skimage; print('lpips', lpips.__version__, 'scipy', scipy.__version__, 'skimage', skimage.__version__)"
```

Notes:
- `lpips` imports `scipy`, so installing it directly avoids a confusing partial install.
- `scikit-image` is used only as an independent SSIM sanity check. The project's
  `evaluation/metrics.ssim` stays authoritative so Phase 2 and Phase 15 report the same
  metric.
- LPIPS downloads its linear weights on first use. They land in `TORCH_HOME`, which
  `common/paths.py` already points at `D:\racs2\cache\torch`.

> **Do not** benchmark or tune LPIPS backbones to make numbers look better. Pick one,
> record it, and never change it mid-experiment.

---

## Step 2: Update `configs/base.yaml`

Add an `evaluation` block:

```yaml
evaluation:
  splits_path: "data/metadata/test_split.json"
  tokens_path: "data/metadata/tokens.json"
  calibration_dir: "data/calibration/unistega"
  evaluation_dir: "data/test_set"
  image_size: 512
  classes: ["common", "facial", "style"]
  per_class: 25                      # 25 x 3 = 75 minimum; freeze 100 if sources allow
  total: 100
  snr_db: [5, 10, 15, 20]
  metrics: ["mse", "psnr", "ssim", "lpips"]
  lpips_net: "alex"                  # paper does not state one; freeze this choice
  lpips_dtype: "float32"
  lpips_batch_size: 1                # keep deterministic and memory-safe on 6 GB
  report_percentiles: [5, 25, 50, 75, 95]
```

**Protocol decision**: `classes`, `snr_db`, `metrics`, `lpips_net`, and the split version are
frozen once the artifacts are written. Changing any of them invalidates every stored result.

---

## Step 3: Create the Phase 6 structure

```
d:\racs2\
├── data\
│   ├── calibration\
│   │   └── unistega\
│   │       ├── common\   *.png      # UniStega-Content
│   │       ├── facial\   *.png      # UniStega-Similar face subset
│   │       └── style\    *.png      # UniStega-Style
│   ├── test_set\
│   │   ├── common\       *.png      # our evaluation set, COCO-based, disjoint
│   │   ├── facial\       *.png
│   │   └── style\        *.png
│   └── metadata\
│       ├── test_split.json          # frozen split manifest, role-tagged
│       └── tokens.json              # fixed token + wrong token per image
├── evaluation\
│   ├── metrics.py                   # extended: lpips, distributions
│   ├── splits.py                    # manifest loader + role enforcement
│   └── report.py                    # distribution aggregation to JSON
├── scripts\
│   ├── phase6_build_testset.py      # build/canonicalise/freeze both splits
│   ├── phase6_eval_selftest.py      # metrics + role + determinism gate
│   └── eval_report.py               # reusable system x class x SNR driver
├── docs\
│   └── DEVIATIONS.md                # created here, appended forever after
└── experiments\
    └── results\
        └── phase6_selftest.json     # gate evidence
```

Add the image folders to `.gitignore` (manifests stay in git, pixels do not):

```gitignore
data/calibration/
data/test_set/
```

---

## Step 4: Extend `evaluation/metrics.py`

Keep every existing function and signature. `all_metrics` must keep its current behaviour so
Phases 2-3 stay reproducible.

Add:

```python
def lpips_metric(a, b, net: str = "alex", device=None, batch_size: int = 1) -> float:
    """LPIPS between unit-range [C,H,W] or [N,C,H,W] tensors. Lower is better."""

def all_metrics(a, b, data_range: float = 1.0, with_lpips: bool = False,
                lpips_net: str = "alex") -> dict: ...

def distribution(values, percentiles=(5, 25, 50, 75, 95)) -> dict: ...
```

**Implementation requirements that matter:**

1. **Range conversion is the classic bug.** The project stores images in `[0, 1]`; LPIPS
   expects `[-1, 1]`. Convert with `x * 2 - 1`, and assert the input range before
   converting. Do not silently `clamp`.
2. **PSNR infinity.** Identical images give `psnr = inf`. `distribution()` must report
   `n`, `n_inf`, and compute `mean/median/std` over the finite values only, while also
   reporting `all_finite_mean` when `n_inf == 0`.
3. **SSIM stays the project implementation.** Use `scikit-image` only for the self-test
   cross-check, never as the reported metric.
4. **LPIPS is stateless and deterministic.** Do not shuffle batches, do not use multiple
   workers, and pass `lpips_batch_size: 1` by default. Record the backbone and the resolved
   weight hash in the report metadata.
5. **No GPU requirement.** LPIPS may run on CPU if VRAM is tight. Record the device used.
6. **`distribution()` output shape**:

```python
{
  "n": 25, "n_inf": 0,
  "mean": 24.13, "median": 24.02, "std": 1.87,
  "min": 20.91, "max": 27.44,
  "p05": 21.30, "p25": 23.10, "p50": 24.02, "p75": 25.20, "p95": 26.90,
  "mean_finite": 24.13
}
```

---

## Step 5: Write `evaluation/splits.py`

The manifest is the contract. Every consumer (Phases 15, 16, 21, 22, 26) goes through this
module, so role enforcement cannot be forgotten.

**Schema for `data/metadata/test_split.json`:**

```json
{
  "schema_version": 1,
  "split_version": "<sha256 over the ordered (role, class, image_id, sha256) tuples>",
  "created_utc": "2026-10-23T00:00:00Z",
  "counts": {"calibration": 100, "evaluation": 100},
  "classes": ["common", "facial", "style"],
  "records": [
    {
      "image_id": "cal_common_0001",
      "role": "calibration",
      "class": "common",
      "path": "data/calibration/unistega/common/cal_common_0001.png",
      "sha256": "<sha256 of the canonical PNG bytes>",
      "width": 512,
      "height": 512,
      "mode": "RGB",
      "source": "unistega-content",
      "license": "research-use, see source",
      "source_ref": "<paper reference [46] or repository tag>"
    }
  ]
}
```

**API:**

```python
class Split:
    @classmethod
    def load(cls, role: str | None = None) -> "Split": ...
    def __len__(self) -> int: ...
    def ids(self, cls: str | None = None) -> list[str]: ...
    def records(self, cls: str | None = None) -> list[dict]: ...
    def get(self, image_id: str) -> dict: ...
    @property
    def version(self) -> str: ...
    def assert_disjoint(self, other: "Split") -> None: ...
```

**Rules:**

- `Split.load(role="evaluation")` returns only evaluation records.
- `Split.load(role="calibration")` requires an explicit `allow_calibration=True` flag, so a
  calibration run is always a deliberate act.
- `get()` on a record whose role does not match the requested role raises `RoleError`.
- The loader verifies every file exists and its sha256 matches at load time. Fail fast.
- `assert_disjoint()` checks both path and sha256, against another split and against
  `data/metadata/image_index.json`.

**Tokens schema for `data/metadata/tokens.json`:**

```json
{
  "schema_version": 1,
  "tokens_version": "<sha256 over the ordered (image_id, token, wrong_token) tuples>",
  "records": [
    {"image_id": "cal_common_0001", "token": "84170233", "wrong_token": "10599347"}
  ]
}
```

Rules: one record per image in **both** splits; `token` is a decimal digit string of fixed
length (8 is enough); `wrong_token` is generated deterministically and must differ from
`token`; no duplicates within or across splits. The "no token" condition is represented by
absence, not by a sentinel value.

---

## Step 6: Write `evaluation/report.py`

One function that turns per-case results into the frozen artifact schema.

```python
def build_report(
    system: str,
    role: str,                    # "calibration" | "evaluation"
    split: Split,
    results: list[dict],          # one dict per case
    *,
    approximate: bool,            # True only for role="calibration"
    meta: dict | None = None,
) -> dict: ...
```

Each case result:

```python
{
  "image_id": "...", "class": "common", "snr_db": 10,
  "mse": 0.0121, "psnr": 19.4, "ssim": 0.552, "lpips": 0.311,
  "reference_latency_s": 1.2, "total_latency_s": 2.1, "peak_vram_gb": 3.4
}
```

Output artifact:

```json
{
  "schema_version": 1,
  "system": "agentsemstecom",
  "role": "evaluation",
  "approximate": false,
  "split_version": "<hash>",
  "corpus_version": "<hash>",
  "config_hash": "<hash of the resolved evaluation+generation config>",
  "created_utc": "...",
  "cells": [
    {
      "class": "common", "snr_db": 10, "n": 25,
      "metrics": {
        "mse":  {"n": 25, "n_inf": 0, "mean": 0.0121, "..." : "..."},
        "psnr": {"n": 25, "n_inf": 0, "mean": 19.4, "..." : "..."},
        "ssim": {"n": 25, "n_inf": 0, "mean": 0.552, "..." : "..."},
        "lpips":{"n": 25, "n_inf": 0, "mean": 0.311, "..." : "..."}
      }
    }
  ]
}
```

**Hard rules:**
- `role="evaluation"` always writes `"approximate": false`. `role="calibration"` always
  writes `"approximate": true`. The function asserts it.
- A report must never mix roles or classes within a cell.
- Missing metrics are omitted, not zero-filled.
- Write atomically (`.tmp` then `os.replace`).

---

## Step 7: Write `scripts/phase6_build_testset.py`

Builds both splits, canonicalises pixels, and freezes the manifests.

**Sourcing, in priority order:**

| Role | Class | Preferred source | Fallback, must be logged |
| :-- | :-- | :-- | :-- |
| calibration | common | UniStega-Content (paper reference [46]) | documented surrogate |
| calibration | facial | UniStega-Similar face subset | documented surrogate |
| calibration | style | UniStega-Style | documented surrogate |
| evaluation | common | COCO val2017, **disjoint** from `data/retrieval_images/` | any openly licensed natural-image set |
| evaluation | facial | an openly licensed face set (for example FFHQ or CelebA-HQ) | documented alternative |
| evaluation | style | an openly licensed artwork set (for example WikiArt public domain) | documented alternative |

Locate UniStega through the paper's reference rather than guessing a URL. If the official
release is gated or unavailable, do not substitute silently: write the surrogate construction
into `docs/DEVIATIONS.md` in the same commit that builds the set.

**Build algorithm:**

```
1. import common.paths; set_seed(42, verbose=True)
2. load configs/base.yaml -> evaluation block
3. for each (role, class):
     a. collect candidate source images
     b. if role == evaluation, exclude any sha256 already present in the retrieval corpus
     c. deterministic sample: sort candidates by (source, filename), seed 42, take per_class
     d. canonicalise: open RGB -> resize shortest side to 512 (bicubic) ->
        center crop 512 -> save PNG (no metadata chunks)
     e. record sha256, width, height, mode, source, license, source_ref
4. assert disjointness:
     calibration x evaluation = empty
     calibration x retrieval = empty
     evaluation x retrieval = empty
     (compare sha256 sets, not just paths)
5. assert class balance: each role has >= 20 per class
6. build split_version = sha256 over ordered tuples
7. build tokens deterministically: seed = int(sha256(image_id)[:16], 16) mod 10**8
     token = zero-padded 8 digits
     wrong_token = (token + 1 + (seed // 10) % 9999999) mod 10**8, zero-padded
     assert token != wrong_token
8. write test_split.json and tokens.json atomically
9. print a summary table: role x class counts, bytes on disk, split_version, tokens_version
10. write experiments/results/phase6_build_summary.json
```

**Rules that matter:**
- Canonicalisation is part of the protocol. Both systems and both roles use the same resize
  and crop. Record it in the manifest (`"canonical": "shortest_side_512_bicubic -> center_crop_512"`).
- Never re-sample an existing split. If a source fails halfway, delete the partial folder and
  rebuild from scratch so the deterministic order is preserved.
- Do not add augmentation, EXIF rotation, or auto-contrast.
- Keep image bytes free of timestamps; PNG is written without ancillary chunks.

---

## Step 8: Write `scripts/phase6_eval_selftest.py`

The gate script. It must run offline and finish in under a minute.

```
1. METRICS
   a. identical image -> mse 0, psnr inf, ssim 1.0, lpips <= 1e-6
   b. mild Gaussian noise -> psnr finite in (20, 45), 0 < lpips < 0.5
   c. structural change (rotate 90 degrees + block shuffle) -> ssim < 0.6, lpips > 0.15
   d. cross-check project ssim against skimage ssim on the same pair (within 0.02)
   e. distribution([1,2,3,4,5]) -> mean 3, median 3, n 5
   f. distribution([1.0, inf, 3.0]) -> n 3, n_inf 1, mean over finite == 2.0
2. SPLITS
   a. both roles load; counts match config
   b. every file exists; every sha256 matches
   c. split_version recomputes to the stored value
   d. calibration x evaluation is disjoint (sha256 set intersection empty)
   e. both are disjoint from data/retrieval_images via image_index.json sha256
   f. each role has >= 20 images per class
3. ROLE ENFORCEMENT
   a. Split.load(role="evaluation") contains no calibration record
   b. requesting a calibration image through the evaluation role raises RoleError
   c. Split.load(role="calibration") without allow_calibration raises
4. TOKENS
   a. one record per image in each split
   b. tokens unique, wrong tokens unique, token != wrong_token for all
   c. tokens_version recomputes
5. REPORT SHAPE
   a. build a synthetic result list with known values
   b. build_report(role="evaluation") -> approximate is False
   c. build_report(role="calibration") -> approximate is True
   d. a mixed-role cell raises
6. Write phase6_selftest.json with pass/fail per check and exit non-zero on any failure.
```

---

## Step 9: Write `scripts/eval_report.py`

The single driver used by Phases 15, 16, 21, 22 and 26. It must be system-agnostic.

```
usage:
  python scripts/eval_report.py \
      --system agentsemstecom \
      --split evaluation \
      --runner pipelines.agentsemstecom:run_case \
      --out experiments/results/baseline_table.json
```

**Runner contract** (implemented by the phase that has a pipeline; Phase 6 ships a dummy):

```python
def run_case(record: dict, snr_db: int, role: str) -> dict:
    """Return at least {"secret": PIL.Image, "recovered": PIL.Image,
    optional "stego", "reference_latency_s", "total_latency_s", "peak_vram_gb"}."""
```

Driver behaviour:

```
1. Split.load(role=...)
2. for each class, each image, each snr_db: call the runner
3. compute metrics on [0,1] tensors: mse, psnr, ssim, lpips
4. aggregate with evaluation.report.build_report
5. write the JSON atomically; print a compact table
6. refuse to run: role="calibration" unless --approximate is passed, and any result whose
   image_id is not in the requested split
```

Ship `scripts/dummy_runner.py` with a `run_case` that returns the secret plus mild noise, so
Phase 6 can prove the driver end to end before any pipeline exists.

---

## Step 10: Create `docs/DEVIATIONS.md`

```markdown
# RACS2: Deviation Register

Every place the reimplementation cannot match the paper exactly is recorded here.
No entry, no claim.

| Phase | Paper component | What we used | Why | Measured impact | Accepted |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 6 | UniStega (paper reference [46]) | <exact source or surrogate> | <gated / license / size> | <n/a for data> | yes/no |
```

Append-only. Phase 15 adds entries for the reference generator, ControlNet, and codec if
they differ.

---

## Step 11: Run and Check

```powershell
cd d:\racs2
.venv\Scripts\python.exe scripts\phase6_build_testset.py
.venv\Scripts\python.exe scripts\phase6_eval_selftest.py
.venv\Scripts\python.exe scripts\eval_report.py --system dummy --split evaluation `
    --runner scripts.dummy_runner:run_case --out experiments/results/phase6_dummy_eval.json
```

Expect roughly 86 MB of PNGs for 200 images at 512x512. The selftest should take seconds;
LPIPS on a handful of pairs is the slowest part.

### ✅ Checklist (all must pass):

```
[ ] lpips, scipy, scikit-image import in the venv
[ ] evaluation.metrics keeps mse/psnr/ssim signatures; all_metrics is backward compatible
[ ] lpips_metric converts [0,1] -> [-1,1] and is correct on identical and noisy pairs
[ ] distribution() handles inf PSNR correctly (n_inf, finite-only mean)
[ ] data/calibration/unistega/{common,facial,style} populated, >= 20 each
[ ] data/test_set/{common,facial,style} populated, >= 20 each, sourced independently
[ ] Calibration and evaluation are disjoint by sha256
[ ] Both splits are disjoint from data/retrieval_images via image_index.json
[ ] test_split.json schema valid; split_version recomputes exactly
[ ] tokens.json: one token and one wrong_token per image in both splits, all distinct
[ ] Split.load(role="evaluation") cannot return a calibration record
[ ] Split.load(role="calibration") requires allow_calibration=True
[ ] build_report marks calibration "approximate": true and evaluation false
[ ] eval_report.py runs end to end on the dummy runner and writes a well-formed table
[ ] docs/DEVIATIONS.md exists with at least the dataset-sourcing entry
[ ] .gitignore excludes data/calibration/ and data/test_set/
[ ] No model, no pipeline, no retrieval, and no FAISS work in this phase
```

**When all checks pass, Phase 6 is done.** Then we move to **Phase 7: Public Conditions
(`Kpub` + `Xfeat`)**, which is the first AgentSemSteCom component.

---

## The Gate Question (answer in your own words)

> *Why must the evaluation harness and the frozen test splits exist before the baseline is
> built, and why is the paper's dataset only a calibration reference?*

Be ready to answer with your own artifacts:

- A metric is only meaningful if it is computed the same way on every system. If the LPIPS
  backbone, image range convention, or SSIM implementation changes between the baseline run
  and the RACS run, the comparison measures the metric change, not the system change.
- The baseline is the control condition. A control measured on the test set that later
  evaluates the treatment is not a control, it is a rehearsal.
- The paper's dataset is used at most to answer "are we in the right neighbourhood of
  Table II?". It cannot answer "is RACS better than AgentSemSteCom?", because that question
  is about our implementation on our data, under our frozen protocol.
- Concretely: if a calibration image ever appears in an evaluation cell, the phase is
  invalid, because the reported number is partly a memory of the paper rather than a
  measurement.

---

## Report back

- What I built / files added
- Dataset sources per role and class, with licence notes, and whether `docs/DEVIATIONS.md`
  gained an entry
- `split_version` and `tokens_version`
- Counts per role and class, bytes on disk
- Metric self-test numbers: identical, noisy, structurally changed
- Cross-check of project SSIM against scikit-image
- What surprised me / what is unclear
- My answer to the gate question

---

## Notes / risks

- **LPIPS range bug**: LPIPS expects `[-1, 1]`. Passing `[0, 1]` silently produces plausible
  but wrong numbers. Assert the input range before conversion.
- **PSNR infinity**: identical images give `inf`. Never let `inf` poison a mean.
- **Role leakage is the failure mode this phase exists to prevent.** Enforce it in the loader,
  not in each script, and test the failure path, not just the success path.
- **Do not sample the evaluation `common` class from `data/retrieval_images/`.** RACS would
  then be retrieving from a corpus that contains the secret image itself. Disjointness is by
  sha256, not by file name.
- **Licensing**: record the licence and source reference per image. UniStega and some face
  sets are research-use only; that is fine for calibration, but keep it out of any release
  artifact.
- **Determinism**: sort candidates before sampling, seed the sample, and make the canonical
  PNG byte-stable. `split_version` is the hash of the resulting ordered tuples.
- **Storage**: 200 images at 512x512 is roughly 86 MB. Keep pixels gitignored, keep manifests
  in git.
- **Scope boundary**: no Kpub, no Xfeat, no EDICT, no retrieval, no FAISS. Those start at
  Phase 7 and Phase 17.
- **Changing the split later invalidates every stored result.** If a source turns out to be
  unusable, bump `schema_version`, rebuild, and re-run every system. Do not patch one role.
