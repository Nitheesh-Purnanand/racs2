# RACS2 — Session Handoff

## UPDATE 2026-10-29 ~16:10 — Phases 17-22 built and passing (read this first)

The baseline rebuild (Phases 10-16) was **stopped on request** after Phases 10-14 finished
(`temp/chunk1.log`, exit 0). Phases 15/16 were NOT rebuilt, so the frozen Phase 15/16 tables are
still the stale substitute-stack ones (§2 below still applies to them).

Instead the RACS side was completed:

| Phase | Script | Result |
| :-- | :-- | :-- |
| 17 FAISS exact retrieval | `scripts/phase17_20_racs.py` | PASS — self-query top-1 = 1.0, faiss `IndexFlatIP` agrees with in-process exact search |
| 18 SHA-256 token selection | same | PASS — cross-process identical, K=1 stable, near-uniform over K=10 |
| 19 structural edges cache | same | PASS — 1000 rows, deterministic, 5 CLIP-vs-structure disagreements |
| 20 rerank (calibration only) | same | PASS — frozen alpha=0.75, beta=0.25, pool 50 -> top 10 |
| 21 RACS-Base end to end | `scripts/phase21_racs.py --limit 2` | PASS — **Xref_tx == Xref_rx 66/66** (separate Tx/Rx processes), downstream is the imported baseline |
| 22 controlled comparison | `scripts/phase22_compare.py` | `phase22_comparison.{json,md}`, `phase22_grid.png` |

Phase 22 (subset: 2 eval images/class x 4 SNRs, both systems measured fresh through one code path):
reference acquisition **3.92 s -> 0.048 s (~82x)**, no GPU memory for Xref; recovery PSNR
22.37 vs 22.62 dB (paired +0.25 dB), LPIPS 0.238 vs 0.227; correct-minus-wrong token gap
8.67 vs 8.94 dB. Indicative only (n=6 images).

Code: `retrieval/racs.py` (all of 17-20, shared Tx/Rx), `pipelines/racs_base.py` (subclass of
`Baseline`, overrides only `reference_path`). Config: `retrieval.faiss/token_selection/
structural/ranking` in `configs/base.yaml`. Deviations: last section of `docs/DEVIATIONS.md`.

Gotchas found: faiss + torch in one process -> `OMP Error #15` (faiss runs in a subprocess);
CLIP text encoding segfaults once the diffusion stack is loaded (retrieval runs in its own
Tx and Rx processes before the pipeline loads).

**Remaining**: Phase 15/16 rebuild (`run_baseline.py --mode paper --from-phase 15 --to-phase 16
--yes --resume yes`, ~4-5 h), then the full Phase 21 split (`phase21_racs.py --limit 0`, ~2 h) and
rerun Phase 22. Nothing is committed yet.

**Later the same day — quick paper-stack baseline**: Phase 15B + 16 were run with
`--limit 2` (24 cases, ~15 min). Measured: recovery PSNR 18.3-25.1 dB by class/SNR; token gap
correct 22.37 / wrong 13.96 / no_token 14.02 dB (**gap 8.41 dB**, paper 6.46); XuNet 0.500
(n=8, chance). Verifiers fail only their N=22 count checks (see DEVIATIONS, last section).
Report: `experiments/results/final_baseline_report.md`.

---

> **Last updated**: 2026-10-29 14:35 (end of the paper-component upgrade session)
> **Repo**: `D:\racs2` · branch `main` · **nothing committed this session** (211 modified
> paths, plus the new files listed in §3)
> **State in one line**: the code and config now run the paper's own components, Phases 7-9
> are rebuilt and verified on them, and **Phases 10-16 still hold artifacts from the old
> substitute stack**. The rebuild was stopped partway on purpose.
> **Plan**: `racs2_implementation_plan.md` v2 (baseline-first). Phases after 5 were renumbered
> on 2026-10-23; the plan's Section 5 has the old-to-new map.
> **Companions**: `README.md` (quickstart), `project_status.md` (full history),
> `docs/DEVIATIONS.md` (every difference from the paper).

**What is true at handoff time** (re-checked on disk at 14:35, not assumed):

| Check | Result |
| :-- | :-- |
| live `configs/base.yaml` hash | `397b0e7c…` |
| frozen baseline manifest config hash | `da93514e…` (substitute stack) |
| Phase 7 / Phase 8 / Phase 9 verify | PASS / 12 of 12 PASS / PASS |
| `experiments/results/final_baseline_report.json` | `status: fail` (written 12:49 today) |
| Phase 16 verify | **16/17**, the single failure is the frozen-hash check |
| free commit / C: free / D: free | **1.9 GB** / 3.3 GB / 15.8 GB |

**Resume here** (details and costs in §5.1):

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\run_baseline.py --mode paper --from-phase 10 --to-phase 16 --yes --retries 2
```

Free memory first. A long phase needs roughly 5 GB of free commit charge, and there is 1.9 GB
right now; see §7.

---

## 1. What this project is

**RACS2** reproduces **AgentSemSteCom** (Meng et al., IEEE TCCN 12, 2026) and then replaces
exactly one of its stages — reference acquisition — with deterministic retrieval instead of
diffusion generation, so the two can be compared under identical conditions.

```
Secret Xs → Kpub (LLM) + Xfeat (structure) → Xref → EDICT hide + token perturbation
         → Xstego → SwinJSCC semantic codec → AWGN → decoder
         → EDICT inversion + inverse perturbation → recovered secret [→ CodeFormer]
```

The baseline generates `Xref` with PicX-Real + ControlNet from a token-derived seed. RACS will
retrieve it from a CLIP/FAISS-indexed corpus and select it with the digital token, importing
the rest of the pipeline unchanged. The efficiency claim is retrieval (sub-second) versus
diffusion (13-21 s per reference). The protocol invariant is `Xref_tx == Xref_rx`.

---

## 2. READ FIRST — the tree is in a mixed state

`configs/base.yaml` and all the code select the **paper components**. The artifacts do not yet
agree with it:

| Artifact | Built by | When | Agrees with the live config? |
| :-- | :-- | :-- | :-- |
| `public_conditions/manifest.json` + 132 `.npz` | **paper stack** | 2026-10-29 13:11 | ✅ |
| `phase7_verify.json` | paper stack | 13:11 | ✅ PASS |
| `references/*.png` (132) + `phase8_references.json` | **paper stack (PicX-Real)** | 13:21 | ✅ |
| `ipadapter_embeds/` (132) + `phase8_ipadapter_embeds.json` | paper stack | 13:28 | ✅ |
| `phase8_verify.json` | paper stack | 13:33 | ✅ 12/12 PASS |
| `phase9_verify.json` | paper stack | 13:34 | ✅ PASS |
| `phase10_edict.json` | old substitute stack | 10-27 20:25 | ❌ stale |
| `phase11_local.json` + `phase11/` | old substitute stack | 10-27 22:36 | ❌ stale |
| `phase12_awgn.json` + `phase12/` | old substitute stack | 10-27 22:45 | ❌ stale |
| `phase13_codec.json` + `phase13/verify` | old substitute stack | 10-27 23:03 | ❌ stale |
| `phase14_codeformer.json` + `phase14/verify` | old substitute stack | 10-27 23:05 | ❌ stale |
| `phase15A_manifest.json`, `phase15A_calibration_table.json` | old substitute stack | 10-28 23:36 | ❌ stale |
| `baseline_table.json`, `baseline_manifest.json` | old substitute stack | 10-28 21:32 UTC | ❌ stale |
| `phase16_manifest.json` (the freeze) | old substitute stack | 10-29 08:28 | ❌ stale, pins hashes that no longer match |
| `phase15_verify.json` | old substitute stack | 10-29 10:22 | ❌ passed then, cannot pass now |
| `baseline_security.json`, `baseline_steganalysis.json`, `phase16_verify.json` | old substitute stack | 10-29 11:32-11:34 | ❌ stale |
| `final_baseline_report.{json,md}`, `final_baseline_table.csv` | reads the stored JSONs | 10-29 12:49 | ❌ reports `status: fail` |

**Consequences right now** (all of this was re-checked on disk, 14:35)

- The 132 reference PNGs were regenerated in place (`git status` shows all 132 as modified), so
  the Phase 11/15/16 per-case records point at reference files whose contents have changed.
  Their recorded hashes no longer match.
- `run_baseline.py --verify-only` will therefore FAIL on Phases 11/13/14/15. That is the mixed
  state being detected, not a regression. Two of the failures are concrete:
  - **Phase 15**: the check "component choices match frozen manifests" compares the live config
    against the frozen evaluation manifest. Live config is `codec.kind: swinjscc`,
    `enhancement.enabled: true`, `ipadapter.enabled: true`, `reference.backend: picx_real`;
    the frozen manifest says `sd_vae_substitute` / `disabled` / no IPAdapter. The stored
    `phase15_verify.json` says `pass: true` only because it ran at 10:22, before the config
    was switched at 13:36.
  - **Phase 16**: 16 of 17 checks pass. The one failure is "frozen security and steganalysis
    hashes match the phase16 manifest": the manifest pins `8ab321a5…` and `78db750b…`, the
    files on disk hash to `54ffbe66…` and `d5936ac4…` after the 11:32 rerun. The manifest's
    own `baseline_table_sha256` (`a6db2536…`) also predates the table now on disk
    (`2d5cc07a…`), so the freeze is stale in three layers and must be regenerated whole.
- `run_baseline.py --show` prints the **old substitute-stack numbers** and a `fail` status,
  because it reads those stored JSONs. They are still the last internally consistent
  baseline, and a copy is archived in `experiments/results/archive/`.
- `docs/phase16_report.md` matches `phase16_manifest.json` (17.39 / 13.05 / 13.07 dB, XuNet
  0.4875), which is the pre-rerun freeze. The files on disk now say 17.33 / 13.01 / 13.04 dB
  and 0.4625. Both are substitute-stack numbers. Do not quote either as final: regenerate the
  report after the rebuild.

**The fix is to finish the rebuild** (section 5).

---

## 3. What was done this session

### Paper components are now the real components

All five fetched by `scripts/fetch_paper_assets.py` and pinned by SHA-256 in
`models/PAPER_ASSETS.json`:

| Stage | Was | Is now |
| :-- | :-- | :-- |
| `Kpub` | BLIP caption + fixed template | **Qwen2.5-0.5B-Instruct** over a BLIP perception caption, greedy |
| `Xfeat` common | SegFormer-B0 | **OneFormer ADE20K Swin-L** |
| `Xfeat` facial | MediaPipe face mesh | **OpenPose** body+face; MediaPipe only where OpenPose finds no body (3 of 22, recorded per image) |
| reference | SD v1.5 + ControlNet | **PicX-Real** + SD v1.5 ControlNet (seg / openpose) |
| image prompt | none | **IPAdapter-plus weight 1**, injected at both ends |
| codec | SD-VAE substitute | **SwinJSCC** C=64, AWGN, trained 10 dB |
| enhancement | pass-through, disabled | **CodeFormer** fidelity 0.5, facial, enabled |

### Measured on the paper stack (component level, all verified)

- **SwinJSCC**: isolated round-trip **32.54 dB at 10 dB SNR**, seed-replay bit-identical
  (the SD-VAE substitute gave 22.88 dB).
- **One image end to end** (facial, 10 dB, before the Phase 8 rebuild): correct recovery
  **24.25 dB** / SSIM 0.656 / LPIPS 0.182, wrong token **13.90 dB**, no-token **14.06 dB** —
  a **10.3 dB** correct-minus-wrong gap, against 4.32 dB on the substitute stack and 6.46 dB
  in the paper. This is a single case, not a table; it is the reason to finish the rebuild.
- OneFormer, OpenPose, Qwen `Kpub`, CodeFormer and the IPAdapter embedding are each
  deterministic on repeat (checked explicitly).
- Phase 7: 132 records, **0 degenerate LLM replies**, 3 documented annotator fallbacks,
  `config_hash 9e83de1b…`.
- Phase 8: 132 PicX-Real references, **exact rerun determinism**, peak VRAM 2.84 GB, 12/12 PASS.

### Code changes worth knowing

- `semantic/keygen.py` — `LLMKeygen` with a one-shot chat prompt, instruction-echo stripping,
  markdown/URL sanitising, and a deterministic fallback to the perception prompt for degenerate
  replies (counted in the Phase 7 manifest). The prompt text and `llm_prompt_version` live in
  the config, so editing either invalidates every cached `Kpub`.
- `semantic/features.py` — real `OneFormerExtractor`; `OpenPoseExtractor` rewritten from a stub
  that returned **fabricated keypoints** to the real `controlnet_aux` detector;
  `FacialFeatureChain` adds the per-image MediaPipe fallback and records which annotator ran.
- `reference/generate.py` — `backend: picx_real`, with per-component provenance (PicX-Real vs
  the SD v1.5 fallback) recorded in the Phase 8 manifest.
- `steganography/ipadapter.py` — IPAdapter-plus attached to the EDICT UNet; **embeddings cached
  by reference SHA-256** so Tx and Rx read the same tensor rather than each deriving one from
  pixels. A float→uint8 truncation on the Tx side that could have desynced them is fixed.
  `scripts/phase8_embeds.py` prefills the cache so the baseline loop never loads CLIP-H.
- Memory discipline (6 GB card, ~4 GB free commit): Phase 7 runs in two passes (all `Kpub`,
  then features) with one extractor resident; CLIP-H and CodeFormer run on the CPU; every phase
  is launched with `-X faulthandler` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Gates updated, not weakened**: Phase 13 no longer assumes the codec has a `.vae`; Phase 14
  no longer asserts "enhancement disabled / delta exactly zero" but requires measured deltas,
  a real backend and size preservation; Phase 15 checks that config and frozen manifest *agree*
  on codec/enhancement/IPAdapter/EDICT instead of hardcoding the old choices.
- **Bug found and fixed**: `edict.sweep_mixing` contained `0.5`, the singular point of the EDICT
  coupling, which the invertibility guard rejects — so `phase10_edict.py` could never complete a
  full rebuild. It had been hidden because the last runs were verify-only. Sweep now starts at
  0.55 and out-of-range cells are skipped with a note instead of crashing.
- **Entry points**: `scripts/run_baseline.py` (renamed from `run_phases_1_to_16.py`) now
  **prints the result tables on every run**, plus `--show` for instant results;
  `scripts/baseline_example.py` runs one image end to end. `README.md` is new.
- **Removed**: `scripts/verify_baseline_example.py` (duplicate), `imp_code`, a 2.5 GB duplicate
  IPAdapter image-encoder checkpoint, ~80 scratch files in `temp/`, the pip cache, stale Phase 16
  PNGs, `experiments/results/custom_example/`. `.gitignore` now covers every model weight
  directory, `third_party/`, and the embedding cache.

---

## 4. One command

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\run_baseline.py --show          # print current results (stale ones for now)
.venv\Scripts\python.exe scripts\run_baseline.py --verify-only --yes   # re-check gates (will fail until the rebuild finishes)
.venv\Scripts\python.exe scripts\baseline_example.py --mode paper --image-id eva_facial_0000 --class facial --snr 10
```

Every form prints its tables to the terminal and refreshes
`experiments/results/final_baseline_report.{md,json}` and `final_baseline_table.csv`.

Right now `--show` ends with `status: fail` and `--verify-only` fails at Phases 11/13/14/15.
Both are the mixed state being reported correctly (see §2), not bugs. `--verify-only` becomes
meaningful again only after the rebuild.

---

## 5. What needs to be done next

### 5.1 Finish the baseline rebuild (the blocking item)

**Pre-flight, do this first.** At handoff the machine has **1.86 GB of free commit** and 3.3 GB
free on C:, which is not enough. A run that loads a diffusion model needs about 5 GB.

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeCommitGB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
```

Close memory holders (an IDE language server held 10 GB earlier in this project) or reboot.
If that is still not enough, move or add a pagefile on D: (15.8 GB free) as in §5.3.

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\run_baseline.py --mode paper --from-phase 10 --to-phase 16 --yes --retries 2
```

Rough costs on this machine: Phase 10 ~15 min · 11 ~20 min · 12 ~5 min · 13 ~10 min ·
14 ~15 min · 15A ~1 h · 15B ~2-3 h · 16 ~1.5-2 h.

Two behaviours of the orchestrator worth knowing:

- `--resume` is only forwarded to Phase 15B when you pass `--resume yes`, or when the mode is
  `frozen`. With `--mode paper` the default is `auto`, which does not resume. Pass
  `--resume yes` on the Phase 15 chunk, and an interruption costs minutes rather than hours.
- A non-frozen run that reaches Phase 15 first archives the existing frozen artifacts to
  `experiments/results/archive/<timestamp>_before_nonfrozen_run/`, so the substitute baseline
  is kept automatically. Do not pass `--no-archive`.

Suggested chunking, because a single 5-6 h window is fragile on this box:

```powershell
.venv\Scripts\python.exe scripts\run_baseline.py --mode paper --from-phase 10 --to-phase 14 --yes --retries 2
.venv\Scripts\python.exe scripts\run_baseline.py --mode paper --from-phase 15 --to-phase 15 --yes --resume yes
.venv\Scripts\python.exe scripts\run_baseline.py --mode paper --from-phase 16 --to-phase 16 --yes
.venv\Scripts\python.exe scripts\run_baseline.py --verify-only --yes
```

Expect, and check:

- Phase 11's matched-vs-mismatched reference gap should now be **real** rather than 0.15 dB,
  because IPAdapter makes the stego depend on reference pixels.
- Phase 13 isolated PSNR should land near 32 dB at 10 dB SNR.
- Phase 14 facial PSNR delta may be **negative** while LPIPS improves — CodeFormer trades
  distortion for perceptual quality. The gate allows that; report it honestly.
- Phase 15 Gate A (vs the paper's Table II) may now pass cells it used to miss. Gate B is the
  authoritative structural gate; any Gate A miss must stay itemised in `docs/DEVIATIONS.md`.
- Phase 16's correct-minus-wrong gap should rise well above the 4 dB threshold.

### 5.2 Then re-freeze and record (order matters)

1. **`docs/phase16_report.md`** still describes the substitute-stack run, and it currently
   matches the stale `phase16_manifest.json` (17.39 / 13.05 / 13.07 dB, XuNet 0.4875), not
   even the JSONs on disk. Regenerate its tables, aggregate line, XuNet paragraph, frozen-hash
   list and `split_version` from the new `baseline_security.json` and
   `baseline_steganalysis.json`.
2. **Re-freeze in this order**, so the hashes chain rather than contradict each other:
   `baseline_table.json` + `baseline_manifest.json` → `phase15A_manifest.json` → security →
   steganalysis → `phase16_manifest.json` → `phase16_verify.py`. The present manifest is stale
   in three layers (§2), so a partial re-freeze will keep failing the hash checks.
3. Add the new headline numbers to `project_status.md` §2 and to §6 of this file. Delete §6's
   "superseded" label only when both tables are rebuilt.
4. Update `docs/DEVIATIONS.md` for anything the paper stack fixes or changes.
5. **Commit.** The working tree holds the whole upgrade plus ~5 GB of gitignored weights.
   Nothing has been committed this session.

### 5.3 Known issues to decide on

1. **Machine memory is the real constraint.** Commit limit 21.2 GB with ~17 GB committed, C:
   at 3.3 GB free so Windows cannot grow the pagefile there, and 1.86 GB of free commit at
   handoff. A long run was already killed by the low-memory watchdog. A pagefile on D:
   (15.8 GB free) and/or a reboot (nonpaged pool was at 1.63 GB after repeated CUDA crashes)
   would make the long phases reliable.
2. **Frozen-split contamination**: `eva_style_0011` is a human portrait sitting in the
   cat-photograph *style* class. It predates this session (Phase 6, 2026-10-23). Fixing it
   means rebuilding the frozen split and everything downstream, so it should be decided before
   the Phase 22 comparison, not during a rebuild. Worth also auditing the other 21 style images.
3. **EDICT steps**: still 20 with strength 0.2, not the paper's 50 / 0.5. One config line
   (`steganography.edict_steps`), roughly 2.5× the runtime.
4. **MDJCM digital branch** is not reproduced (no public checkpoint); only the analog SwinJSCC
   branch exists, and no digital-branch number is claimed.

### 5.4 After the baseline is frozen — the RACS work (Phases 17-22)

1. **17** FAISS `IndexFlatIP` over the banked Phase 5 `.npz` (`pip install faiss-cpu==1.8.0.post1`).
   Row order must stay identical to `image_index.json`.
2. **18** token selection, `H(domain|token)[:8] mod K`, one implementation shared by both ends.
3. **19** structural representation (edge IoU) cached over the corpus.
4. **20** structure-aware reranking; freeze α, β, K on calibration data only.
5. **21 (GATE)** RACS-Base with `Xref_tx == Xref_rx` bit-exact on 100 % of evaluation images;
   the downstream must be **imported**, not copied.
6. **22** the controlled comparison against the frozen Phase 15/16 baseline.

Guides: `PHASE_RUNBOOK_17_TO_20.md`, `phase17_instructions.md` … `phase20_instructions.md`.

**Reference sensitivity is resolved.** IPAdapter-plus at weight 1 means the stego depends on
the reference pixels, so reranking has a mechanism and the Phase 20 A/B is meaningful. This was
an open risk while the reference entered only through the prompt.

---

## 6. Current numbers (superseded, substitute stack)

These come from `baseline_table.json` built 2026-10-28 21:32 UTC with the SD-VAE codec, no
IPAdapter and no CodeFormer. They remain the last internally consistent baseline and are kept
only until the rebuild finishes. `--show` currently prints these with `status: fail`, because
the Phase 15/16 verifies fail on the mixed state described in §2.

| Class | 5 dB | 10 dB | 15 dB | 20 dB |
| :-- | --: | --: | --: | --: |
| common PSNR | 13.14 | 15.65 | 17.78 | 19.19 |
| facial PSNR | 14.23 | 17.35 | 20.04 | 22.23 |
| style PSNR | 13.60 | 16.53 | 18.46 | 19.78 |

Security (aggregate recovery PSNR): correct **17.33 dB**, wrong **13.01**, no_token **13.04**;
gap **4.32 dB** (paper anchor 6.46). XuNet test accuracy **0.4625** on n=80 (chance 0.5, paper
anchor 0.5091). Phase 15 Gate B passed with 8 documented Gate A misses.

---

## 7. Environment — read before running anything

- Windows, **RTX 3050 Laptop 6 GB**, 16 GB RAM. venv at `.venv` (PyTorch 2.6.0+cu124).
- HF/diffusers/torch caches and pip temp are redirected to `D:\racs2\cache` and `D:\racs2\temp`
  (C: is full) — see `common/paths.py`.
- **Commit-charge trap**: safetensors mmaps blow the Windows paging file (`OSError 1455`, or a
  0xC0000005 access violation). `common/hf_lowmem.py` installs a read-only mmap shim and must be
  imported **before** `diffusers`/`transformers`. Do not remove that import.
- Check headroom before a long run:
  ```powershell
  Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeCommitGB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
  ```
- Several components are deliberately off the GPU (CodeFormer, the IPAdapter image encoder) —
  see `docs/DEVIATIONS.md`.

---

## 8. Repo layout

```
common/        paths, seed, device, hf_lowmem
configs/       base.yaml  (the single source of component choices)
data/          secret_images/ · retrieval_images/ (1000 PNGs, gitignored) · metadata/
semantic/      keygen.py (BLIP + LLM Kpub), features.py (OneFormer/OpenPose), cache.py
reference/     seed.py, conditions.py, generate.py (PicX-Real backend), train_controlnet.py
steganography/ edict.py, mask.py, hide.py, recover.py, pipeline.py, ipadapter.py
communication/ awgn.py, semantic_codec.py, swinjscc.py
enhancement/   codeformer.py
security/      token_conditions.py, steganalysis.py (XuNet)
pipelines/     agentsemstecom.py  (the baseline, assembled)
retrieval/     schema.py, corpus.py            (Phase 17 will add index/rerank here)
evaluation/    metrics.py, splits.py, report.py
models/        clip-vit-base-patch32 · picx-real · ip-adapter · oneformer-… · qwen2.5-… ·
               codeformer · swinjscc   (weights gitignored, hashes in PAPER_ASSETS.json)
third_party/   CodeFormer, SwinJSCC   (vendored upstream, gitignored)
scripts/       run_baseline.py (orchestrator) · baseline_example.py (one image) ·
               fetch_paper_assets.py · phaseN_*.py (build + verify per phase)
experiments/results/  tables, verify JSONs, grids, archive/ (superseded baselines)
docs/          DEVIATIONS.md, RACS_ROADMAP.md, PROJECT_UNDERSTANDING.md, phase16_report.md
```

---

## 9. Git state

- Remote `https://github.com/Nitheesh-Purnanand/racs2.git`; local `main` is ahead of
  `origin/main`. **This session's work is uncommitted**: 211 modified paths plus these new
  files: `README.md`, `communication/swinjscc.py`, `steganography/ipadapter.py`,
  `scripts/run_baseline.py`, `scripts/baseline_example.py`, `scripts/fetch_paper_assets.py`,
  `scripts/phase8_embeds.py`. `scripts/verify_baseline_example.py` is staged for deletion.
- `.gitignore` covers `.venv temp pip_cache cache`, corpus and test-set images, every model
  weight directory, `third_party/`, the IPAdapter embedding cache, per-case tensors
  (`experiments/results/baseline_cases/`) and build noise.
- `experiments/results/archive/` keeps the superseded substitute baseline and **is** tracked.
- Last commit: `cd45d0a Add phases 7-16 execution runbook`. Everything after it is this
  session's work.

---

## 10. Resume checklist for the next session

Work top to bottom. Each line is a check you can actually run.

- [ ] Read §2. Confirm that the tree is still mixed (config hash `397b0e7c…`, frozen manifest
      `da93514e…`) before changing anything. If they already agree, someone finished the
      rebuild and this section is history.
- [ ] Check free commit (§5.1). Below ~5 GB, stop and fix memory first. Do not start Phase 15
      on a starved box; it is the phase that gets killed.
- [ ] Run the rebuild in chunk 1: `--from-phase 10 --to-phase 14`. Watch Phase 10's EDICT sweep
      finish without the singular-mixing crash, and Phase 11's matched-vs-mismatched gap become
      real.
- [ ] Chunk 2: Phase 15 (15A calibration, then 15B evaluation). Confirm Gate B passes and that
      every Gate A miss is already a row in `docs/DEVIATIONS.md`.
- [ ] Chunk 3: Phase 16. Confirm the aggregate correct-minus-wrong gap clears 4 dB, and that
      the low-SNR exceptions are itemised rather than hidden.
- [ ] `--verify-only --yes` must now be clean. If it is not, read which phase failed; do not
      loosen the gate without recording why in `docs/DEVIATIONS.md`.
- [ ] Re-freeze in the order in §5.2, then regenerate `docs/phase16_report.md`.
- [ ] Update `project_status.md` §2 and this file's §6 with the new numbers, drop the
      "superseded" label, and run `--show` once to regenerate the reports and CSV.
- [ ] Commit in one or two logical commits (code+config first, artifacts second).
- [ ] Only then start Phase 17 (`PHASE_RUNBOOK_17_TO_20.md`). Nothing RACS-specific may be
      wired into the pipeline before Phase 15 passes.
