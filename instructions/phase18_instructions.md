# Phase 18: Deterministic Token Selection

> **Goal**: Use the digital token to choose exactly one candidate from the ordered Top-K list,
> with a stable hash and a documented 1/K guessing bound.
> **Why**: The transmitter and receiver must independently choose the same reference. The token
> supplies reproducibility and a keyed choice. This is also the security primitive that prevents
> a receiver without the token from selecting the same candidate.

> **Precondition**: Phase 17 is frozen: `phase17_index_manifest.json` and
> `phase17_verify.json` pass, and `retrieve()` returns a stable ordered list.

> **Hard rule**: never use Python's built-in `hash()`. It is salted per process
> (`PYTHONHASHSEED`) and differs across runs and versions. Use SHA-256.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase17_verify.py
.venv\Scripts\python.exe -c "import hashlib, sys; print(sys.version); print(hashlib.sha256(b'x').hexdigest()[:16])"
```

Expected: Phase 17 PASS. The SHA-256 value must be `2d711642b726b044`; if not, the environment
is broken.

---

## Step 1: Config

Append under `retrieval:`:

```yaml
retrieval:
  # ... existing keys stay unchanged ...
  token_selection:
    domain: "racs2-token-selection-v1"
    hash: "sha256"
    index_bytes: 8                 # first 8 bytes -> 64-bit integer
    mode: "hash_mod_k"             # index = int(H(domain|token)[:8]) % K
    top_k: 10                      # must equal retrieval.faiss.default_top_k
```

`domain` is a domain-separation string. Changing it changes every selection and is a protocol
break; it may only change together with a full rerun of Phases 18, 21 and 22.

---

## Step 2: Structure

```text
retrieval/
  token_selection.py    # selection_index(), select(), distribution()
scripts/
  phase18_token_selection.py
  phase18_verify.py
experiments/results/
  phase18_token_selection.json
  phase18_distribution.png
  phase18_verify.json
```

---

## Step 3: Implement `retrieval/token_selection.py`

```python
SELECTION_DOMAIN = "racs2-token-selection-v1"

def selection_index(token: str, k: int, *, domain: str = SELECTION_DOMAIN,
                    index_bytes: int = 8) -> int:
    """Deterministic index in [0, k). Raises for k < 1 or an empty token."""

def select(token: str, candidates: Sequence[dict], *,
           domain: str = SELECTION_DOMAIN, index_bytes: int = 8) -> dict:
    """Return the chosen candidate dict; candidates are the ordered Top-K list."""

def selection_distribution(tokens: Iterable[str], k: int, *,
                           domain: str = SELECTION_DOMAIN) -> dict:
    """Counts, frequencies, min/max bin deviation and a chi-square statistic."""
```

Rules:

- `token` is `str(token).strip()`. Reject an empty string. Accept any non-empty string; the
  frozen token map uses 8 lowercase hex characters, but the function must not assume that.
- Compute `digest = hashlib.sha256(f"{domain}|{token}".encode("utf-8")).digest()`.
- `index = int.from_bytes(digest[:index_bytes], "big") % k`.
- `select()` returns `candidates[index]`, preserving the candidate's rank and score. It must not
  reorder or re-score candidates.
- `k == len(candidates)` must be validated. If the caller passes a different K, raise; do not
  silently truncate, because truncation changes the selection.
- `k == 1` returns index 0 by construction. Record this explicitly.
- Do not add a salt per call, a timestamp, randomness or model inference to the selection path.
- Tx and Rx must call the same function with the same candidates. There is no second
  implementation.

Optional helper, but keep the same primitive:

```python
def selection_metadata(token: str, k: int, candidates: Sequence[dict]) -> dict:
    """Domain, token hash, index, k, chosen image_id, candidate ids hash."""
```

Never log the raw token in a public artifact if a hashed form is enough. The frozen baseline
token map is already project-internal, but Phase 22 tables should use `token_sha256`.

---

## Step 4: Implement `scripts/phase18_token_selection.py`

Algorithm:

```text
1. set_seed(42)
2. determinism: run selection_index for 20 tokens x K in [1,5,10,20,50] in this process
3. cross-process: spawn two fresh Python processes that print the same table
   -> byte-identical stdout
4. distribution: 10,000 synthetic tokens (seeded) per K
   -> counts per bin, min/max deviation, chi-square vs uniform
5. real tokens: load data/metadata/tokens.json (66 evaluation + calibration tokens)
   -> selections for each K; record how many distinct indices are used
6. identity checks: K=1 -> index 0 for every token;
   same token + same K -> same index across 1,000 repeats
7. save phase18_token_selection.json and phase18_distribution.png
```

Synthetic tokens must come from a fixed seed generator and be recorded only as a seed and count,
not as a 10,000-line list.

Output schema:

```json
{
  "schema_version": 1,
  "phase": 18,
  "domain": "racs2-token-selection-v1",
  "hash": "sha256",
  "index_bytes": 8,
  "k_sweep": [1, 5, 10, 20, 50],
  "determinism": {
    "same_process_repeats": 1000,
    "cross_process_equal": true,
    "cross_process_runs": 2
  },
  "synthetic": {
    "tokens": 10000,
    "seed": 42,
    "per_k": {"10": {"counts": [1001, 998, ...], "min_bin": 950, "max_bin": 1050,
                     "chi2": 9.7, "dof": 9, "p_value_approx": 0.37}}
  },
  "real_tokens": {
    "count": 132,
    "per_k": {"10": {"distinct_indices_used": 10, "max_bin": 19,
                     "min_bin": 8, "counts": []}}
  },
  "k1_stable": true,
  "guessing_probability": "1/K",
  "token_sha256_sample": ["..."],
  "paper_note": "index = H(token) mod K as described in the RACS design"
}
```

Do not tune the domain or byte count to make the real-token distribution look prettier. The
uniformity claim is about the synthetic population; a finite real token set is allowed to be
uneven.

---

## Step 5: Implement `scripts/phase18_verify.py`

Checks:

```text
[ ] domain, hash and index_bytes match config
[ ] no source file in retrieval/token_selection.py calls hash()
[ ] K=1 yields index 0 for 1,000 distinct tokens
[ ] same token and K yield the same index within a process
[ ] two fresh processes produce identical selection tables
[ ] every synthetic-token index is in [0, K)
[ ] chi-square is reported with dof=K-1; no p-value is fabricated if scipy is absent
[ ] real tokens produce a valid index for every K
[ ] selection returns the candidate at that index, with unchanged rank and score
[ ] K mismatch between candidates and K raises
[ ] empty token raises
[ ] phase18_distribution.png exists
[ ] phase17 manifest hash matches the one recorded here
```

Exit non-zero on failure.

---

## Step 6: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase18_token_selection.py
.venv\Scripts\python.exe scripts\phase18_verify.py
```

Expected summary:

```text
domain            : racs2-token-selection-v1
K sweep           : 1, 5, 10, 20, 50
cross-process     : equal
K=1               : always 0
synthetic chi2    : reported, dof = K-1
real tokens       : 132 selections per K, all in range
guessing bound    : 1/K
verify            : PASS
```

---

## Checklist

```text
[ ] SHA-256 based, not Python hash()
[ ] domain separation string frozen in config
[ ] index = int(H(domain|token)[:8]) % K
[ ] select() returns the candidate without reordering
[ ] K=1 is stable
[ ] same token + candidates -> same selection across processes
[ ] synthetic distribution over 10k tokens recorded for every K
[ ] real-token distribution recorded
[ ] guessing probability stated as 1/K
[ ] Tx/Rx use the same function (no second implementation)
[ ] phase18_token_selection.json and phase18_verify.json written
[ ] no Phase 17 artifact modified
```

**When this passes, token selection is frozen.** Any later change to the domain, hash or byte
count invalidates `Xref_tx == Xref_rx` and requires a full rerun.

---

## The Gate Question (answer in your own words)

> *Why is `index = H(token) mod K` both a reproducibility mechanism and a (weak) security
> mechanism, and what exactly does "1/K guessing" mean?*

Answer with your artifacts:

- Reproducibility: both ends compute the same integer from the same token and the same ordered
  candidate list, so they select the same reference without communicating anything else.
- Security: a receiver who has the database and ranking but not the token must guess which of
  the K candidates was used. With a uniformly distributed hash, each guess succeeds with
  probability 1/K.
- This is not cryptographic secrecy. K is small (often 10), so the bound is weak by construction.
  The token raises the cost of blind selection; it does not make the reference unknowable.
- Uniformity must be tested on a large synthetic token set. A finite real token map may be
  uneven and that is not a defect.
- Python's `hash()` would silently break reproducibility across processes because of hash
  randomisation, which is why SHA-256 is mandatory.

---

## Report back

- Domain string, hash, index bytes, K values
- Cross-process determinism result
- Synthetic and real-token distributions, with chi-square dof
- K=1 result
- The 1/K statement and its limits
- Your answer to the gate question

---

## Notes and risks

- **Selection is not quality tuning.** Never choose a token or domain because a candidate
  "looks better"; that would leak the secret into the protocol and break the security claim.
- **Candidate order is part of the protocol.** Top-K must come from the frozen Phase 17 index and
  the frozen tie-break.
- **Do not use `random`, `secrets`, `uuid`, time or process ids** anywhere in this module.
- **K is public.** The security comes from token unpredictability, not from hiding K.
- **`hashlib` is stable across Python versions**; `hash()` is not. Do not "optimise" to `hash()`.
- **If K ever changes at runtime, selection changes.** Freeze one K for Phases 21-22 and record it.
