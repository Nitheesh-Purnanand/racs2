"""
Role-tagged frozen test splits for RACS2.

Every consumer of the calibration or evaluation data goes through this module so
that role enforcement cannot be forgotten. The manifest is the contract:

    data/metadata/test_split.json

Two roles exist:
  - ``calibration``: the paper's dataset (UniStega). Used only to check
    approximate reproduction of published numbers. Loading it requires an
    explicit ``allow_calibration=True``.
  - ``evaluation``: our authoritative dataset. Every claim and every
    baseline-versus-RACS comparison uses this role only.

Usage:
    from evaluation.splits import Split, RoleError, load_tokens

    split = Split.load(role="evaluation")
    for record in split.records("common"):
        ...
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from common.paths import IMAGE_INDEX_PATH, PROJECT_ROOT, TEST_SPLIT_PATH, TOKENS_PATH

ROLES = ("calibration", "evaluation")
CLASSES = ("common", "facial", "style")


def resolve_path(value):
    """Resolves a manifest path against the project root when relative."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class RoleError(ValueError):
    """Raised when a record is used under the wrong dataset role."""


class SplitError(ValueError):
    """Raised when a split manifest is malformed or inconsistent."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_split_version(records) -> str:
    tuples = [[r["role"], r["class"], r["image_id"], r["sha256"]] for r in records]
    return sha256_json(tuples)


def compute_tokens_version(records) -> str:
    tuples = [[r["image_id"], r["token"], r["wrong_token"]] for r in records]
    return sha256_json(tuples)


class Split:
    """A frozen, role-tagged set of image records."""

    def __init__(self, records, manifest, role=None, verify=True):
        self._records = list(records)
        self._manifest = dict(manifest)
        self.role = role
        if verify:
            self._verify_files()

    @classmethod
    def load(cls, role=None, allow_calibration=False, manifest_path=None,
             verify=True):
        if role is not None and role not in ROLES:
            raise RoleError(f"unknown role {role!r}; expected one of {ROLES}")
        if role == "calibration" and not allow_calibration:
            raise RoleError(
                "loading the calibration split requires allow_calibration=True; "
                "calibration data must never enter an evaluation run"
            )

        path = Path(manifest_path or TEST_SPLIT_PATH)
        if not path.exists():
            raise SplitError(f"split manifest not found: {path}")

        with io.open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        expected = manifest.get("split_version")
        actual = compute_split_version(manifest.get("records", []))
        if expected != actual:
            raise SplitError(
                f"split_version mismatch: manifest says {expected}, recomputed {actual}"
            )

        records = manifest.get("records", [])
        if role is not None:
            records = [r for r in records if r.get("role") == role]

        instance = cls(records, manifest, role=role, verify=verify)
        instance.assert_classes()
        return instance

    # ------------------------------------------------------------------ basics
    def __len__(self):
        return len(self._records)

    @property
    def version(self):
        return self._manifest.get("split_version")

    @property
    def manifest(self):
        return dict(self._manifest)

    def ids(self, cls=None):
        return [r["image_id"] for r in self._select(cls)]

    def records(self, cls=None):
        return [dict(r) for r in self._select(cls)]

    def get(self, image_id):
        for record in self._records:
            if record["image_id"] == image_id:
                if self.role is not None and record["role"] != self.role:
                    raise RoleError(
                        f"{image_id} has role {record['role']!r}, "
                        f"but this split is {self.role!r}"
                    )
                return dict(record)
        raise KeyError(f"{image_id} is not in this split")

    def counts(self):
        out = {}
        for record in self._records:
            key = (record["role"], record["class"])
            out[key] = out.get(key, 0) + 1
        return out

    def sha256_set(self):
        return {r["sha256"] for r in self._records}

    def path_set(self):
        return {str(resolve_path(r["path"]).resolve()) for r in self._records}

    # -------------------------------------------------------------- invariants
    def _select(self, cls):
        if cls is None:
            return self._records
        if cls not in CLASSES:
            raise ValueError(f"unknown class {cls!r}; expected one of {CLASSES}")
        return [r for r in self._records if r["class"] == cls]

    def _verify_files(self):
        for record in self._records:
            path = resolve_path(record["path"])
            if not path.exists():
                raise SplitError(f"missing image file: {path}")
            actual = sha256_file(path)
            if actual != record["sha256"]:
                raise SplitError(
                    f"sha256 mismatch for {path}: manifest {record['sha256']}, file {actual}"
                )

    def assert_classes(self):
        for record in self._records:
            if record.get("class") not in CLASSES:
                raise SplitError(f"bad class in {record.get('image_id')}")
            if record.get("role") not in ROLES:
                raise SplitError(f"bad role in {record.get('image_id')}")

    def assert_disjoint(self, other, corpus_index_path=None):
        if not isinstance(other, Split):
            raise TypeError("assert_disjoint expects another Split")
        overlap_sha = self.sha256_set() & other.sha256_set()
        overlap_path = self.path_set() & other.path_set()
        if overlap_sha or overlap_path:
            raise SplitError(
                f"splits overlap: {len(overlap_sha)} shared sha256, "
                f"{len(overlap_path)} shared paths"
            )
        if corpus_index_path is not None:
            self.assert_disjoint_from_index(corpus_index_path)
            other.assert_disjoint_from_index(corpus_index_path)
        return True

    def assert_disjoint_from_index(self, index_path=None):
        path = Path(index_path or IMAGE_INDEX_PATH)
        with io.open(path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        corpus_sha = {record["sha256"] for record in index.get("images", [])}
        overlap = self.sha256_set() & corpus_sha
        if overlap:
            raise SplitError(
                f"{len(overlap)} images in role {self.role!r} also appear in the "
                "retrieval corpus"
            )
        return True


def load_tokens(path=None, split=None, strict=False):
    """
    Loads the fixed token map. Returns ``(mapping, version)`` where mapping is
    ``{image_id: {"token": str, "wrong_token": str}}``.

    When ``split`` is given, every image in that split must have a token. Extra
    tokens for the other role are allowed unless ``strict=True``, because both
    roles share one token file.
    """
    token_path = Path(path or TOKENS_PATH)
    if not token_path.exists():
        raise SplitError(f"token manifest not found: {token_path}")
    with io.open(token_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("records", [])
    expected = payload.get("tokens_version")
    actual = compute_tokens_version(records)
    if expected != actual:
        raise SplitError(
            f"tokens_version mismatch: manifest says {expected}, recomputed {actual}"
        )

    mapping = {r["image_id"]: {"token": r["token"], "wrong_token": r["wrong_token"]}
               for r in records}
    if len(mapping) != len(records):
        raise SplitError("duplicate image_id in tokens.json")

    if split is not None:
        missing = set(split.ids()) - set(mapping)
        if missing:
            raise SplitError(
                f"token/split mismatch: {len(missing)} split images have no token"
            )
        if strict:
            extra = set(mapping) - set(split.ids())
            if extra:
                raise SplitError(
                    f"token/split mismatch: {len(extra)} tokens outside the split"
                )
    return mapping, expected


def write_manifests(split_payload, token_payload, split_path=None,
                    token_path=None):
    """Atomically writes both manifests."""
    split_path = Path(split_path or TEST_SPLIT_PATH)
    token_path = Path(token_path or TOKENS_PATH)
    for path, payload in ((split_path, split_payload), (token_path, token_payload)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        tmp.replace(path)
    return split_path, token_path
