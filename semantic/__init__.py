"""
Phase 7 — Public Conditions (Kpub and Xfeat) for RACS2.

This package provides deterministic extraction of the two public conditioning
inputs used by both AgentSemSteCom (Phase 8) and RACS retrieval (Phase 17):

  * ``Kpub``: a textual generation prompt derived from the image content.
  * ``Xfeat``: an implicit structural feature (segmentation map for common,
    skeleton for facial, absent for style).

Both inputs are *public* in the paper's threat model (learning them does not
reveal the secret), but *determinism-critical*: if Tx and Rx produce different
values, they generate different reference images and EDICT inversion fails.
"""
