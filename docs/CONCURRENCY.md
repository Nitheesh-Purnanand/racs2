# Working with concurrent sessions

Multiple agent sessions can be active on this repository at the same time.
This file records the situation and the conventions that keep simultaneous
edits safe.

## Observed situation (2026-10-06)

- All commits share the same configured git identity
  (`Nitheesh-Purnand <chalumuri.nitheesh@gmail.com>`), so the author field
  **cannot** distinguish who wrote a commit.
- Five `jcode` processes were running; one started at 20:22:22 and an unrelated
  commit (`a2f3679`, `tests/` + `project_status.md`) landed at 20:23:44, 82 s
  later. That timing points to a **second agent session**, not the user typing.
- That session's process was still alive at 20:48 but had not committed again.

## Risk

Two writers editing the same files (especially `project_status.md`, which the
roadmap designates as the single source of truth) can silently overwrite each
other's edits. Git does not auto-merge concurrent, unsequenced working-tree
writes in the same checkout; whichever `git add` runs last wins for that path.

## Conventions

1. **Refresh before editing.** Run `git status` and `git log --oneline -3`
   before starting an edit to see if another writer landed a commit.
2. **Keep edits additive.** Prefer new files over rewriting shared ones. Do not
   rewrite whole sections of `project_status.md`; append or make targeted edits.
3. **Re-read immediately before writing** shared files, so an interleaved change
   is not clobbered.
4. **Stage narrowly.** `git add <specific paths>` rather than `git add -A`, to
   avoid sweeping in another writer's in-progress files.
5. **Announce the owner in commit messages** when touching shared files
   (e.g. prefix `[session-A]`/`[session-B]`), so the other writer can see it.
6. **Verify before committing.** Run the CPU tests (`python -m unittest discover
   -s tests`) so a red tree is caught early.
7. **One logical change per commit**, committed promptly, to shrink the window
   for collisions.
8. **If a conflict is discovered post-hoc**, do not force-push; reconcile by
   hand and note it in `project_status.md`.

## Current state (checked 2026-10-06 20:49 local)

- `HEAD = ccd45be`, working tree clean.
- No commits from the other session since `a2f3679`.
- `tests/` pass (15/15); all modules compile.
