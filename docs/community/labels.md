---
title: Project labels
description: >-
  Recommended GitHub labels for freshdata issue triage, including colors,
  descriptions, and when contributors should use each label.
keywords: freshdata labels, issue triage, contributor labels, good first issue
---

# Project labels

Use labels to make the issue backlog easy to scan. The table below is the
recommended set for maintainer-created labels and contributor-facing triage.

| Label | Color | Description | Use it for |
|---|---:|---|---|
| `examples` | `#1D76DB` | Example notebooks and recipes | Docs, examples, notebooks, and workflow recipes. |
| `benchmarks` | `#5319E7` | Performance benchmarks | Runtime, memory, regression, or comparative benchmark work. |
| `packaging` | `#D4C5F9` | Packaging, dependencies, and distribution | PyPI metadata, wheels, extras, dependency bounds, and install issues. |
| `ci` | `#FBCA04` | CI / GitHub Actions | Test matrix, coverage, lint, docs deploys, and automation fixes. |
| `release` | `#B60205` | Release and packaging | Changelog, versioning, release notes, and publish flow work. |
| `needs reproduction` | `#D876E3` | Reproduction steps or sample data needed | Bug reports that need a minimal DataFrame, environment details, or failing command. |
| `needs maintainer decision` | `#FBCA04` | Maintainer decision needed before implementation | Scope, API, dependency, behavior, or roadmap decisions that should not be guessed. |

## Current repository state

As of the label audit for issue #5, these labels already exist on GitHub:
`examples`, `benchmarks`, `ci`, and `release`.

These labels are recommended for maintainers to create when needed:
`packaging`, `needs reproduction`, and `needs maintainer decision`.

## Triage guidance

- Add `needs reproduction` before asking a contributor to debug a bug without a
  minimal DataFrame or command.
- Add `needs maintainer decision` when the next step changes public behavior,
  dependencies, or the documented cleaning contract.
- Prefer `examples` for tutorial-style work and `documentation` for reference
  fixes, README updates, or prose-only cleanup.
- Use `packaging` for install and distribution issues even when they also touch
  `ci` or `release`.
