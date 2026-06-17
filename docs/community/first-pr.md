---
title: First PR guide
description: >-
  How new contributors can choose a first freshdata issue, understand labels,
  and submit a focused pull request.
keywords: freshdata first pull request, beginner contribution, good first issue
---

# First PR guide

Start with an issue labeled `good first issue` or `help wanted`. Small docs,
examples, and test-only changes are usually the best first contribution because
they can be reviewed without changing the cleaning engine.

## Pick an issue

Use labels to understand the shape of the work:

| Label | Good first PR shape |
|---|---|
| `examples` | Add or improve a short recipe, notebook, or inline DataFrame example. |
| `benchmarks` | Add a small timing fixture or document a repeatable benchmark command. |
| `packaging` | Clarify install extras, dependency bounds, or packaging metadata. |
| `ci` | Improve a workflow, test matrix, coverage step, or automation check. |
| `release` | Update changelog, release notes, or release process documentation. |
| `needs reproduction` | Add the missing minimal DataFrame, command, or failing test before fixing. |
| `needs maintainer decision` | Ask for a decision before implementing; do not guess the API or behavior. |

The complete label set and suggested colors live in
[Project labels](labels.md).

## Keep the PR focused

1. Branch from `main`.
2. Change only the files needed for the issue.
3. Add a test when behavior changes.
4. Update docs when user-facing behavior or contributor workflow changes.
5. Run the checks listed in [Contributing](../contributing.md).

For bugs, include the smallest DataFrame or command that proves the fix. For
docs or examples, include the exact page or file you changed and why.
