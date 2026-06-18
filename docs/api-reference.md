---
title: API reference
description: >-
  Complete freshdata API reference — clean, plan, repair, profile, suggest_plan,
  compare_plans, compare_clean, explain_clean, Cleaner, CleanConfig, and reports.
keywords: freshdata api, fd.clean, CleanConfig, CleanReport, pandas cleaning api
---

# API reference

Auto-generated from the source docstrings. Everything below is available as a
top-level attribute of `freshdata` (e.g. `import freshdata as fd; fd.clean(...)`).

## Cleaning

::: freshdata.clean

::: freshdata.repair

::: freshdata.Cleaner

## Profiling & inspection

::: freshdata.profile

::: freshdata.infer_roles

::: freshdata.explain_clean

## Planning & comparison

::: freshdata.suggest_plan

::: freshdata.plan

::: freshdata.compare_plans

::: freshdata.compare_clean

## Validator bridges

::: freshdata.from_gx

::: freshdata.from_dbt_failures

::: freshdata.from_pandera_errors

::: freshdata.emit_gx_expectations

::: freshdata.emit_dbt_tests

::: freshdata.ValidationBridgeResult

::: freshdata.ValidationFailure

## Schema drift

::: freshdata.SchemaHarmonizer

::: freshdata.SchemaContract

::: freshdata.ColumnContract

::: freshdata.SchemaColumnMapping

::: freshdata.SchemaHarmonizationResult

::: freshdata.MigrationDiff

::: freshdata.QuarantineResult

## Duplicate and replay defense

::: freshdata.DuplicateDefense

::: freshdata.DuplicateDefenseReport

::: freshdata.DuplicateExplanation

::: freshdata.IdempotencyKey

::: freshdata.BatchManifest

## Human review queues

::: freshdata.ReviewQueue

::: freshdata.ReviewDataset

::: freshdata.ReviewTask

::: freshdata.ReviewOption

## Configuration

::: freshdata.CleanConfig

## Reports & results

::: freshdata.CleanReport

::: freshdata.Action

::: freshdata.CleanPlan

::: freshdata.ColumnPlan

::: freshdata.RepairPlan

::: freshdata.RepairPatch

::: freshdata.ReviewItem

::: freshdata.Profile

::: freshdata.ColumnProfile

::: freshdata.ExplainReport

## Enterprise layer

The `freshdata.enterprise` subpackage is documented in the
[feature overview](feature-overview.md). Import its symbols lazily:

```python
from freshdata.enterprise import clean_enterprise, EnterpriseConfig
```
