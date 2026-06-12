# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-12

Initial release.

### Added
- `freshdata.clean()` — automatic, audited cleaning: column-name
  normalization, whitespace stripping, sentinel-string normalization,
  empty row/column pruning, validated dtype inference (numeric incl.
  currency/thousands separators, datetime, boolean), and exact duplicate
  removal.
- Opt-in steps: imputation (`auto`/`mean`/`median`/`mode`), outlier
  clipping/flagging (IQR or z-score), constant-column dropping, memory
  optimization (numeric downcasting + category conversion), index reset.
- `freshdata.profile()` — read-only profiling whose dtype suggestions are
  produced by the same inference code `clean` uses.
- `freshdata.Cleaner` — reusable configured pipeline with `report_`.
- `freshdata.CleanConfig` — frozen, self-validating configuration;
  unknown options raise with a "did you mean" suggestion.
- `freshdata.CleanReport` / `freshdata.Action` — structured audit trail
  with `summary()`, `to_dict()`, `to_frame()`.
- Type hints throughout (`py.typed`), zero dependencies beyond
  pandas/numpy, support for Python 3.9–3.13.
