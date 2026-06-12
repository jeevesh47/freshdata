# Contributing to freshdata

Thanks for your interest! freshdata aims to stay **small and sharp** — a
focused data-cleaning library, not a framework. Contributions that fit that
philosophy are very welcome.

## Ground rules

- **Safety first.** Default behavior may only repair representation; anything
  that changes the statistics of the data must be opt-in.
- **Everything is reported.** Any new transformation must record an
  `Action` with an affected-cell count.
- **Vectorized only.** No row-wise `apply` / Python loops over rows in the
  cleaning path.
- **Tested.** New behavior needs tests, including the "does not fire when it
  shouldn't" case.

## Setup

```bash
git clone https://github.com/JohnnyWilson-Portfolio/freshdata
cd freshdata
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks to run before a PR

```bash
pytest                 # all tests
ruff check src tests   # lint
mypy                   # types
```

## Reporting bugs

Please include a minimal DataFrame that reproduces the issue and the output of
`fd.profile(df)` — it usually contains exactly the information needed.
