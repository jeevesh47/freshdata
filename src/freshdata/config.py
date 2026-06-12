"""Configuration for the cleaning pipeline.

:class:`CleanConfig` is the single source of truth for every option accepted
by :func:`freshdata.clean`, :func:`freshdata.profile`, and
:class:`freshdata.Cleaner`. It is frozen (hashable, safely shareable) and
validates itself on construction so that bad options fail loudly and early.
"""

from __future__ import annotations

import dataclasses
import difflib
from dataclasses import dataclass

_IMPUTE_CHOICES = (None, "auto", "mean", "median", "mode")
_OUTLIER_CHOICES = (None, "clip", "flag")
_OUTLIER_METHODS = ("iqr", "zscore")

#: Default outlier factor per method: 1.5×IQR (Tukey) or 3.0 standard deviations.
_DEFAULT_FACTOR = {"iqr": 1.5, "zscore": 3.0}


@dataclass(frozen=True)
class CleanConfig:
    """Options controlling what :func:`freshdata.clean` does.

    Defaults are conservative: steps that only repair representation
    (whitespace, sentinel strings, wrong dtypes, exact duplicate rows,
    structurally empty rows/columns) are on; steps that change the *statistics*
    of the data (imputation, outlier handling, lossy downcasting) are opt-in.
    """

    #: Normalize column names to snake_case and deduplicate collisions.
    column_names: bool = True
    #: Drop rows where every cell is missing.
    drop_empty_rows: bool = True
    #: Drop columns where every cell is missing.
    drop_empty_columns: bool = True
    #: Drop columns with a single distinct value (off by default).
    drop_constant_columns: bool = False
    #: Trim leading/trailing whitespace in text cells.
    strip_whitespace: bool = True
    #: Replace sentinel strings ("N/A", "null", "-", …) with missing values.
    normalize_sentinels: bool = True
    #: Additional sentinel strings to treat as missing (case-insensitive).
    extra_sentinels: tuple[str, ...] = ()
    #: Infer better dtypes for text columns (numeric, datetime, boolean).
    fix_dtypes: bool = True
    #: Fraction of non-missing values that must parse for a numeric conversion.
    numeric_threshold: float = 0.95
    #: Fraction of non-missing values that must parse for a datetime conversion.
    datetime_threshold: float = 0.95
    #: Drop exact duplicate rows (keeps the first occurrence).
    drop_duplicates: bool = True
    #: Restrict duplicate detection to these columns (post-rename names).
    duplicate_subset: tuple[str, ...] | None = None
    #: Missing-value imputation: None (off), "auto", "mean", "median", "mode".
    impute: str | None = None
    #: Outlier handling for numeric columns: None (off), "clip", "flag".
    outliers: str | None = None
    #: Outlier detection method: "iqr" or "zscore".
    outlier_method: str = "iqr"
    #: Detection factor; defaults to 1.5 for "iqr" and 3.0 for "zscore".
    outlier_factor: float | None = None
    #: Downcast numerics and convert low-cardinality text to category.
    optimize_memory: bool = False
    #: Max unique/total ratio for object→category conversion.
    category_threshold: float = 0.5
    #: Reset the index to 0..n-1 after cleaning (off: original labels kept).
    reset_index: bool = False
    #: Sample size used to cheaply pre-screen expensive type inference.
    sample_size: int = 10_000
    #: Seed for the (rare) sampling used during inference pre-screening.
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.impute not in _IMPUTE_CHOICES:
            raise ValueError(f"impute must be one of {_IMPUTE_CHOICES}, got {self.impute!r}")
        if self.outliers not in _OUTLIER_CHOICES:
            raise ValueError(f"outliers must be one of {_OUTLIER_CHOICES}, got {self.outliers!r}")
        if self.outlier_method not in _OUTLIER_METHODS:
            raise ValueError(
                f"outlier_method must be one of {_OUTLIER_METHODS}, got {self.outlier_method!r}"
            )
        for name in ("numeric_threshold", "datetime_threshold", "category_threshold"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
        if self.outlier_factor is not None and not self.outlier_factor > 0:
            raise ValueError(f"outlier_factor must be > 0, got {self.outlier_factor!r}")
        if self.sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {self.sample_size!r}")
        if not all(isinstance(s, str) for s in self.extra_sentinels):
            raise TypeError("extra_sentinels must be strings")
        # Normalize user-facing conveniences onto the frozen instance.
        object.__setattr__(
            self, "extra_sentinels", tuple(s.casefold().strip() for s in self.extra_sentinels)
        )
        if self.duplicate_subset is not None:
            object.__setattr__(self, "duplicate_subset", tuple(self.duplicate_subset))

    @property
    def resolved_outlier_factor(self) -> float:
        """The outlier factor in effect, applying the per-method default."""
        if self.outlier_factor is not None:
            return self.outlier_factor
        return _DEFAULT_FACTOR[self.outlier_method]


_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(CleanConfig))


def merge_options(base: CleanConfig | None, **options: object) -> CleanConfig:
    """Build a config from *base* (or defaults) plus keyword overrides.

    Unknown option names raise :class:`TypeError` with a "did you mean"
    suggestion, so typos never silently fall back to defaults.
    """
    unknown = sorted(set(options) - _FIELD_NAMES)
    if unknown:
        hints = []
        for name in unknown:
            match = difflib.get_close_matches(name, _FIELD_NAMES, n=1)
            hints.append(f"{name!r}" + (f" (did you mean {match[0]!r}?)" if match else ""))
        raise TypeError(
            f"unknown option(s): {', '.join(hints)}. "
            f"Valid options: {', '.join(sorted(_FIELD_NAMES))}"
        )
    if base is None:
        return CleanConfig(**options)  # type: ignore[arg-type]
    if not isinstance(base, CleanConfig):
        raise TypeError(f"config must be a CleanConfig, got {type(base).__name__}")
    return dataclasses.replace(base, **options)  # type: ignore[arg-type]
