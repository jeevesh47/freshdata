"""freshdata — fast, safe, automatic data cleaning for real-world tabular data.

>>> import freshdata as fd
>>> cleaned = fd.clean(df)
>>> cleaned, report = fd.clean(df, report=True)
>>> print(fd.profile(df))

Design principles
-----------------
- **No surprises.** Defaults only fix representation (whitespace, sentinel
  strings, wrong dtypes, exact duplicates, empty rows/columns). Anything that
  changes your data's statistics is opt-in.
- **Everything is reported.** Each transformation is recorded with the column
  and the number of affected cells.
- **Never mutates input.** ``clean`` returns a new frame; profiling is
  read-only.
- **Fast by construction.** Vectorized pandas operations only, with
  sample-based pre-screening so type inference stays cheap on large frames.
"""

from .api import clean, profile
from .cleaner import Cleaner
from .config import CleanConfig
from .profile import ColumnProfile, Profile
from .report import Action, CleanReport

__version__ = "0.1.0"

__all__ = [
    "Action",
    "CleanConfig",
    "CleanReport",
    "Cleaner",
    "ColumnProfile",
    "Profile",
    "__version__",
    "clean",
    "profile",
]
