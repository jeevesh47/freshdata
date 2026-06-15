from typing import Any


def is_polars_frame(df: Any) -> bool:
    return False

def to_pandas(df: Any) -> Any:
    return df

def from_pandas(df: Any, original: Any = None) -> Any:
    return df

def _polars_module() -> Any:
    raise ImportError("Polars support has been removed.")
