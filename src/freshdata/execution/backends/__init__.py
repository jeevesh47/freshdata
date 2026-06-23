"""Concrete execution backends for freshdata.

Backends are imported lazily by :class:`freshdata.execution.EngineSelector` so
that ``import freshdata`` never requires polars or duckdb.
"""
