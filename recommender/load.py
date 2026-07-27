"""Raw CSV -> typed DataFrame.

Loading is deliberately separate from cleaning: this module only fixes the
column contract and coerces types. Row filtering happens in ``clean.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender import config, schema


def load_raw(path: Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Read the raw dataset with correctly aligned columns.

    ``header=0`` skips the malformed published header; ``names=RAW_COLUMNS``
    supplies the true 40-field layout positionally. See ``schema`` for why.
    """
    path = Path(path or config.RAW_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the Steam dataset there, or pass the "
            f"committed sample at {config.SAMPLE_CSV}."
        )

    if validate:
        schema.validate_raw_shape(path)

    df = pd.read_csv(
        path,
        header=0,
        names=schema.RAW_COLUMNS,
        usecols=schema.USED_COLUMNS,
        index_col=False,
        low_memory=False,
    )
    df = df.rename(columns=schema.COLUMN_RENAME)
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce").astype("Int64")

    for col in schema.NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in schema.BOOLEAN_COLUMNS:
        df[col] = df[col].astype(str).str.strip().str.lower().eq("true")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")

    for col in ["name", "description", "developers", "publishers"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in schema.MULTIVALUE_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df
