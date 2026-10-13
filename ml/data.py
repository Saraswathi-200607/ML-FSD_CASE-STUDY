"""Dataset loading, validation, cleaning and the train/test split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_CHOICES, FEATURES, NUMERIC_FEATURES, NUMERIC_LIMITS,
    TARGET, TARGET_MAP,
)


class DatasetSchemaError(Exception):
    """The CSV does not look like heart_2020_cleaned.csv."""


def load_dataset(csv_path):
    """Read the CSV, verify its schema, clean it. Returns (df, report dict).

    Cleaning steps (all done BEFORE splitting, none of them learn from data):
      1. keep only the columns we use,
      2. verify the target and every categorical column contain only the
         values we expect (otherwise stop: the web form must match exactly),
      3. drop rows with missing values in used columns (the 2020 file has none,
         but we check rather than assume),
      4. drop rows whose numeric values are outside the supported limits,
      5. drop exact duplicate rows so copies cannot land in both train & test.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download it from Kaggle and put it in the data/ folder (see README)."
        )
    raw = pd.read_csv(csv_path)
    report = {"csv": csv_path.name, "rows_raw": int(len(raw))}

    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise DatasetSchemaError(
            f"Missing expected columns: {missing}\n"
            f"Columns found in your CSV: {list(raw.columns)}\n"
            "This project expects the 2020 file 'heart_2020_cleaned.csv'. The newer "
            "2022 file has different columns. Either use the 2020 file, or edit "
            "ml/config.py to match your file."
        )

    df = raw[needed].copy()

    bad_target = set(df[TARGET].dropna().unique()) - set(TARGET_MAP)
    if bad_target:
        raise DatasetSchemaError(f"Unexpected values in target '{TARGET}': {bad_target}")

    for col, allowed in CATEGORICAL_CHOICES.items():
        unexpected = set(df[col].dropna().astype(str).unique()) - set(allowed)
        if unexpected:
            raise DatasetSchemaError(
                f"Column '{col}' has values not listed in ml/config.py: {unexpected}\n"
                f"Allowed: {allowed}"
            )

    n = len(df)
    df = df.dropna()
    report["dropped_missing"] = n - len(df)

    n = len(df)
    for col in NUMERIC_FEATURES:
        lo, hi = NUMERIC_LIMITS[col]
        df = df[pd.to_numeric(df[col], errors="coerce").between(lo, hi)]
    report["dropped_out_of_range"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["dropped_duplicates"] = n - len(df)

    df[TARGET] = df[TARGET].map(TARGET_MAP).astype(int)
    report["rows_clean"] = int(len(df))
    report["positive_rate"] = float(df[TARGET].mean())
    return df, report


def split_data(df, test_size=0.2, seed=42):
    """Stratified split: both parts keep the same proportion of positives."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
