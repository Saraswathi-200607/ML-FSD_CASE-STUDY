"""
Inspect a CSV before modelling.   python ml/inspect_dataset.py [data/file.csv]
"""
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 200)


def section(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        csvs = sorted(DATA_DIR.glob("*.csv"))
        if not csvs:
            print(f"No CSV files in {DATA_DIR}. Download the dataset first.")
            sys.exit(1)
        if len(csvs) > 1:
            print("Several CSVs found; pass one explicitly:")
            for c in csvs:
                print("  ", c.name)
            sys.exit(0)
        path = csvs[0]
    df = pd.read_csv(path)
    section(f"FILE: {path.name}  rows={len(df):,} cols={df.shape[1]}")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "n_unique": df.nunique()}))
    section("FIRST 5 ROWS"); print(df.head())
    section("DUPLICATES"); print(f"{int(df.duplicated().sum()):,} exact duplicate rows")
    section("CATEGORICAL VALUES")
    for c in df.select_dtypes(exclude="number").columns:
        print(f"\n[{c}]"); print(df[c].value_counts(dropna=False).head(25).to_string())
    section("NUMERIC SUMMARY"); print(df.describe().T)
    section("TARGET CANDIDATES")
    for c in [c for c in df.columns if "heart" in c.lower()]:
        print(f"\n[{c}]"); print(df[c].value_counts(normalize=True, dropna=False).round(4))


if __name__ == "__main__":
    main()
