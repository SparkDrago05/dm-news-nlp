import pandas as pd
from pathlib import Path

# === CONFIG: change this to the folder where your CSVs are ===
DATA_DIR = Path("../Labs/datasets")

FILES = {
    "pakistan_today": "pakistan_today(full-data).csv",
    "tribune": "tribune(full-data).csv",
    "dawn": "dawn (full-data).csv",
    "daily_times": "daily_times(full-data).csv",
    # "business_reorder": "business_reorder(2020-2023).csv",
}


def load_file(label, filename):
    path = DATA_DIR / filename
    print(f"\n\n========== Loading {label} from {path} ==========")

    # encodings to try (order matters)
    encodings_to_try = ["utf-8", "cp1252", "latin1"]

    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                on_bad_lines="skip",
                encoding_errors="strict"  # fail fast so we can try next encoding
            )
            print(f"[OK] Loaded with encoding: {enc}")
            break
        except UnicodeDecodeError as e:
            last_err = e
            print(f"[FAIL] {enc}: {e}")
    else:
        # if all encodings failed
        raise last_err

    # Drop useless unnamed columns created by extra commas
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    print(f"{label}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def summarize_df(label, df):
    print(f"\n----- SUMMARY: {label} -----")
    print("\nColumns & dtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    if "categories" in df.columns:
        print("\nTop 10 'categories' values:")
        print(df["categories"].value_counts().head(10))

    if "source" in df.columns:
        print("\nTop 'source' values:")
        print(df["source"].value_counts().head(10))

    # Basic text length info if description exists
    if "description" in df.columns:
        desc_len = df["description"].astype(str).str.len()
        print("\nDescription length (chars):")
        print("  mean:", desc_len.mean())
        print("  median:", desc_len.median())
        print("  min:", desc_len.min())
        print("  max:", desc_len.max())

    # Show a few example rows
    print("\nSample 3 rows:")
    print(df.sample(3, random_state=42))


def main():
    all_dfs = []

    for label, filename in FILES.items():
        df = load_file(label, filename)
        summarize_df(label, df)
        df["__file__"] = label
        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True, sort=False)

    print("\n\n========== COMBINED DATASET OVERVIEW ==========")
    print(f"Total rows across all files: {len(all_df)}")
    print("Columns:", list(all_df.columns))

    if "source" in all_df.columns and "categories" in all_df.columns:
        print("\nTop 20 (source, categories) combinations:")
        print(all_df.groupby(["source", "categories"]).size().sort_values(ascending=False).head(20))

    # Tiny subset: first 2 rows per file
    subset = all_df.groupby("__file__").head(2).copy()
    subset = subset.drop(columns=["__file__"])

    print("\n\n========== SMALL SUBSET (2 ROWS PER FILE) ==========")
    # This prints in a copy-pastable markdown table for you to send here
    try:
        print(subset.to_markdown(index=False))
    except Exception:
        # fallback if tabulate is not installed
        print(subset.head(10))


if __name__ == "__main__":
    main()
