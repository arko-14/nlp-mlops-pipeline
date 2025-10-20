# training/data_prep.py
import argparse
import os
import sys
import pandas as pd

def normalize_ag_news(df: pd.DataFrame) -> pd.DataFrame:
    # Expect: Class Index, Title, Description
    required = {"Class Index", "Title", "Description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Combine title + description into 'text'
    text = df["Title"].astype(str).str.strip() + " " + df["Description"].astype(str).str.strip()

    # Map labels to 0..3 (AG News is 1..4)
    labels = df["Class Index"].astype(int) - 1
    if labels.min() < 0 or labels.max() > 3:
        raise ValueError("Unexpected labels. Expected Class Index in {1,2,3,4}.")

    out = pd.DataFrame({"text": text, "label": labels})
    return out

def main(src_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    train_fp = os.path.join(src_dir, "train.csv")
    test_fp  = os.path.join(src_dir, "test.csv")

    if not os.path.exists(train_fp) or not os.path.exists(test_fp):
        raise FileNotFoundError(
            f"Expected CSVs at:\n  {train_fp}\n  {test_fp}\n"
            "Put your AG News train/test CSVs in the data/ folder with these exact names."
        )

    # Read with safe defaults for Windows/Excel-exported CSVs
    train_df_raw = pd.read_csv(train_fp, encoding="utf-8", engine="python")
    test_df_raw  = pd.read_csv(test_fp,  encoding="utf-8", engine="python")

    train_df = normalize_ag_news(train_df_raw)
    test_df  = normalize_ag_news(test_df_raw)

    out_train = os.path.join(out_dir, "train.csv")
    out_test  = os.path.join(out_dir, "test.csv")
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)

    # Small confirmation
    print(f"[OK] Wrote {out_train} ({len(train_df)} rows)")
    print(f"[OK] Wrote {out_test}  ({len(test_df)} rows)")
    print(train_df.head(3).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare AG News CSVs for training.")
    ap.add_argument("--src", required=True, help="Folder containing train.csv and test.csv")
    ap.add_argument("--out", required=True, help="Output folder for processed CSVs")
    args = ap.parse_args()
    try:
        main(args.src, args.out)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
