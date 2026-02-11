\
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def clean_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to IMDB Dataset.csv (columns: review,sentiment)")
    ap.add_argument("--out_dir", required=True, help="Output directory for train/val/test CSVs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError("Expected columns: review, sentiment")

    df["review"] = df["review"].astype(str).map(clean_html)
    df = df.drop_duplicates(subset=["review", "sentiment"]).reset_index(drop=True)

    # binary labels: 1=pos, 0=neg
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)

    train_df, test_df = train_test_split(
        df[["review", "label"]],
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.125,  # 0.125 of 0.8 => 0.1 overall
        random_state=args.seed,
        stratify=train_df["label"],
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    meta = {
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_val": len(val_df),
        "rows_test": len(test_df),
        "duplicates_removed": None,  # not tracked precisely
        "label_distribution_total": df["label"].value_counts().to_dict(),
    }
    (out_dir / "meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    print("Wrote:", out_dir)

if __name__ == "__main__":
    main()
