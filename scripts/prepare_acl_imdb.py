
\
import argparse
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

def clean_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_split(split_dir: Path):
    rows = []
    for label_name, label in [("neg", 0), ("pos", 1)]:
        folder = split_dir / label_name
        for fp in folder.glob("*.txt"):
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            rows.append({"review": clean_html(txt), "label": label})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acl_dir", required=True, help="Path to aclImdb directory (contains train/test)")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSVs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    acl_dir = Path(args.acl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(acl_dir / "train")
    test_df = load_split(acl_dir / "test")

    # Create validation from train
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=args.seed, stratify=train_df["label"]
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("Wrote:", out_dir)

if __name__ == "__main__":
    main()
