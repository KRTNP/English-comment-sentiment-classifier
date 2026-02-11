\
import argparse
from pathlib import Path
import pandas as pd

def find_author_dirs(root: Path):
    # Author dirs contain files like subj.<author> and label.3class.<author>
    dirs = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if list(d.glob("subj.*")) and list(d.glob("label.3class.*")):
            dirs.append(d)
    return dirs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale_dir", required=True, help="Directory containing extracted Cornell scale dataset v1.0 (any level)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path for 3-class data")
    args = ap.parse_args()

    scale_dir = Path(args.scale_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    author_dirs = find_author_dirs(scale_dir)
    if not author_dirs:
        raise ValueError(
            "Could not find author directories. Ensure scale_data.tar.gz was extracted and --scale_dir points to the extraction folder."
        )

    for author_dir in author_dirs:
        subj_path = list(author_dir.glob("subj.*"))[0]
        author = subj_path.name.replace("subj.", "")

        label_path = author_dir / f"label.3class.{author}"
        if not label_path.exists():
            label_path = list(author_dir.glob("label.3class.*"))[0]

        subj_lines = subj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        label_lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        if len(subj_lines) != len(label_lines):
            raise ValueError(f"Line count mismatch in {author_dir} ({len(subj_lines)} vs {len(label_lines)})")

        for text, lab in zip(subj_lines, label_lines):
            lab = int(lab.strip())
            rows.append({"text": text.strip(), "label": lab, "source": f"cornell_scale:{author}"})

    df = pd.DataFrame(rows).dropna()
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv, "rows:", len(df))

if __name__ == "__main__":
    main()
