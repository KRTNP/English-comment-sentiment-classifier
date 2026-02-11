\
import argparse
from pathlib import Path
import tarfile
import urllib.request

SCALE_URL = "https://www.cs.cornell.edu/people/pabo/movie-review-data/scale_data.tar.gz"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output directory for Cornell scale dataset")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = out_dir / "scale_data.tar.gz"
    if not tgz_path.exists():
        print("Downloading:", SCALE_URL)
        urllib.request.urlretrieve(SCALE_URL, tgz_path)
    else:
        print("Found existing:", tgz_path)

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(path=out_dir)

    print("Done. Extracted directories should be inside:", out_dir)

if __name__ == "__main__":
    main()
