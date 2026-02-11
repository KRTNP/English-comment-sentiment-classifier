import argparse
from pathlib import Path
import tarfile
import urllib.request

ACL_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output directory, will contain aclImdb/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = out_dir / "aclImdb_v1.tar.gz"
    if not tgz_path.exists():
        print("Downloading:", ACL_URL)
        urllib.request.urlretrieve(ACL_URL, tgz_path)
    else:
        print("Found existing:", tgz_path)

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(path=out_dir)

    print("Done. Expected directory:", out_dir / "aclImdb")

if __name__ == "__main__":
    main()
