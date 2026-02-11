import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def get_token() -> str:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    for env_path in [Path(".env"), Path(".env.example")]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN not found. Set env var HF_TOKEN or put it in .env")


def main():
    ap = argparse.ArgumentParser(description="Publish local model folder to Hugging Face Hub")
    ap.add_argument(
        "--model_dir",
        default="outputs/seed_stability_stage2_best/seed_52",
        help="Local model directory to upload",
    )
    ap.add_argument(
        "--repo_id",
        default=None,
        help="Target repo id (e.g. username/model-name). If omitted, auto-uses username/sentiment-english-3class-seed52",
    )
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    args = ap.parse_args()

    token = get_token()
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = args.repo_id or f"{user}/sentiment-english-3class-seed52"

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    readme = model_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# sentiment-english-3class-seed52\n\n"
            "DistilBERT-based 3-class sentiment model (negative/neutral/positive).\n",
            encoding="utf-8",
        )

    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model artifacts for Render deployment",
    )
    print("Uploaded:", repo_id)
    print(f"URL: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
