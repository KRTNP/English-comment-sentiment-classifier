Sentiment (Negative/Neutral/Positive) training pipeline for English internet comments
==========================================================================

Goal
- Train a 3-class sentiment classifier: negative / neutral / positive.
- Primary domain: English reviews/comments.
- Uses staged fine-tuning:
  Stage 1: binary (pos/neg) on IMDb-style data to adapt the encoder.
  Stage 2: 3-class on Cornell scale dataset (neutral is a real class) to learn neutral.

Why staged
- IMDb datasets are binary only.
- Cornell scale dataset provides 3-class labels derived from normalized ratings (0,1,2).
- Stage 1 improves general sentiment signal; Stage 2 teaches neutral explicitly.

Folder layout
- data/ (created by you)
- outputs/ (created by scripts)
- scripts/ (this folder)

1) Setup
- Python 3.10+ recommended
- Install deps:
  pip install -r requirements.txt

2) Data options (you can use any subset)
A) IMDb CSV (your uploaded file)
- Input: /mnt/data/IMDB Dataset.csv (or your own path)
- Script will:
  - remove HTML tags
  - drop duplicates
  - create train/val/test splits (70/10/20)
  - output to data/imdb_csv/

Command:
  python scripts/prepare_imdb_csv.py --input "/mnt/data/IMDB Dataset.csv" --out_dir "data/imdb_csv"

B) Stanford ACL IMDb (Large Movie Review Dataset)
- Downloads and extracts aclImdb_v1.tar.gz
Command:
  python scripts/download_acl_imdb.py --out_dir "data/aclImdb"

Then prepare CSV:
  python scripts/prepare_acl_imdb.py --acl_dir "data/aclImdb/aclImdb" --out_dir "data/aclImdb_csv"

C) Cornell scale dataset (3-class with neutral)
- Downloads and extracts scale_data.tar.gz
- Produces data/cornell_scale/scale_3class.csv

Command:
  python scripts/download_cornell_scale.py --out_dir "data/cornell_scale"
  python scripts/prepare_cornell_scale.py --scale_dir "data/cornell_scale" --out_csv "data/cornell_scale/scale_3class.csv"

D) Rotten Tomatoes (optional binary add-on)
- Pulled via Hugging Face datasets in Stage 1 (no download script needed)

3) Training (recommended path)
Stage 1 (binary)
- Uses IMDb CSV if provided; otherwise uses ACL IMDb CSV.
- Optionally mixes Rotten Tomatoes sentence polarity.

Command:
  python scripts/train_stage1_binary.py \
    --model_name "microsoft/deberta-v3-base" \
    --imdb_csv_dir "data/imdb_csv" \
    --out_dir "outputs/stage1_binary" \
    --max_length 256 \
    --epochs 2

Stage 2 (3-class with neutral)
- Continues from Stage 1 encoder weights, re-initializes classification head for 3 labels.
- Trains on Cornell scale dataset.

Command:
  python scripts/train_stage2_3class.py \
    --base_checkpoint "outputs/stage1_binary" \
    --cornell_csv "data/cornell_scale/scale_3class.csv" \
    --out_dir "outputs/final_3class" \
    --max_length 256 \
    --epochs 3

4) Evaluation
Binary test (IMDb):
  python scripts/evaluate.py --task binary --model_dir "outputs/stage1_binary" --imdb_csv_dir "data/imdb_csv" --max_length 256

3-class test (Cornell):
  python scripts/evaluate.py --task 3class --model_dir "outputs/final_3class" --cornell_csv "data/cornell_scale/scale_3class.csv" --max_length 256

5) Inference
Single text:
  python scripts/predict.py --model_dir "outputs/final_3class" --text "This is amazing."

File (one comment per line):
  python scripts/predict.py --model_dir "outputs/final_3class" --input_file "comments.txt" --output_file "predictions.csv"

Notes / practical defaults
- max_length=256 is a good default; if your comments are longer, use 384.
- For production safety, you can add an "uncertainty neutral" rule:
  if max(probabilities) < 0.55 => force label = neutral.
  (supported in predict.py)

Expected performance (ballpark, will vary)
- Stage 1 binary on IMDb-like data with DeBERTa/RoBERTa: typically high accuracy on in-domain test.
- Stage 2 3-class: macro-F1 depends on how strict neutral is; expect lower than binary.

Licensing / terms
- Respect dataset terms of use and any platform ToS if you add internet comments.

6) Deploy API on Render (Free Plan)
- This repo now includes:
  - `api/server.py` (FastAPI app)
  - `requirements-api.txt` (serving dependencies)
  - `render.yaml` (Render blueprint)
  - `runtime.txt` (Python version)

Important:
- Render Free is CPU-only.
- GitHub ปกติไม่รองรับไฟล์ >100MB ต่อไฟล์
  (`model.safetensors` ของโมเดลนี้ใหญ่กว่า 100MB)
- ทางที่แนะนำ: อัปโหลดโมเดลไป Hugging Face Hub แล้วตั้ง `MODEL_ID`
- ทางเลือก: ใช้ Git LFS แล้วชี้ `MODEL_DIR` ไป path ใน repo

Deploy steps:
1. Push this repo to GitHub.
2. In Render, create service from `render.yaml` (Blueprint) or create a Web Service manually.
3. Build command:
   `pip install -r requirements-api.txt`
4. Start command:
   `uvicorn api.server:app --host 0.0.0.0 --port $PORT`
5. Environment variables:
   - `MODEL_ID=<your-hf-model-id>` (recommended)
   - `HF_TOKEN=<your-hf-token>` (if model is private)
   - `MODEL_DIR=outputs/seed_stability_stage2_best/seed_52` (fallback when files exist on disk)
   - `MAX_LENGTH=256`

API endpoints:
- `GET /health`
- `POST /predict`
  body example:
  `{"text":"This movie is great","max_length":256}`
- `POST /predict/batch`
  body example:
  `{"texts":["great movie","boring plot"],"max_length":256}`
