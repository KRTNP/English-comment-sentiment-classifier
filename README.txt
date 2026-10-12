# English comment sentiment classifier

A training pipeline for a 3-class sentiment classifier (negative, neutral, positive) on English reviews and comments, plus a FastAPI server for deploying the model on Render.

## Why two training stages

IMDb data only has positive and negative labels, so it can't teach the model what neutral looks like. The Cornell scale dataset does have a neutral class, with 3-class labels (0, 1, 2) derived from normalized ratings. Training happens in two stages:

1. **Binary (positive/negative)** on IMDb-style data, to adapt the encoder to sentiment in general.
2. **3-class** on the Cornell scale dataset, to learn neutral explicitly.

## Folder layout

- `data/`: datasets (you create this)
- `outputs/`: model checkpoints (created by the scripts)
- `scripts/`: training, evaluation, and inference scripts

## Setup

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

## Data

You can use any combination of the datasets below.

### IMDb CSV

A single CSV of labeled IMDb reviews. The script strips HTML tags, removes duplicates, splits the data 70/10/20 into train, validation, and test, and writes the result to `data/imdb_csv/`.

```bash
python scripts/prepare_imdb_csv.py --input "/mnt/data/IMDB Dataset.csv" --out_dir "data/imdb_csv"
```

### Stanford Large Movie Review Dataset (ACL IMDb)

Download and extract `aclImdb_v1.tar.gz`, then convert it to CSV:

```bash
python scripts/download_acl_imdb.py --out_dir "data/aclImdb"
python scripts/prepare_acl_imdb.py --acl_dir "data/aclImdb/aclImdb" --out_dir "data/aclImdb_csv"
```

### Cornell scale dataset

Download and extract `scale_data.tar.gz`, then build `data/cornell_scale/scale_3class.csv`:

```bash
python scripts/download_cornell_scale.py --out_dir "data/cornell_scale"
python scripts/prepare_cornell_scale.py --scale_dir "data/cornell_scale" --out_csv "data/cornell_scale/scale_3class.csv"
```

### Rotten Tomatoes (optional)

Stage 1 can mix in Rotten Tomatoes sentence polarity data. It's loaded through Hugging Face `datasets`, so there's no download script.

## Training

### Stage 1: binary

Uses the IMDb CSV if you prepared it, otherwise the ACL IMDb CSV. Rotten Tomatoes can be mixed in optionally.

```bash
python scripts/train_stage1_binary.py \
  --model_name "microsoft/deberta-v3-base" \
  --imdb_csv_dir "data/imdb_csv" \
  --out_dir "outputs/stage1_binary" \
  --max_length 256 \
  --epochs 2
```

### Stage 2: 3-class

Starts from the Stage 1 encoder weights, replaces the classification head with a new 3-label head, and trains on the Cornell scale dataset.

```bash
python scripts/train_stage2_3class.py \
  --base_checkpoint "outputs/stage1_binary" \
  --cornell_csv "data/cornell_scale/scale_3class.csv" \
  --out_dir "outputs/final_3class" \
  --max_length 256 \
  --epochs 3
```

## Evaluation

Binary test on IMDb:

```bash
python scripts/evaluate.py --task binary --model_dir "outputs/stage1_binary" --imdb_csv_dir "data/imdb_csv" --max_length 256
```

3-class test on Cornell:

```bash
python scripts/evaluate.py --task 3class --model_dir "outputs/final_3class" --cornell_csv "data/cornell_scale/scale_3class.csv" --max_length 256
```

Stage 1 with DeBERTa or RoBERTa usually reaches high accuracy on in-domain IMDb-style test data. Stage 2 macro-F1 will be lower than the binary score, and how much lower depends on how strictly neutral is defined.

## Inference

Single text:

```bash
python scripts/predict.py --model_dir "outputs/final_3class" --text "This is amazing."
```

A file with one comment per line:

```bash
python scripts/predict.py --model_dir "outputs/final_3class" --input_file "comments.txt" --output_file "predictions.csv"
```

`max_length=256` works for most comments. Use 384 if yours are longer.

`predict.py` also supports an uncertainty rule: if the highest class probability is below 0.55, the label is forced to neutral. This is useful in production, where a low-confidence positive or negative is usually worse than a neutral.

## Deploying the API on Render (free plan)

The repo includes:

- `api/server.py`: the FastAPI app
- `requirements-api.txt`: serving dependencies
- `render.yaml`: the Render blueprint
- `runtime.txt`: the Python version

The free plan is CPU-only. GitHub also rejects files over 100 MB, and this model's `model.safetensors` is larger than that. The recommended approach is to upload the model to the Hugging Face Hub and set `MODEL_ID`. Alternatively, store the model with Git LFS and point `MODEL_DIR` at its path in the repo.

### Steps

1. Push the repo to GitHub.
2. In Render, create a service from `render.yaml` (Blueprint), or create a Web Service manually.
3. Build command: `pip install -r requirements-api.txt`
4. Start command: `uvicorn api.server:app --host 0.0.0.0 --port $PORT`
5. Set the environment variables:
   - `MODEL_ID=<your-hf-model-id>` (recommended)
   - `HF_TOKEN=<your-hf-token>` (only if the model is private)
   - `MODEL_DIR=outputs/seed_stability_stage2_best/seed_52` (used when the model files are on disk)
   - `MAX_LENGTH=256`

### Endpoints

`GET /health`

`POST /predict`

```json
{"text":"This movie is great","max_length":256}
```

`POST /predict/batch`

```json
{"texts":["great movie","boring plot"],"max_length":256}
```

## Licensing

Follow each dataset's terms of use, and the terms of service of any platform you collect comments from.
