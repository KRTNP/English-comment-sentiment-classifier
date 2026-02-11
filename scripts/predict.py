\
import argparse
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS_3 = ["negative", "neutral", "positive"]
LABELS_2 = ["negative", "positive"]
DEFAULT_MODEL_DIR = "outputs/seed_stability_stage2_best/seed_52"

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict_texts(model, tokenizer, texts, max_length):
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
    probs = np.apply_along_axis(softmax, 1, logits)
    preds = probs.argmax(axis=1)
    return preds, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Path to model directory (default: {DEFAULT_MODEL_DIR})",
    )
    ap.add_argument("--text", default=None)
    ap.add_argument("--input_file", default=None, help="One comment per line")
    ap.add_argument("--output_file", default=None, help="CSV output (text,label,prob_*)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--uncertainty_neutral_threshold", type=float, default=None,
                    help="If set and max(prob)<threshold then force neutral (3-class only)")
    args = ap.parse_args()

    if (args.text is None) == (args.input_file is None):
        raise ValueError("Provide exactly one of --text or --input_file")

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    num_labels = model.config.num_labels
    label_names = LABELS_3 if num_labels == 3 else LABELS_2

    if args.text is not None:
        texts = [args.text]
    else:
        texts = Path(args.input_file).read_text(encoding="utf-8", errors="ignore").splitlines()
        texts = [t.strip() for t in texts if t.strip()]

    preds, probs = predict_texts(model, tokenizer, texts, args.max_length)

    out_rows = []
    for t, p_idx, p in zip(texts, preds, probs):
        label = label_names[p_idx]
        if num_labels == 3 and args.uncertainty_neutral_threshold is not None:
            if float(np.max(p)) < args.uncertainty_neutral_threshold:
                label = "neutral"
                p_idx = 1
        row = {"text": t, "label": label}
        for i, name in enumerate(label_names):
            row[f"prob_{name}"] = float(p[i])
        out_rows.append(row)

    if args.output_file:
        pd.DataFrame(out_rows).to_csv(args.output_file, index=False)
        print("Wrote:", args.output_file)
    else:
        for r in out_rows[:10]:
            print(r)

if __name__ == "__main__":
    main()
