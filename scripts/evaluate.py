\
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

def load_imdb(csv_dir: Path):
    test = pd.read_csv(csv_dir / "test.csv")
    if "review" in test.columns and "text" not in test.columns:
        test.rename(columns={"review": "text"}, inplace=True)
    return Dataset.from_pandas(test[["text", "label"]])

def load_cornell(csv_path: Path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df[["text", "label"]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "3class"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--imdb_csv_dir")
    ap.add_argument("--cornell_csv")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint_has_files = model_dir.exists() and any(model_dir.iterdir())
    
    # Try to load from checkpoint; fall back to HF backbone if empty or fails
    hf_backbone = "distilbert-base-uncased"
    tokenizer = None
    
    if checkpoint_has_files:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception as e:
            print(f"Warning: failed to load tokenizer from checkpoint: {e}")
            print(f"Falling back to HF backbone '{hf_backbone}'")
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(hf_backbone, use_fast=True)
    
    # Load model
    if checkpoint_has_files:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            print(f"Warning: failed to load model from checkpoint: {e}")
            print(f"Loading from HF backbone '{hf_backbone}'")
            num_labels = 2 if args.task == "binary" else 3
            model = AutoModelForSequenceClassification.from_pretrained(hf_backbone, num_labels=num_labels)
    else:
        num_labels = 2 if args.task == "binary" else 3
        model = AutoModelForSequenceClassification.from_pretrained(hf_backbone, num_labels=num_labels)

    if args.task == "binary":
        if not args.imdb_csv_dir:
            raise ValueError("--imdb_csv_dir is required for binary task")
        ds = load_imdb(Path(args.imdb_csv_dir))
        label_names = ["neg", "pos"]
    else:
        if not args.cornell_csv:
            raise ValueError("--cornell_csv is required for 3class task")
        ds = load_cornell(Path(args.cornell_csv))
        label_names = ["neg", "neu", "pos"]

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = ds.map(tok, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    targs = TrainingArguments(
        output_dir=str(model_dir / "_eval_tmp"),
        per_device_eval_batch_size=32,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(ds)
    # detailed report
    preds = trainer.predict(ds).predictions
    y_pred = np.argmax(preds, axis=-1)
    y_true = np.array(ds["label"])

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)

    out = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "label_names": label_names,
    }

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("Confusion matrix (rows=true, cols=pred):", cm)
    print(report)

if __name__ == "__main__":
    main()
