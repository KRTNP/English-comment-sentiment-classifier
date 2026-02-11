import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_csv_dir(csv_dir: Path):
    train = pd.read_csv(csv_dir / "train.csv")
    val = pd.read_csv(csv_dir / "val.csv")
    test = pd.read_csv(csv_dir / "test.csv")
    # normalize columns
    for df in (train, val, test):
        if "review" in df.columns and "text" not in df.columns:
            df.rename(columns={"review": "text"}, inplace=True)
    return Dataset.from_pandas(train), Dataset.from_pandas(val), Dataset.from_pandas(test)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--imdb_csv_dir", required=True, help="Directory containing train.csv/val.csv/test.csv with text,label")
    ap.add_argument("--add_rotten_tomatoes", action="store_true", help="Mix in Rotten Tomatoes sentence polarity (binary)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="DataLoader workers (Windows users may keep this at 0 for stability)",
    )
    ap.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision")
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision")
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16")

    # Reproducibility across Python, NumPy, torch, and transformers.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_csv_dir(Path(args.imdb_csv_dir))

    if args.add_rotten_tomatoes:
        rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        # HF dataset uses: text, label where label 0=neg 1=pos
        rt_train = rt["train"].rename_column("text", "text").rename_column("label", "label")
        rt_test = rt["test"].rename_column("text", "text").rename_column("label", "label")
        # split small validation from rt_train
        rt_train = rt_train.train_test_split(test_size=0.1, seed=args.seed, stratify_by_column="label")
        train_ds = concatenate_datasets([train_ds, rt_train["train"]])
        val_ds = concatenate_datasets([val_ds, rt_train["test"]])
        test_ds = concatenate_datasets([test_ds, rt_test])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    targs = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        seed=args.seed,
        data_seed=args.seed,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_pin_memory=(device.type == "cuda"),
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # quick test eval
    metrics = trainer.evaluate(test_ds)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved model to:", out_dir)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
