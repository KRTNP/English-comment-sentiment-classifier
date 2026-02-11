import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Load HF_TOKEN from .env file if it exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        # If python-dotenv not installed, try reading manually
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                break


def main():
    # Auto-detect device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True, help="Directory of Stage 1 binary checkpoint")
    ap.add_argument(
        "--require_base_checkpoint",
        action="store_true",
        help="Fail if --base_checkpoint is missing/empty or cannot be loaded",
    )
    ap.add_argument("--cornell_csv", required=True, help="CSV with columns: text,label where label in {0,1,2}")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--hf_backbone",
        default="distilbert-base-uncased",
        help="HF model id to use if local checkpoint lacks tokenizer/model",
    )
    ap.add_argument("--epochs", type=int, default=3)
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
    ap.add_argument("--save_error_cases", type=int, default=200, help="Max misclassified samples to save")
    ap.add_argument("--class_weighting", choices=["none", "inverse", "inverse_sqrt"], default="none")
    ap.add_argument("--early_stopping_patience", type=int, default=0)
    ap.add_argument("--save_total_limit", type=int, default=2)
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

    df = pd.read_csv(args.cornell_csv)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Cornell CSV must have columns: text,label")

    # Split train/val/test
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.cast_column("label", ClassLabel(names=["neg", "neu", "pos"]))
    ds = ds.train_test_split(test_size=0.2, seed=args.seed, stratify_by_column="label")
    train = ds["train"]
    test = ds["test"]
    train = train.train_test_split(test_size=0.125, seed=args.seed, stratify_by_column="label")  # 70/10/20
    train_ds, val_ds = train["train"], train["test"]

    # Compute class weights from train split (for imbalance handling).
    labels_train = np.asarray(train_ds["label"])
    class_counts = np.bincount(labels_train, minlength=3).astype(np.float64)
    inv = 1.0 / np.maximum(class_counts, 1.0)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    if args.class_weighting == "inverse":
        raw_weights = inv
    elif args.class_weighting == "inverse_sqrt":
        raw_weights = inv_sqrt
    else:
        raw_weights = np.ones_like(inv)
    class_weights_np = raw_weights * (len(raw_weights) / raw_weights.sum())
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    base_ckpt = Path(args.base_checkpoint)
    checkpoint_has_files = base_ckpt.exists() and base_ckpt.is_dir() and any(p.is_file() for p in base_ckpt.iterdir())
    if args.require_base_checkpoint and not checkpoint_has_files:
        raise FileNotFoundError(
            f"Required base checkpoint is missing/empty: '{args.base_checkpoint}'. "
            "Expected files like config.json and model.safetensors/pytorch_model.bin."
        )

    # Try to load a fast tokenizer from the provided checkpoint. If the checkpoint
    # is empty or the tokenizer can't be instantiated, fall back to the HF backbone.
    tokenizer = None
    if checkpoint_has_files:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint, use_fast=True)
        except Exception as e:
            print("Warning: failed to load fast tokenizer from checkpoint; will try slow tokenizer. Error:", e)
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint, use_fast=False)
            except Exception as e2:
                print("Warning: failed to load slow tokenizer from checkpoint. Error:", e2)

    if tokenizer is None:
        if args.require_base_checkpoint:
            raise RuntimeError(f"Tokenizer loading from required checkpoint failed: '{args.base_checkpoint}'")
        print(f"Loading tokenizer from HF backbone '{args.hf_backbone}' instead of checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_backbone, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    test_ds = test.map(tok, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load encoder weights from stage1; re-init classification head for 3 labels.
    if checkpoint_has_files:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.base_checkpoint,
                num_labels=3,
                ignore_mismatched_sizes=True,
            )
        except Exception:
            if args.require_base_checkpoint:
                raise
            print(
                f"Warning: failed to load model from checkpoint '{args.base_checkpoint}'. "
                f"Falling back to '{args.hf_backbone}'."
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.hf_backbone,
                num_labels=3,
            )
    else:
        print(f"Checkpoint '{args.base_checkpoint}' appears empty - loading model from '{args.hf_backbone}'")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.hf_backbone,
            num_labels=3,
        )

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
        save_total_limit=args.save_total_limit,
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

    class WeightedTrainer(Trainer):
        def __init__(self, *wargs, class_weights=None, **wkwargs):
            super().__init__(*wargs, **wkwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if labels is None or self.class_weights is None:
                loss = outputs.get("loss")
            else:
                weight = self.class_weights.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=(class_weights if args.class_weighting != "none" else None),
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = trainer.evaluate(test_ds)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Detailed test diagnostics: per-class metrics, confusion matrix, and error cases.
    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    y_true = np.asarray(pred_out.label_ids)
    y_pred = np.argmax(logits, axis=-1)

    # Stable softmax on logits for confidence analysis.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    pred_conf = probs[np.arange(len(y_pred)), y_pred]

    label_names = ["neg", "neu", "pos"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    per_class = []
    for idx, name in enumerate(label_names):
        per_class.append(
            {
                "label_id": idx,
                "label_name": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    detail = {
        "label_names": label_names,
        "class_counts_train": class_counts.astype(int).tolist(),
        "class_weighting": args.class_weighting,
        "class_weights": class_weights_np.tolist(),
        "confusion_matrix": cm.tolist(),  # rows=true labels, cols=predicted labels
        "per_class_metrics": per_class,
    }
    (out_dir / "test_detailed_metrics.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")

    texts = test_ds["text"]
    error_rows = []
    for i in range(len(y_true)):
        if int(y_true[i]) == int(y_pred[i]):
            continue
        error_rows.append(
            {
                "text": texts[i],
                "true_label_id": int(y_true[i]),
                "true_label": label_names[int(y_true[i])],
                "pred_label_id": int(y_pred[i]),
                "pred_label": label_names[int(y_pred[i])],
                "pred_confidence": float(pred_conf[i]),
            }
        )

    # Save highest-confidence mistakes first (typically the most useful failures).
    error_rows.sort(key=lambda r: r["pred_confidence"], reverse=True)
    if args.save_error_cases > 0:
        error_rows = error_rows[: args.save_error_cases]
    pd.DataFrame(error_rows).to_csv(out_dir / "test_error_cases.csv", index=False, encoding="utf-8")

    print("Saved final 3-class model to:", out_dir)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
