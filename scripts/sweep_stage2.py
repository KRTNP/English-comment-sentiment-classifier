import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Grid sweep for Stage 2 3-class training")
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--cornell_csv", required=True)
    ap.add_argument("--out_root", required=True, help="Root output directory for all runs")
    ap.add_argument("--lrs", default="1e-5,1.5e-5,2e-5,3e-5", help="Comma-separated learning rates")
    ap.add_argument("--batch_sizes", default="16,32", help="Comma-separated batch sizes")
    ap.add_argument("--epochs_list", default="2,3,4,5", help="Comma-separated epoch values")
    ap.add_argument("--max_lengths", default="128,256", help="Comma-separated max_length values")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--dataloader_num_workers", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--require_base_checkpoint", action="store_true")
    ap.add_argument("--save_error_cases", type=int, default=100)
    ap.add_argument("--class_weighting", choices=["none", "inverse", "inverse_sqrt"], default="inverse_sqrt")
    ap.add_argument("--early_stopping_patience", type=int, default=2)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--skip_existing", action="store_true", help="Skip runs that already have test_metrics.json")
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lrs = parse_list(args.lrs, float)
    bss = parse_list(args.batch_sizes, int)
    eps = parse_list(args.epochs_list, int)
    mls = parse_list(args.max_lengths, int)

    runs = []
    run_idx = 0
    for lr in lrs:
        for bs in bss:
            for ep in eps:
                for ml in mls:
                    run_idx += 1
                    run_name = f"run_{run_idx:03d}_lr{lr:g}_bs{bs}_ep{ep}_ml{ml}"
                    run_dir = out_root / run_name
                    metrics_path = run_dir / "test_metrics.json"
                    if args.skip_existing and metrics_path.exists():
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        runs.append(
                            {
                                "run_name": run_name,
                                "status": "skipped_existing",
                                "lr": lr,
                                "batch_size": bs,
                                "epochs": ep,
                                "max_length": ml,
                                "eval_accuracy": metrics.get("eval_accuracy"),
                                "eval_f1_macro": metrics.get("eval_f1_macro"),
                                "eval_loss": metrics.get("eval_loss"),
                            }
                        )
                        continue

                    cmd = [
                        sys.executable,
                        "scripts/train_stage2_3class.py",
                        "--base_checkpoint",
                        args.base_checkpoint,
                        "--cornell_csv",
                        args.cornell_csv,
                        "--out_dir",
                        str(run_dir),
                        "--lr",
                        str(lr),
                        "--batch_size",
                        str(bs),
                        "--epochs",
                        str(ep),
                        "--max_length",
                        str(ml),
                        "--seed",
                        str(args.seed),
                        "--dataloader_num_workers",
                        str(args.dataloader_num_workers),
                        "--gradient_accumulation_steps",
                        str(args.gradient_accumulation_steps),
                        "--save_error_cases",
                        str(args.save_error_cases),
                        "--class_weighting",
                        args.class_weighting,
                        "--early_stopping_patience",
                        str(args.early_stopping_patience),
                        "--save_total_limit",
                        str(args.save_total_limit),
                    ]
                    if args.require_base_checkpoint:
                        cmd.append("--require_base_checkpoint")
                    if args.fp16:
                        cmd.append("--fp16")
                    if args.bf16:
                        cmd.append("--bf16")

                    print(f"\n=== [{run_idx}] {run_name} ===")
                    print(" ".join(cmd))
                    completed = subprocess.run(cmd, check=False)

                    row = {
                        "run_name": run_name,
                        "status": "ok" if completed.returncode == 0 else f"failed_{completed.returncode}",
                        "lr": lr,
                        "batch_size": bs,
                        "epochs": ep,
                        "max_length": ml,
                    }
                    if metrics_path.exists():
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        row.update(
                            {
                                "eval_accuracy": metrics.get("eval_accuracy"),
                                "eval_f1_macro": metrics.get("eval_f1_macro"),
                                "eval_loss": metrics.get("eval_loss"),
                            }
                        )
                    runs.append(row)

    summary_df = pd.DataFrame(runs)
    summary_csv = out_root / "sweep_summary.csv"
    summary_json = out_root / "sweep_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")

    if not summary_df.empty and "eval_f1_macro" in summary_df.columns:
        ranked = summary_df.sort_values("eval_f1_macro", ascending=False, na_position="last")
        print("\nTop runs by eval_f1_macro:")
        print(ranked[["run_name", "eval_f1_macro", "eval_accuracy", "eval_loss"]].head(10).to_string(index=False))
    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
