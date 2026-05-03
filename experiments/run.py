import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path

import yaml
from src.pipeline import run_pipeline


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_config(config_path):
    """Load experiment config, merging shared evaluation defaults."""
    config_abs = Path(config_path).resolve()
    with open(config_abs, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    current = config_abs.parent
    shared_eval = None
    while current != current.parent:
        candidate = current / "shared" / "evaluation.yaml"
        if candidate.exists():
            shared_eval = candidate
            break
        current = current.parent

    if shared_eval is not None:
        with open(shared_eval, encoding="utf-8") as handle:
            shared = yaml.safe_load(handle)
        config = _deep_merge(shared, config)

    return config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--output-dir", type=str, default="outputs/logs")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--no-save", action="store_true")

args = parser.parse_args()

config = _resolve_config(args.config)

metrics = run_pipeline(config)

print("Mode:", metrics.get("mode", "unknown"))
if metrics.get("dropped_labels"):
    print("Dropped labels (insufficient samples):", ", ".join(metrics.get("dropped_labels", [])))
if metrics.get("mode") == "cv":
    if metrics.get("effective_folds") is not None:
        print("CV folds:", metrics.get("effective_folds"), "(requested:", metrics.get("requested_folds"), ")")
    print(metrics.get("report", ""))
else:
    print("ACC:", metrics.get("accuracy"))
    print("Balanced ACC:", metrics.get("balanced_accuracy"))
    print("Macro F1:", metrics.get("macro_f1"))
    print(metrics.get("classification_report", ""))


def _to_scalar_row(config_path, tag_name, run_timestamp, metrics_dict):
    mode = metrics_dict.get("mode", "unknown")
    scalar_row = {
        "timestamp": run_timestamp,
        "tag": tag_name,
        "config": config_path,
        "exp": metrics_dict.get("exp", config.get("exp", "unknown")),
        "mode": mode,
        "accuracy": metrics_dict.get("accuracy"),
        "balanced_accuracy": metrics_dict.get("balanced_accuracy"),
        "macro_f1": metrics_dict.get("macro_f1"),
        "weighted_f1": metrics_dict.get("weighted_f1"),
        "macro_precision": metrics_dict.get("macro_precision"),
        "macro_recall": metrics_dict.get("macro_recall"),
        "accuracy_mean": metrics_dict.get("accuracy_mean"),
        "accuracy_std": metrics_dict.get("accuracy_std"),
        "accuracy_ci95_low": metrics_dict.get("accuracy_ci95_low"),
        "accuracy_ci95_high": metrics_dict.get("accuracy_ci95_high"),
        "balanced_accuracy_mean": metrics_dict.get("balanced_accuracy_mean"),
        "balanced_accuracy_std": metrics_dict.get("balanced_accuracy_std"),
        "balanced_accuracy_ci95_low": metrics_dict.get("balanced_accuracy_ci95_low"),
        "balanced_accuracy_ci95_high": metrics_dict.get("balanced_accuracy_ci95_high"),
        "macro_f1_mean": metrics_dict.get("macro_f1_mean"),
        "macro_f1_std": metrics_dict.get("macro_f1_std"),
        "macro_f1_ci95_low": metrics_dict.get("macro_f1_ci95_low"),
        "macro_f1_ci95_high": metrics_dict.get("macro_f1_ci95_high"),
        "weighted_f1_mean": metrics_dict.get("weighted_f1_mean"),
        "weighted_f1_std": metrics_dict.get("weighted_f1_std"),
        "weighted_f1_ci95_low": metrics_dict.get("weighted_f1_ci95_low"),
        "weighted_f1_ci95_high": metrics_dict.get("weighted_f1_ci95_high"),
        "macro_precision_mean": metrics_dict.get("macro_precision_mean"),
        "macro_precision_std": metrics_dict.get("macro_precision_std"),
        "macro_precision_ci95_low": metrics_dict.get("macro_precision_ci95_low"),
        "macro_precision_ci95_high": metrics_dict.get("macro_precision_ci95_high"),
        "macro_recall_mean": metrics_dict.get("macro_recall_mean"),
        "macro_recall_std": metrics_dict.get("macro_recall_std"),
        "macro_recall_ci95_low": metrics_dict.get("macro_recall_ci95_low"),
        "macro_recall_ci95_high": metrics_dict.get("macro_recall_ci95_high"),
        "fold_count": metrics_dict.get("fold_count"),
        "base_model_type": metrics_dict.get("base_model_type"),
        "meta_model_type": metrics_dict.get("meta_model_type"),
        "variant": metrics_dict.get("variant", ""),
    }
    return scalar_row


if not args.no_save:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_name = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = args.tag.strip() if args.tag.strip() else "default"

    json_path = output_dir / f"{config_name}_{run_tag}_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "tag": run_tag,
        "config_path": args.config,
        "metrics": metrics,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    summary_csv = output_dir / "summary.csv"
    summary_v2_csv = output_dir / "summary_v2.csv"
    row = _to_scalar_row(args.config, run_tag, timestamp, metrics)
    fieldnames = list(row.keys())

    target_csv = summary_csv
    if summary_csv.exists():
        with open(summary_csv, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])
        if existing_header and existing_header != fieldnames:
            target_csv = summary_v2_csv

    write_header = not target_csv.exists()
    with open(target_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("Saved JSON:", json_path)
    print("Updated CSV:", target_csv)
    if target_csv != summary_csv:
        print("Note: Detected legacy summary.csv schema; wrote to summary_v2.csv for compatibility.")