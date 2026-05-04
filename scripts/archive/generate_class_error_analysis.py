#!/usr/bin/env python3
"""Generate class-level error analysis tables from CV result JSON files."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_MAP = {
    "0": "LumA",
    "1": "LumB",
    "2": "HER2",
    "3": "Basal",
}


def load_cv_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    if metrics.get("mode") != "cv":
        raise ValueError(f"Not a CV result file: {path}")
    return data


def parse_classification_report(report_text: str) -> dict:
    """Parse sklearn text classification report into per-class metrics dict."""
    class_metrics = {}
    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Expected rows: label precision recall f1 support
        m = re.match(
            r"^(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]+)$",
            line,
        )
        if not m:
            continue
        label, precision, recall, f1, support = m.groups()
        class_metrics[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(support),
        }
    return class_metrics


def sort_label_key(label: str) -> int:
    return int(label)


def aggregate_one_method(path: Path) -> dict:
    payload = load_cv_json(path)
    metrics = payload.get("metrics", {})
    exp = metrics.get("exp", path.stem)

    fold_metrics = metrics.get("fold_metrics", [])
    collector = {}
    for fold in fold_metrics:
        report_text = fold.get("classification_report")
        if not isinstance(report_text, str):
            continue
        parsed = parse_classification_report(report_text)
        for label, vals in parsed.items():
            collector.setdefault(label, {"precision": [], "recall": [], "f1": [], "support": []})
            for key in ("precision", "recall", "f1", "support"):
                collector[label][key].append(vals[key])

    row = {
        "method": exp,
        "accuracy_mean": metrics.get("accuracy_mean"),
        "balanced_accuracy_mean": metrics.get("balanced_accuracy_mean"),
        "macro_f1_mean": metrics.get("macro_f1_mean"),
    }

    # Keep stable class order for report readability.
    for label in sorted(collector.keys(), key=sort_label_key):
        pretty = LABEL_MAP.get(label, f"Class{label}")
        row[f"{pretty}_precision"] = float(np.mean(collector[label]["precision"]))
        row[f"{pretty}_recall"] = float(np.mean(collector[label]["recall"]))
        row[f"{pretty}_f1"] = float(np.mean(collector[label]["f1"]))
        row[f"{pretty}_support_mean"] = float(np.mean(collector[label]["support"]))

    return row


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Class-level Error Analysis",
        "",
        "Main comparison based on fold-level mean metrics from CV JSON logs.",
        "",
        "| Method | Accuracy | Balanced Acc | Macro-F1 | LumA Recall | LumB Recall | Basal Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| {method} | {acc:.4f} | {bal:.4f} | {mf1:.4f} | {lumA:.4f} | {lumB:.4f} | {basal:.4f} |".format(
                method=r["method"],
                acc=r["accuracy_mean"],
                bal=r["balanced_accuracy_mean"],
                mf1=r["macro_f1_mean"],
                lumA=r.get("LumA_recall", float("nan")),
                lumB=r.get("LumB_recall", float("nan")),
                basal=r.get("Basal_recall", float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- HER2 may be absent when the aligned sample count is below CV minimum and gets dropped before training.",
            "- Values are fold-level averages, not single split scores.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate class-level error tables from CV logs")
    parser.add_argument("--files", nargs="+", required=True, help="CV JSON result files")
    parser.add_argument("--out-csv", default="outputs/logs/class_error_analysis.csv")
    parser.add_argument("--out-md", default="outputs/logs/class_error_analysis.md")
    args = parser.parse_args()

    rows = [aggregate_one_method(Path(p)) for p in args.files]
    df = pd.DataFrame(rows)

    # Stable display order in paper.
    preferred = ["rna", "concat", "mofa", "stacking"]
    if "method" in df.columns:
        df["_order"] = df["method"].apply(lambda x: preferred.index(x) if x in preferred else 99)
        df = df.sort_values(["_order", "method"]).drop(columns=["_order"])

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    out_md.write_text(build_markdown(df), encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved markdown: {out_md}")


if __name__ == "__main__":
    main()
