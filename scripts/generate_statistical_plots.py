#!/usr/bin/env python3
"""Generate publication-ready plots from CV result JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_cv_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_method_name(cv_data, file_path):
    metrics = cv_data.get("metrics", {})
    config_path = cv_data.get("config_path", "")

    if config_path:
        return Path(config_path).stem
    if metrics.get("exp"):
        return str(metrics["exp"])

    payload = cv_data.get("payload", {})
    if payload.get("exp_type"):
        return str(payload["exp_type"])

    return Path(file_path).stem


def _extract_fold_values(cv_data, metric):
    metrics = cv_data.get("metrics", {})

    # New schema: metrics.fold_metrics
    fold_metrics = metrics.get("fold_metrics", [])
    values = []
    for item in fold_metrics:
        if item.get(metric) is not None:
            values.append(float(item[metric]))
    if values:
        return values

    # Legacy schema: payload/results/folds
    folds = cv_data.get("results", {}).get("folds", [])
    values = []
    for fold in folds:
        if fold.get(metric) is not None:
            values.append(float(fold[metric]))
    return values


def _extract_mean_ci_from_metrics(cv_data, metric):
    metrics = cv_data.get("metrics", {})
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    ci_low_key = f"{metric}_ci95_low"
    ci_high_key = f"{metric}_ci95_high"

    if metrics.get(mean_key) is None:
        return None

    mean = float(metrics[mean_key])
    if metrics.get(ci_low_key) is not None and metrics.get(ci_high_key) is not None:
        return mean, float(metrics[ci_low_key]), float(metrics[ci_high_key])

    std = float(metrics.get(std_key, 0.0) or 0.0)
    n = len(_extract_fold_values(cv_data, metric)) or int(metrics.get("fold_count", 1) or 1)
    se = std / np.sqrt(max(n, 1))
    return mean, mean - 1.96 * se, mean + 1.96 * se

def plot_ci_comparison(json_files, metric="accuracy", output_path="ci_comparison.png"):
    """Plot mean +/- 95% CI for multiple methods."""
    methods = []
    means = []
    cis_low = []
    cis_high = []

    for filepath in json_files:
        cv_data = load_cv_json(filepath)
        method_name = _extract_method_name(cv_data, filepath)

        # Prefer metrics-level CI if available.
        stats = _extract_mean_ci_from_metrics(cv_data, metric)
        if stats is None:
            fold_values = _extract_fold_values(cv_data, metric)
            if not fold_values:
                continue
            mean = float(np.mean(fold_values))
            std = float(np.std(fold_values, ddof=0))
            se = std / np.sqrt(max(len(fold_values), 1))
            ci_low = mean - 1.96 * se
            ci_high = mean + 1.96 * se
        else:
            mean, ci_low, ci_high = stats

        ci_low = max(0.0, ci_low)
        ci_high = min(1.0, ci_high)
        methods.append(method_name)
        means.append(mean)
        cis_low.append(ci_low)
        cis_high.append(ci_high)

    if not methods:
        raise ValueError(f"No available metric '{metric}' in provided files")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(methods))
    errors = [np.array(means) - np.array(cis_low), np.array(cis_high) - np.array(means)]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    ax.bar(x_pos, means, yerr=errors, capsize=10, alpha=0.8, color=colors[: len(methods)])

    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{metric.replace('_', ' ').title()} with 95% Confidence Intervals",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylim(0, 1)

    for i, (mean, ci_low, ci_high) in enumerate(zip(means, cis_low, cis_high)):
        ax.text(i, min(mean + 0.03, 0.98), f"{mean:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.text(i, max(ci_low - 0.06, 0.01), f"[{ci_low:.3f}, {ci_high:.3f}]", ha="center", va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    return fig, ax

def plot_fold_distribution(json_files, metric="accuracy", output_path="fold_distribution.png"):
    """Plot distribution of fold scores."""
    fig, ax = plt.subplots(figsize=(12, 6))

    all_fold_values = []
    labels = []

    for filepath in json_files:
        cv_data = load_cv_json(filepath)
        method_name = _extract_method_name(cv_data, filepath)
        fold_values = _extract_fold_values(cv_data, metric)
        if not fold_values:
            continue

        all_fold_values.append(fold_values)
        labels.append(method_name)

    if not all_fold_values:
        raise ValueError(f"No fold values for metric '{metric}' in provided files")

    bp = ax.boxplot(all_fold_values, widths=0.6, patch_artist=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Distribution of {metric.replace('_', ' ').title()} Across Folds",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description="Generate statistical plots from CV JSON files")
    parser.add_argument("--files", nargs="+", required=True, help="Path to CV JSON files")
    parser.add_argument("--metric", default="accuracy", help="Metric to plot")
    parser.add_argument("--out-dir", default="outputs/figures", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ci_plot = output_dir / f"statistical_{args.metric}_ci.png"
    plot_ci_comparison(args.files, args.metric, str(ci_plot))

    dist_plot = output_dir / f"statistical_{args.metric}_distribution.png"
    plot_fold_distribution(args.files, args.metric, str(dist_plot))

    print(f"\n✓ Statistical plots generated in {output_dir}/")


if __name__ == "__main__":
    main()
