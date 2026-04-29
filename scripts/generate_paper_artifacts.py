#!/usr/bin/env python3
"""Generate paper-grade figures and archive legacy visualizations.

This script rebuilds the figures that belong in the main report from the
actual experiment logs, then optionally archives legacy mock/demo images so
they cannot be confused with the paper figures.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"
STABILITY_DIR = LOG_DIR / "stability"

METHOD_ORDER = ["rna", "concat", "mofa", "stacking"]
METHOD_LABELS = {
    "rna": "RNA-only",
    "concat": "Concat",
    "mofa": "MOFA",
    "stacking": "Stacking",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "macro_f1": "Macro-F1",
}

KEEP_FIGURES = {
    "statistical_accuracy_ci.png",
    "statistical_accuracy_distribution.png",
    "statistical_balanced_accuracy_ci.png",
    "statistical_balanced_accuracy_distribution.png",
    "statistical_macro_f1_ci.png",
    "statistical_macro_f1_distribution.png",
    "stability_feature_dim_accuracy.png",
    "stability_repeat_convergence.png",
    "shap_summary.png",
}


@dataclass(frozen=True)
class FigureArtifact:
    name: str
    caption: str
    path: Path


def _configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _load_summary() -> pd.DataFrame:
    summary_path = LOG_DIR / "summary_v2.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    df = pd.read_csv(summary_path)
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(
            df["timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce"
        )
    else:
        df["timestamp_dt"] = pd.NaT
    return df


def _load_stability() -> pd.DataFrame:
    stability_path = STABILITY_DIR / "stability_sweeps_summary.csv"
    if not stability_path.exists():
        raise FileNotFoundError(f"Missing stability file: {stability_path}")
    return pd.read_csv(stability_path)


def _latest_mainline_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["mode"] == "cv"].copy()
    work = work[work["exp"].isin(METHOD_ORDER)].copy()
    if work.empty:
        raise ValueError("No mainline CV rows found in summary_v2.csv")
    work = work.sort_values("timestamp_dt")
    latest = work.groupby("exp", as_index=False).tail(1).reset_index(drop=True)
    latest["exp"] = pd.Categorical(latest["exp"], categories=METHOD_ORDER, ordered=True)
    return latest.sort_values("exp")


def _metric_ci(row: pd.Series, metric: str) -> tuple[float, float, float]:
    mean_key = f"{metric}_mean"
    low_key = f"{metric}_ci95_low"
    high_key = f"{metric}_ci95_high"
    std_key = f"{metric}_std"

    mean = float(row[mean_key])
    if pd.notna(row.get(low_key)) and pd.notna(row.get(high_key)):
        return mean, float(row[low_key]), float(row[high_key])

    std = float(row.get(std_key, 0.0) or 0.0)
    n = int(row.get("fold_count", 1) or 1)
    se = std / np.sqrt(max(n, 1))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _plot_metric_ci(df: pd.DataFrame, metric: str, out_name: str, title: str) -> FigureArtifact:
    rows = []
    for _, row in df.iterrows():
        mean, low, high = _metric_ci(row, metric)
        rows.append(
            {
                "exp": METHOD_LABELS[row["exp"]],
                "mean": mean,
                "low": low,
                "high": high,
            }
        )

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    colors = ["#2f6fb0", "#f28e2b", "#59a14f", "#e15759"]
    x = np.arange(len(plot_df))
    yerr = np.vstack([plot_df["mean"] - plot_df["low"], plot_df["high"] - plot_df["mean"]])
    ax.bar(
        x,
        plot_df["mean"],
        yerr=yerr,
        capsize=7,
        color=colors[: len(plot_df)],
        edgecolor="black",
        linewidth=1.0,
        alpha=0.88,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["exp"], rotation=0)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlabel("Method")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    for idx, row in plot_df.iterrows():
        ax.text(idx, min(row["mean"] + 0.035, 0.985), f"{row['mean']:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.text(idx, max(row["low"] - 0.055, 0.02), f"[{row['low']:.3f}, {row['high']:.3f}]", ha="center", va="top", fontsize=9)

    fig.tight_layout()
    out_path = FIGURE_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return FigureArtifact(out_name, title, out_path)


def _fold_values_from_json(json_path: Path, metric: str) -> list[float]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = payload.get("metrics", {})
    fold_metrics = metrics.get("fold_metrics", [])
    values: list[float] = []
    for item in fold_metrics:
        if item.get(metric) is not None:
            values.append(float(item[metric]))
    return values


def _latest_json_for_method(method: str) -> Path | None:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in LOG_DIR.glob("*.json"):
        if "stability" in path.parts:
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        metrics = payload.get("metrics", {})
        if metrics.get("exp") != method or metrics.get("mode") != "cv":
            continue
        ts = payload.get("timestamp") or metrics.get("timestamp")
        ts_dt = pd.to_datetime(str(ts), format="%Y%m%d_%H%M%S", errors="coerce")
        candidates.append((ts_dt, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _plot_metric_distribution(metric: str, out_name: str, title: str) -> FigureArtifact:
    series = []
    labels = []
    for method in METHOD_ORDER:
        json_path = _latest_json_for_method(method)
        if json_path is None:
            continue
        values = _fold_values_from_json(json_path, metric)
        if not values:
            continue
        series.append(values)
        labels.append(METHOD_LABELS[method])

    if not series:
        raise ValueError(f"No fold values found for metric '{metric}'")

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bp = ax.boxplot(series, tick_labels=labels, patch_artist=True, widths=0.58)
    palette = ["#2f6fb0", "#f28e2b", "#59a14f", "#e15759"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlabel("Method")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path = FIGURE_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return FigureArtifact(out_name, title, out_path)


def _plot_stability(stability_df: pd.DataFrame) -> list[FigureArtifact]:
    artifacts: list[FigureArtifact] = []

    feature_dim = stability_df[stability_df["sweep_type"] == "feature_dim"].copy()
    if not feature_dim.empty:
        feature_dim = feature_dim.sort_values("param_value")
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.errorbar(
            feature_dim["param_value"],
            feature_dim["accuracy_mean"],
            yerr=[
                feature_dim["accuracy_mean"] - feature_dim["accuracy_ci95_low"],
                feature_dim["accuracy_ci95_high"] - feature_dim["accuracy_mean"],
            ],
            marker="o",
            linewidth=2.2,
            capsize=6,
            color="#f28e2b",
            label="Accuracy",
        )
        ax.set_xlabel("Top variable features per modality")
        ax.set_ylabel("Accuracy")
        ax.set_title("Concat feature-dimension sensitivity")
        ax.set_ylim(0.7, 1.0)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        out_path = FIGURE_DIR / "stability_feature_dim_accuracy.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(FigureArtifact("stability_feature_dim_accuracy.png", "Concat feature-dimension sensitivity", out_path))

    repeats = stability_df[stability_df["sweep_type"] == "repeat_convergence"].copy()
    if not repeats.empty:
        repeats = repeats.sort_values("param_value")
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.errorbar(
            repeats["param_value"],
            repeats["accuracy_mean"],
            yerr=[
                repeats["accuracy_mean"] - repeats["accuracy_ci95_low"],
                repeats["accuracy_ci95_high"] - repeats["accuracy_mean"],
            ],
            marker="o",
            linewidth=2.2,
            capsize=6,
            color="#4e79a7",
            label="Accuracy",
        )
        if "accuracy_ci95_high" in repeats.columns:
            ci_width = repeats["accuracy_ci95_high"] - repeats["accuracy_ci95_low"]
            ax2 = ax.twinx()
            ax2.plot(repeats["param_value"], ci_width, marker="s", linestyle="--", color="#f28e2b", label="95% CI width")
            ax2.set_ylabel("95% CI width")
            ax2.set_ylim(0, max(float(ci_width.max()) * 1.15, 0.05))
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(handles1 + handles2, labels1 + labels2, loc="best")
        else:
            ax.legend()
        ax.set_xlabel("Number of repeats")
        ax.set_ylabel("Accuracy")
        ax.set_title("Concat repeated-CV convergence")
        ax.set_ylim(0.7, 1.0)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out_path = FIGURE_DIR / "stability_repeat_convergence.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(FigureArtifact("stability_repeat_convergence.png", "Concat repeated-CV convergence", out_path))

    return artifacts


def _write_manifest(artifacts: Iterable[FigureArtifact], archived: list[tuple[Path, Path, str]]) -> None:
    lines = ["# Paper Figure Manifest", "", "| File | Caption |", "| --- | --- |"]
    for art in artifacts:
        lines.append(f"| {art.name} | {art.caption} |")
    manifest_path = FIGURE_DIR / "paper_figure_manifest.md"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    archive_lines = ["source,target,reason"]
    for src, dst, reason in archived:
        archive_lines.append(f"{src.name},{dst.name},{reason}")
    archive_manifest = FIGURE_DIR / "archive_manifest.csv"
    archive_manifest.write_text("\n".join(archive_lines) + "\n", encoding="utf-8")


def _archive_legacy_figures(dry_run: bool) -> list[tuple[Path, Path, str]]:
    archive_dir = FIGURE_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: list[tuple[Path, Path, str]] = []

    for path in FIGURE_DIR.glob("*"):
        if path.is_dir() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            continue
        if path.name in KEEP_FIGURES:
            continue
        dst = archive_dir / f"legacy_{path.name}"
        archived.append((path, dst, "not in paper manifest"))
        if not dry_run:
            shutil.move(str(path), str(dst))

    return archived


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-grade figures and archive legacy assets")
    parser.add_argument("--archive-legacy", action="store_true", help="Move non-paper figures into outputs/figures/archive")
    parser.add_argument("--dry-run", action="store_true", help="Show which figures would be archived without moving them")
    args = parser.parse_args()

    _configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = _load_summary()
    mainline = _latest_mainline_rows(summary_df)

    artifacts: list[FigureArtifact] = []
    for metric, label in METRIC_LABELS.items():
        artifacts.append(
            _plot_metric_ci(
                mainline,
                metric,
                f"statistical_{metric}_ci.png",
                f"{label} with 95% Confidence Intervals (repeated CV, 5×3)",
            )
        )
        artifacts.append(
            _plot_metric_distribution(
                metric,
                f"statistical_{metric}_distribution.png",
                f"Distribution of {label} Across Folds",
            )
        )

    stability_df = _load_stability()
    artifacts.extend(_plot_stability(stability_df))

    archived: list[tuple[Path, Path, str]] = []
    if args.archive_legacy:
        archived = _archive_legacy_figures(dry_run=args.dry_run)

    _write_manifest(artifacts, archived)

    print("Generated paper figures:")
    for artifact in artifacts:
        print(f"- {artifact.path}")

    if args.archive_legacy:
        if args.dry_run:
            print("\nArchive preview:")
            for src, dst, reason in archived:
                print(f"- {src.name} -> {dst} ({reason})")
        else:
            print(f"\nArchived {len(archived)} legacy figures into {FIGURE_DIR / 'archive'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())