"""Generate publication-ready figures from real experiment logs.

This script reads outputs/logs/summary.csv and creates figures based on
actual experiment results (holdout + cv). It avoids synthetic/mock data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp_dt"] = pd.to_datetime(
            out["timestamp"].astype(str), format="%Y%m%d_%H%M%S", errors="coerce"
        )
    else:
        out["timestamp_dt"] = pd.NaT
    return out


def _latest_by(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.sort_values("timestamp_dt")
    return work.groupby(keys, as_index=False).tail(1).reset_index(drop=True)


def _best_score(row: pd.Series) -> float:
    if pd.notna(row.get("accuracy")):
        return float(row["accuracy"])
    if pd.notna(row.get("accuracy_mean")):
        return float(row["accuracy_mean"])
    return float("nan")


def plot_holdout_metrics(df: pd.DataFrame, out_dir: Path) -> Path | None:
    hold = df[df["mode"] == "holdout"].copy()
    hold = _latest_by(hold, ["exp", "mode"])
    if hold.empty:
        return None

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    metrics = [m for m in metrics if m in hold.columns]
    if not metrics:
        return None

    plot_df = hold[["exp"] + metrics].melt(
        id_vars="exp", var_name="metric", value_name="value"
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="exp")
    plt.ylim(0, 1.0)
    plt.title("Holdout Metrics by Experiment (Latest Run)")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    out_path = out_dir / "real_holdout_metrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_cv_errorbars(df: pd.DataFrame, out_dir: Path) -> Path | None:
    cv = df[df["mode"] == "cv"].copy()
    cv = _latest_by(cv, ["exp", "mode"])
    if cv.empty:
        return None

    metric_pairs = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("balanced_accuracy_mean", "balanced_accuracy_std", "Balanced Accuracy"),
        ("macro_f1_mean", "macro_f1_std", "Macro F1"),
    ]
    rows = []
    for _, r in cv.iterrows():
        for mean_col, std_col, label in metric_pairs:
            if mean_col in cv.columns and pd.notna(r.get(mean_col)):
                rows.append(
                    {
                        "exp": r["exp"],
                        "metric": label,
                        "mean": float(r[mean_col]),
                        "std": float(r.get(std_col, 0.0) or 0.0),
                    }
                )
    if not rows:
        return None

    plot_df = pd.DataFrame(rows)
    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="metric",
        y="mean",
        hue="exp",
        height=5,
        aspect=1.8,
        legend_out=False,
    )
    ax = g.ax
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-Validation Metrics (Mean ± Std)")
    ax.set_ylabel("Score")

    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.errorbar(x=x, y=y, yerr=row["std"], color="black", capsize=3, linewidth=1)

    out_path = out_dir / "real_cv_metrics_errorbar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close("all")
    return out_path


def plot_score_timeline(df: pd.DataFrame, out_dir: Path) -> Path | None:
    work = df.copy()
    work["score"] = work.apply(_best_score, axis=1)
    work = work[pd.notna(work["score"])].copy()
    if work.empty:
        return None

    work = work.sort_values("timestamp_dt")
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=work, x="timestamp_dt", y="score", hue="exp", style="mode", marker="o")
    plt.ylim(0, 1.0)
    plt.title("Experiment Score Timeline")
    plt.xlabel("Run Time")
    plt.ylabel("Accuracy / Accuracy Mean")
    plt.tight_layout()
    out_path = out_dir / "real_score_timeline.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_target_gap(df: pd.DataFrame, out_dir: Path, threshold: float) -> Path | None:
    work = df.copy()
    work["score"] = work.apply(_best_score, axis=1)
    work = work[pd.notna(work["score"])].copy()
    if work.empty:
        return None

    best = work.groupby("exp", as_index=False)["score"].max()
    best["gap_to_target"] = threshold - best["score"]

    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=best, x="exp", y="gap_to_target")
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Gap to Target Accuracy ({threshold:.2f})")
    plt.xlabel("Experiment")
    plt.ylabel("Target - Best Score")
    plt.tight_layout()
    out_path = out_dir / "real_target_gap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real experiment results from summary.csv")
    parser.add_argument(
        "--summary-csv",
        default="outputs/logs/summary.csv",
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Target threshold used in gap plot",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    df = _coerce_timestamp(df)

    generated: List[Path] = []
    for fn in [
        lambda: plot_holdout_metrics(df, out_dir),
        lambda: plot_cv_errorbars(df, out_dir),
        lambda: plot_score_timeline(df, out_dir),
        lambda: plot_target_gap(df, out_dir, args.threshold),
    ]:
        out = fn()
        if out is not None:
            generated.append(out)

    snapshot = out_dir / "real_results_latest_snapshot.csv"
    latest = _latest_by(df, ["exp", "mode"])
    latest.to_csv(snapshot, index=False)
    generated.append(snapshot)

    print("Generated artifacts:")
    for p in generated:
        print(f"- {p}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
