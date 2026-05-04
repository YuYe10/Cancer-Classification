#!/usr/bin/env python3
"""Statistical comparison for CV experiment JSON files with multi-metric support."""

import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np


def _load_cv_file(path: Path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    if metrics.get("mode") != "cv":
        raise ValueError(f"Not a CV result file: {path}")
    return payload


def _extract_fold_values(payload, metric):
    fold_metrics = payload.get("metrics", {}).get("fold_metrics", [])
    values = [float(item.get(metric)) for item in fold_metrics if item.get(metric) is not None]
    if not values:
        raise ValueError(f"No fold values for metric '{metric}' in {payload.get('config_path')}")
    return np.array(values, dtype=float)


def _bootstrap_ci_mean(values: np.ndarray, n_bootstrap: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 42):
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = abs(pooled[:n_a].mean() - pooled[n_a:].mean())
        if d >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1)), float(observed)


def _paired_permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 42):
    """Paired permutation test for matched fold-level results."""
    rng = np.random.default_rng(seed)
    if len(a) != len(b):
        raise ValueError("Paired permutation requires equal length arrays")
    observed_diff = a - b
    observed_stat = abs(observed_diff.mean())
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(a))
        perm_diff = observed_diff * signs
        if abs(perm_diff.mean()) >= observed_stat:
            count += 1
    return float((count + 1) / (n_perm + 1)), float(observed_stat)


def _method_name(payload):
    metrics = payload.get("metrics", {})
    exp = metrics.get("exp") or payload.get("config_path", "")
    base = metrics.get("base_model_type")
    meta = metrics.get("meta_model_type")
    if base and meta:
        return f"{exp}({base}->{meta})"
    return str(exp)


def _compute_effect_size(a: np.ndarray, b: np.ndarray):
    """Compute Cohen's d effect size."""
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / pooled_std)


def _interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def generate_statistical_report(payloads, metrics_list, n_perm=20000, n_bootstrap=5000):
    """Generate comprehensive statistical report for multiple metrics.

    Args:
        payloads: list of loaded JSON payloads
        metrics_list: list of metric names to analyze
        n_perm: number of permutations
        n_bootstrap: number of bootstrap samples

    Returns:
        dict with report sections
    """
    method_values = {}

    for payload in payloads:
        name = _method_name(payload)
        method_values[name] = {}

        for metric in metrics_list:
            vals = _extract_fold_values(payload, metric)
            ci_low, ci_high = _bootstrap_ci_mean(vals, n_bootstrap=n_bootstrap)
            method_values[name][metric] = {
                "values": vals,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "n": int(len(vals)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "file": payload.get("config_path", ""),
            }

    report_sections = []

    report_sections.append("# Statistical Evaluation Report\n")

    for metric in metrics_list:
        metric_title = metric.replace("_", " ").title()
        report_sections.append(f"## {metric_title}\n")

        report_sections.append("### Summary Statistics\n")
        report_sections.append(f"| Method | N Folds | Mean | Std | 95% Bootstrap CI | Source |")
        report_sections.append(f"| --- | ---: | ---: | ---: | --- | --- |")

        for method, item in method_values.items():
            metric_data = item.get(metric, {})
            report_sections.append(
                f"| {method} | {metric_data.get('n', 'N/A')} | "
                f"{metric_data.get('mean', 0):.4f} | {metric_data.get('std', 0):.4f} | "
                f"[{metric_data.get('ci_low', 0):.4f}, {metric_data.get('ci_high', 0):.4f}] | "
                f"{metric_data.get('file', '')} |"
            )

        report_sections.append("\n### Pairwise Permutation Tests\n")
        report_sections.append(f"| Method A | Method B | |Δ mean| | p-value | Effect Size | Interpretation |")
        report_sections.append(f"| --- | --- | ---: | ---: | ---: | --- |")

        methods = list(method_values.keys())
        for a, b in combinations(methods, 2):
            a_vals = method_values[a][metric]["values"]
            b_vals = method_values[b][metric]["values"]
            p_value, delta = _permutation_pvalue(a_vals, b_vals, n_perm=n_perm)
            effect_size = _compute_effect_size(a_vals, b_vals)
            interpretation = _interpret_effect_size(effect_size)

            significance = "**" if p_value < 0.05 else ""
            report_sections.append(
                f"| {a} | {b} | {delta:.4f} | {p_value:.6f}{significance} | "
                f"{effect_size:.4f} | {interpretation} |"
            )

        report_sections.append("\n### Paired Permutation Tests (Matched Folds)\n")
        report_sections.append(f"| Method A | Method B | |Δ mean| | p-value |")
        report_sections.append(f"| --- | --- | ---: | ---: |")

        for a, b in combinations(methods, 2):
            a_vals = method_values[a][metric]["values"]
            b_vals = method_values[b][metric]["values"]

            if len(a_vals) == len(b_vals):
                try:
                    p_value, delta = _paired_permutation_pvalue(a_vals, b_vals, n_perm=n_perm)
                    significance = "**" if p_value < 0.05 else ""
                    report_sections.append(
                        f"| {a} | {b} | {delta:.4f} | {p_value:.6f}{significance} |"
                    )
                except Exception:
                    report_sections.append(f"| {a} | {b} | N/A | N/A |")
            else:
                report_sections.append(f"| {a} | {b} | N/A (unequal folds) | N/A |")

        report_sections.append("\n")

    return "\n".join(report_sections), method_values


def main():
    parser = argparse.ArgumentParser(
        description="Statistical comparison for CV experiment JSON files with multi-metric support"
    )
    parser.add_argument("--files", nargs="+", required=True, help="CV result JSON files")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "balanced_accuracy", "macro_f1"],
        help="Metrics to compare"
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--out-md", default="outputs/logs/statistical_evaluation.md")
    parser.add_argument("--out-json", default=None, help="Output JSON with full statistics")
    args = parser.parse_args()

    payloads = [_load_cv_file(Path(p)) for p in args.files]

    report, method_values = generate_statistical_report(
        payloads,
        args.metrics,
        n_perm=args.n_perm,
        n_bootstrap=args.n_bootstrap
    )

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"Saved statistical report: {out_path}")

    if args.out_json:
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(method_values, open(out_json_path, "w"), indent=2)
        print(f"Saved JSON statistics: {out_json_path}")


if __name__ == "__main__":
    main()