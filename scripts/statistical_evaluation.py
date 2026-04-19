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


def _method_name(payload):
    metrics = payload.get("metrics", {})
    exp = metrics.get("exp") or payload.get("config_path", "")
    base = metrics.get("base_model_type")
    meta = metrics.get("meta_model_type")
    if base and meta:
        return f"{exp}({base}->{meta})"
    return str(exp)


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison for CV experiment JSON files")
    parser.add_argument("--files", nargs="+", required=True, help="CV result JSON files")
    parser.add_argument("--metric", default="accuracy", help="Fold metric to compare")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--out-md", default="outputs/logs/statistical_evaluation.md")
    args = parser.parse_args()

    payloads = [_load_cv_file(Path(p)) for p in args.files]

    method_values = {}
    for payload in payloads:
        name = _method_name(payload)
        vals = _extract_fold_values(payload, args.metric)
        ci_low, ci_high = _bootstrap_ci_mean(vals, n_bootstrap=args.n_bootstrap)
        method_values[name] = {
            "values": vals,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "n": int(len(vals)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "file": payload.get("config_path", ""),
        }

    lines = [
        "# Statistical Evaluation",
        "",
        f"Metric: {args.metric}",
        "",
        "## Per-Method Summary",
        "",
        "| Method | N folds | Mean | Std | 95% Bootstrap CI | Source |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]

    for method, item in method_values.items():
        lines.append(
            f"| {method} | {item['n']} | {item['mean']:.4f} | {item['std']:.4f} | [{item['ci_low']:.4f}, {item['ci_high']:.4f}] | {item['file']} |"
        )

    lines.extend([
        "",
        "## Pairwise Permutation Tests",
        "",
        "| Method A | Method B | |Δ mean| | p-value |",
        "| --- | --- | ---: | ---: |",
    ])

    for a, b in combinations(method_values.keys(), 2):
        p_value, delta = _permutation_pvalue(
            method_values[a]["values"], method_values[b]["values"], n_perm=args.n_perm
        )
        lines.append(f"| {a} | {b} | {delta:.4f} | {p_value:.6f} |")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved statistical report: {out_path}")


if __name__ == "__main__":
    main()
