#!/usr/bin/env python3
"""
Generate LaTeX tables from statistical evaluation results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_cv_json(filepath: str) -> dict:
    """Load CV results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_method_info(cv_data: dict) -> Tuple[str, str]:
    """Extract method name and config from CV data."""
    metrics = cv_data.get('metrics', {})
    config_path = cv_data.get('config_path', '')
    exp_type = metrics.get('exp', 'unknown')
    
    # Build method name from config or exp type
    if config_path:
        method = config_path.split('/')[-1].replace('.yaml', '')
    else:
        method = exp_type
    
    return method, config_path

def compute_metrics_with_ci(fold_values: list) -> dict:
    """Compute mean, std, and 95% CI for fold values."""
    mean = np.mean(fold_values)
    std = np.std(fold_values, ddof=0)
    n = len(fold_values)
    se = std / np.sqrt(n)
    z_95 = 1.96
    ci_low = mean - z_95 * se
    ci_high = mean + z_95 * se
    
    return {
        'mean': mean,
        'std': std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n_folds': n
    }

def extract_all_metrics(cv_data: dict) -> dict:
    """Extract all metrics with CI from CV data."""
    metrics_dict = cv_data.get('metrics', {})
    results_by_metric = {}
    
    metrics_to_extract = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 
                          'macro_precision', 'macro_recall']
    
    for metric in metrics_to_extract:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        ci_low_key = f'{metric}_ci95_low'
        ci_high_key = f'{metric}_ci95_high'
        
        if mean_key in metrics_dict and std_key in metrics_dict:
            n_folds = len(metrics_dict.get("fold_metrics", [])) or metrics_dict.get("effective_folds") or metrics_dict.get("fold_count") or 1
            std = float(metrics_dict[std_key])
            se = std / np.sqrt(max(float(n_folds), 1.0))
            results_by_metric[metric] = {
                'mean': metrics_dict[mean_key],
                'std': std,
                'ci_low': metrics_dict.get(ci_low_key, metrics_dict[mean_key] - 1.96 * se),
                'ci_high': metrics_dict.get(ci_high_key, metrics_dict[mean_key] + 1.96 * se),
                'n_folds': int(n_folds),
            }
    
    return results_by_metric

def create_summary_dataframe(json_files: List[str]) -> pd.DataFrame:
    """Create a summary DataFrame from multiple CV JSON files."""
    rows = []
    
    for filepath in json_files:
        cv_data = load_cv_json(filepath)
        method_name, config = extract_method_info(cv_data)
        metrics = extract_all_metrics(cv_data)
        
        row = {
            'Method': method_name,
            'Config': config.split('/')[-1].replace('.yaml', ''),
            'N_Folds': metrics.get('accuracy', {}).get('n_folds', 0)
        }
        
        # Add all metrics
        for metric_name, metric_stats in metrics.items():
            row[f'{metric_name}_mean'] = metric_stats['mean']
            row[f'{metric_name}_std'] = metric_stats['std']
            row[f'{metric_name}_ci_low'] = metric_stats['ci_low']
            row[f'{metric_name}_ci_high'] = metric_stats['ci_high']
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def format_result_with_ci(mean: float, ci_low: float, ci_high: float) -> str:
    """Format result as mean (95% CI: [low, high])."""
    return f"{mean:.4f} ({ci_low:.4f}--{ci_high:.4f})"


def parse_pairwise_pvalues(stats_md_path: str) -> Dict[Tuple[str, str], float]:
    """Parse pairwise p-values from statistical evaluation markdown."""
    text = Path(stats_md_path).read_text(encoding="utf-8")
    out: Dict[Tuple[str, str], float] = {}
    in_pairwise = False
    for line in text.splitlines():
        if line.strip().startswith("## Pairwise Permutation Tests"):
            in_pairwise = True
            continue
        if not in_pairwise:
            continue
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().split("|")[1:-1]]
        if len(parts) != 4 or parts[0] in {"Method A", "---"}:
            continue
        a, b, _, pval = parts
        try:
            pv = float(pval)
        except ValueError:
            continue
        out[(a, b)] = pv
        out[(b, a)] = pv
    return out


def canonical_method_name(name: str) -> str:
    """Normalize method naming across configs and statistical reports."""
    n = str(name).lower()
    if n.startswith("exp_"):
        n = n[4:]
    if n.endswith("_cv"):
        n = n[:-3]
    return n

def generate_latex_table(
    df: pd.DataFrame,
    metrics: List[str] = None,
    pvalue_map: Dict[Tuple[str, str], float] = None,
    baseline_method: str = None,
) -> str:
    """Generate LaTeX table from summary dataframe."""
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'macro_f1']
    
    # Build table header
    include_pvalue = pvalue_map is not None and baseline_method is not None

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Method Comparison with 95\% Confidence Intervals (repeated 5-fold CV)\\",
        r"Results are reported as mean (95\% CI: [lower--upper]).}",
        r"\label{tab:method-comparison}",
        r"\small",
        r"\begin{tabular}{l" + "r" * (len(metrics) + (1 if include_pvalue else 0)) + "}",
        r"\toprule",
    ]
    
    # Header row
    header_cells = ["Method"]
    for metric in metrics:
        # Pretty print metric name
        pretty_name = metric.replace('_', '\\_').title()
        header_cells.append(pretty_name)
    if include_pvalue:
        baseline_for_header = baseline_method.replace('_', '\\_')
        header_cells.append(f"p-value vs {baseline_for_header}")
    latex_lines.append(" & ".join(header_cells) + r" \\")
    latex_lines.append(r"\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        cells = [row['Method'].replace('_', '\\_')]
        for metric in metrics:
            mean_key = f'{metric}_mean'
            ci_low_key = f'{metric}_ci_low'
            ci_high_key = f'{metric}_ci_high'
            
            if mean_key in row:
                formatted = format_result_with_ci(
                    row[mean_key], 
                    row[ci_low_key], 
                    row[ci_high_key]
                )
                cells.append(formatted)
        if include_pvalue:
            method = row['Method']
            if method == baseline_method:
                cells.append("--")
            else:
                pv = pvalue_map.get((method, baseline_method))
                if pv is None:
                    pv = pvalue_map.get((canonical_method_name(method), canonical_method_name(baseline_method)))
                cells.append(f"{pv:.6f}" if pv is not None else "N/A")
        
        latex_lines.append(" & ".join(cells) + r" \\")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(latex_lines)

def generate_markdown_table(
    df: pd.DataFrame,
    metrics: List[str] = None,
    pvalue_map: Dict[Tuple[str, str], float] = None,
    baseline_method: str = None,
) -> str:
    """Generate Markdown table from summary dataframe."""
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'macro_f1']
    
    # Header
    include_pvalue = pvalue_map is not None and baseline_method is not None
    header = ["Method"] + [m.replace('_', ' ').title() for m in metrics]
    if include_pvalue:
        header.append(f"p-value vs {baseline_method}")
    
    # Separator
    sep = ["-" * len(h) for h in header]
    
    # Data rows
    rows = [header, sep]
    for _, row in df.iterrows():
        cells = [row['Method']]
        for metric in metrics:
            mean_key = f'{metric}_mean'
            ci_low_key = f'{metric}_ci_low'
            ci_high_key = f'{metric}_ci_high'
            
            if mean_key in row:
                formatted = format_result_with_ci(
                    row[mean_key], 
                    row[ci_low_key], 
                    row[ci_high_key]
                )
                cells.append(formatted)
        if include_pvalue:
            method = row['Method']
            if method == baseline_method:
                cells.append("--")
            else:
                pv = pvalue_map.get((method, baseline_method))
                if pv is None:
                    pv = pvalue_map.get((canonical_method_name(method), canonical_method_name(baseline_method)))
                cells.append(f"{pv:.6f}" if pv is not None else "N/A")
        rows.append(cells)
    
    # Convert to markdown
    md_lines = []
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md_lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate LaTeX/Markdown tables from CV results')
    parser.add_argument('--files', nargs='+', required=True, help='CV JSON files')
    parser.add_argument('--metrics', nargs='+', default=['accuracy', 'balanced_accuracy', 'macro_f1'],
                       help='Metrics to include')
    parser.add_argument('--out-latex', help='Output LaTeX file')
    parser.add_argument('--out-md', help='Output Markdown file')
    parser.add_argument('--stats-md', help='Statistical evaluation markdown path for p-values')
    parser.add_argument('--baseline-method', help='Baseline method name used in p-value column')
    args = parser.parse_args()
    
    # Create summary
    df = create_summary_dataframe(args.files)
    print("\n=== Summary DataFrame ===")
    print(df.to_string())
    
    # Generate LaTeX if requested
    pvalue_map = None
    if args.stats_md:
        pvalue_map = parse_pairwise_pvalues(args.stats_md)

    if args.out_latex:
        latex_table = generate_latex_table(df, args.metrics, pvalue_map, args.baseline_method)
        Path(args.out_latex).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_latex, 'w') as f:
            f.write(latex_table)
        print(f"\n✓ LaTeX table saved: {args.out_latex}")
    
    # Generate Markdown if requested
    if args.out_md:
        md_table = generate_markdown_table(df, args.metrics, pvalue_map, args.baseline_method)
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, 'w') as f:
            f.write(md_table)
        print(f"✓ Markdown table saved: {args.out_md}")
    
    # Also output the raw CSV for reference
    csv_path = Path(args.out_latex or args.out_md).parent / "method_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Summary CSV saved: {csv_path}")

if __name__ == '__main__':
    main()
