#!/usr/bin/env python3
"""
Academic Figure Generator - Example Usage Script
==============================================

This script demonstrates the capabilities of the Academic Figure Generator
package for generating publication-quality figures for research papers.

Author: Cancer-Classification Project
License: MIT

Usage:
    python generate_example_figures.py

Output:
    - Example figures saved to outputs/figures/
    - Example data saved to outputs/data/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from academic_figure_generator import (
    DataReader,
    ChartGenerator,
    OutputManager,
    DocumentIntegrator,
    ChartStyle,
    OutputConfig
)


def generate_sample_data():
    """
    Generate sample experimental data for demonstration.

    Returns:
        Dictionary containing sample DataFrames.
    """
    np.random.seed(42)

    methods = ['RNA-only', 'Concat', 'MOFA', 'Stacking']
    metrics = ['Accuracy', 'Balanced Accuracy', 'Macro-F1']

    performance_data = {
        'Method': [],
        'Accuracy': [],
        'Balanced Accuracy': [],
        'Macro-F1': [],
        'Std_Accuracy': [],
        'Std_Balanced': [],
        'Std_F1': []
    }

    base_means = {
        'RNA-only': [0.8655, 0.8450, 0.8380],
        'Concat': [0.8929, 0.8750, 0.8670],
        'MOFA': [0.7857, 0.7620, 0.7510],
        'Stacking': [0.8488, 0.8300, 0.8220]
    }

    stds = {
        'RNA-only': [0.045, 0.048, 0.050],
        'Concat': [0.038, 0.040, 0.042],
        'MOFA': [0.052, 0.055, 0.058],
        'Stacking': [0.044, 0.046, 0.048]
    }

    for method in methods:
        performance_data['Method'].append(method)
        performance_data['Accuracy'].append(base_means[method][0])
        performance_data['Balanced Accuracy'].append(base_means[method][1])
        performance_data['Macro-F1'].append(base_means[method][2])
        performance_data['Std_Accuracy'].append(stds[method][0])
        performance_data['Std_Balanced'].append(stds[method][1])
        performance_data['Std_F1'].append(stds[method][2])

    performance_df = pd.DataFrame(performance_data)

    repeats = list(range(1, 11))
    fold_data = {
        'Repeat': [],
        'Fold': [],
        'Method': [],
        'Accuracy': [],
        'Balanced Accuracy': [],
        'Macro-F1': []
    }

    for repeat in repeats:
        for fold in range(1, 6):
            for method in methods:
                fold_data['Repeat'].append(repeat)
                fold_data['Fold'].append(fold)
                fold_data['Method'].append(method)

                base_idx = methods.index(method)
                base_acc = base_means[method][0]
                std_acc = stds[method][0]

                acc = base_acc + np.random.normal(0, std_acc / 2)
                bal_acc = base_acc - 0.02 + np.random.normal(0, std_acc / 2)
                f1 = base_acc - 0.03 + np.random.normal(0, std_acc / 2)

                fold_data['Accuracy'].append(max(0, min(1, acc)))
                fold_data['Balanced Accuracy'].append(max(0, min(1, bal_acc)))
                fold_data['Macro-F1'].append(max(0, min(1, f1)))

    fold_df = pd.DataFrame(fold_data)

    confusion_data = {
        'Predicted': [],
        'Actual': [],
        'Count': []
    }

    actual_classes = ['Luminal A', 'Luminal B', 'Basal-like']
    predicted_classes = ['Luminal A', 'Luminal B', 'Basal-like']

    confusion_counts = np.array([
        [85, 12, 3],
        [15, 68, 17],
        [5, 18, 77]
    ])

    for i, actual in enumerate(actual_classes):
        for j, predicted in enumerate(predicted_classes):
            confusion_data['Predicted'].append(predicted)
            confusion_data['Actual'].append(actual)
            confusion_data['Count'].append(confusion_counts[i, j])

    confusion_df = pd.DataFrame(confusion_data)

    correlation_data = {
        'RNA Expression': [],
        'Methylation': [],
        'Sample': []
    }

    for i in range(100):
        rna = np.random.normal(5, 1)
        meth = 0.7 * rna + np.random.normal(0, 0.5)
        sample = f'S{i+1}'
        correlation_data['RNA Expression'].append(rna)
        correlation_data['Methylation'].append(meth)
        correlation_data['Sample'].append(sample)

    correlation_df = pd.DataFrame(correlation_data)

    return {
        'performance': performance_df,
        'fold_results': fold_df,
        'confusion': confusion_df,
        'correlation': correlation_df
    }


def example_basic_charts():
    """Generate basic chart examples."""
    print("\n" + "="*60)
    print("Example 1: Basic Charts (Line, Bar, Scatter)")
    print("="*60)

    style = ChartStyle(
        figsize=(8, 6),
        dpi=300,
        font_family='serif',
        font_size=12,
        color_scheme='tab10'
    )

    generator = ChartGenerator(style=style)
    output_manager = OutputManager()

    sample_data = generate_sample_data()
    performance_df = sample_data['performance']

    bar_fig = generator.bar_plot(
        x=[0, 1, 2, 3],
        y=[performance_df['Accuracy'].values],
        labels=['Methods'],
        categories=performance_df['Method'].tolist(),
        title='Model Performance Comparison (Accuracy)',
        xlabel='Method',
        ylabel='Accuracy',
        errors=performance_df['Std_Accuracy'].values,
        show_grid=True,
        bar_labels=True
    )

    output_manager.save_figure(
        bar_fig, 'example_bar_performance',
        fig_type='bar', formats=['pdf', 'svg']
    )
    print("Saved: example_bar_performance.pdf/svg")
    generator.close()

    generator = ChartGenerator(style=style)
    fold_df = sample_data['fold_results']

    repeat_means = fold_df.groupby('Repeat')['Accuracy'].mean().values

    line_fig = generator.line_plot(
        x=list(range(1, 11)),
        y=[repeat_means],
        labels=['Mean Accuracy'],
        title='Accuracy Trend Across Repeated Cross-Validation',
        xlabel='Repeat Number',
        ylabel='Mean Accuracy',
        show_grid=True
    )

    output_manager.save_figure(
        line_fig, 'example_line_trend',
        fig_type='line', formats=['pdf', 'svg']
    )
    print("Saved: example_line_trend.pdf/svg")
    generator.close()

    generator = ChartGenerator(style=style)
    corr_df = sample_data['correlation']

    scatter_fig = generator.scatter_plot(
        x=corr_df['RNA Expression'].values,
        y=corr_df['Methylation'].values,
        title='RNA Expression vs Methylation Correlation',
        xlabel='RNA Expression (log2)',
        ylabel='Methylation Beta Value',
        show_regression=True,
        regression_line=True
    )

    output_manager.save_figure(
        scatter_fig, 'example_scatter_correlation',
        fig_type='scatter', formats=['pdf', 'svg']
    )
    print("Saved: example_scatter_correlation.pdf/svg")
    generator.close()


def example_heatmap():
    """Generate heatmap example."""
    print("\n" + "="*60)
    print("Example 2: Heatmap Visualization")
    print("="*60)

    generator = ChartGenerator()
    output_manager = OutputManager()

    sample_data = generate_sample_data()
    confusion_df = sample_data['confusion']

    confusion_matrix = confusion_df.pivot_table(
        index='Actual',
        columns='Predicted',
        values='Count',
        aggfunc='sum'
    )

    heatmap_fig = generator.heatmap(
        data=confusion_matrix,
        title='Confusion Matrix',
        xlabel='Predicted Label',
        ylabel='Actual Label',
        cmap='Blues',
        annot=True,
        fmt='d',
        linewidths=1,
        colorbar_label='Sample Count'
    )

    output_manager.save_figure(
        heatmap_fig, 'example_heatmap_confusion',
        fig_type='heatmap', formats=['pdf', 'svg']
    )
    print("Saved: example_heatmap_confusion.pdf/svg")
    generator.close()


def example_multi_panel():
    """Generate multi-panel figure example."""
    print("\n" + "="*60)
    print("Example 3: Multi-Panel Comparison")
    print("="*60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    style = ChartStyle(figsize=(5, 4), dpi=300)
    generator = ChartGenerator(style=style)

    sample_data = generate_sample_data()
    performance_df = sample_data['performance']

    metrics = ['Accuracy', 'Balanced Accuracy', 'Macro-F1']
    y_positions = [0, 1, 2, 3]

    for idx, metric in enumerate(metrics):
        col = metric.replace(' ', '_').lower()
        values = performance_df[metric].values
        errors = performance_df[f'Std_{performance_df.columns[performance_df.columns.str.contains(col)].tolist()[0].split("_")[0] if "_" in col else metric.split()[0][:-4] if len(metric.split()) > 1 else metric[:4]}'].values

        if idx == 0:
            errors = performance_df['Std_Accuracy'].values
        elif idx == 1:
            errors = performance_df['Std_Balanced'].values
        else:
            errors = performance_df['Std_F1'].values

        axes[0, 0].barh(y_positions, values, xerr=errors, alpha=0.7,
                       label=metric, color=generator._get_colors(1)[0])

    axes[0, 0].set_yticks(y_positions)
    axes[0, 0].set_yticklabels(performance_df['Method'])
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    fold_df = sample_data['fold_results']
    methods = ['RNA-only', 'Concat', 'MOFA', 'Stacking']
    colors = generator._get_colors(4)

    for i, method in enumerate(methods):
        method_data = fold_df[fold_df['Method'] == method]
        repeat_means = method_data.groupby('Repeat')['Accuracy'].mean().values
        axes[0, 1].plot(range(1, 11), repeat_means, marker='o',
                       color=colors[i], label=method, linewidth=2)

    axes[0, 1].set_xlabel('Repeat Number')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Trends by Method')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    corr_df = sample_data['correlation']
    scatter = axes[1, 0].scatter(
        corr_df['RNA Expression'],
        corr_df['Methylation'],
        alpha=0.6, c='steelblue', edgecolors='black', linewidths=0.5
    )
    axes[1, 0].set_xlabel('RNA Expression')
    axes[1, 0].set_ylabel('Methylation')
    axes[1, 0].set_title('Correlation Analysis')
    axes[1, 0].grid(True, alpha=0.3)

    confusion_df = sample_data['confusion']
    confusion_matrix = confusion_df.pivot_table(
        index='Actual', columns='Predicted', values='Count', aggfunc='sum'
    )
    im = axes[1, 1].imshow(confusion_matrix.values, cmap='Blues', aspect='auto')
    axes[1, 1].set_xticks(range(len(confusion_matrix.columns)))
    axes[1, 1].set_yticks(range(len(confusion_matrix.index)))
    axes[1, 1].set_xticklabels(confusion_matrix.columns, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(confusion_matrix.index)
    axes[1, 1].set_title('Confusion Matrix')
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

    for i in range(len(confusion_matrix.index)):
        for j in range(len(confusion_matrix.columns)):
            axes[1, 1].text(j, i, confusion_matrix.values[i, j],
                           ha='center', va='center', fontsize=12,
                           color='white' if confusion_matrix.values[i, j] > 50 else 'black')

    fig.suptitle('Comprehensive Analysis of Multi-Omics Classification Results',
                fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    output_manager = OutputManager()
    output_manager.save_figure(fig, 'example_multipanel_comprehensive',
                              fig_type='other', formats=['pdf', 'svg'])
    print("Saved: example_multipanel_comprehensive.pdf/svg")

    plt.close(fig)


def example_document_integration():
    """Demonstrate document integration."""
    print("\n" + "="*60)
    print("Example 4: Document Integration (LaTeX)")
    print("="*60)

    output_manager = OutputManager()

    figures_dir = output_manager.config.output_dir / 'figures'
    sample_figure = figures_dir / 'bar' / 'example_bar_performance.pdf'

    if sample_figure.exists():
        latex_snippet = f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{../outputs/figures/bar/example_bar_performance.pdf}}
    \\caption{{Model performance comparison across different methods.}}
    \\label{{fig:performance_comparison}}
\\end{{figure}}
"""
        print("Generated LaTeX snippet:")
        print(latex_snippet)

        snippet_path = Path(output_manager.config.output_dir) / 'data' / 'figure_snippet.tex'
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snippet_path, 'w') as f:
            f.write(latex_snippet)
        print(f"Saved: {snippet_path}")
    else:
        print("Sample figure not found. Run other examples first.")


def example_data_processing():
    """Demonstrate data reading and processing."""
    print("\n" + "="*60)
    print("Example 5: Data Reading and Processing")
    print("="*60)

    output_manager = OutputManager()
    data_dir = Path(output_manager.config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_data = generate_sample_data()
    performance_df = sample_data['performance']

    csv_path = data_dir / 'sample_performance_data.csv'
    performance_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    reader = DataReader()
    reader.read(csv_path)

    print(f"\nData shape: {reader.shape}")
    print(f"Columns: {reader.columns}")

    stats = reader.get_statistics()
    print("\nStatistics:")
    print(stats.to_string())

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


def main():
    """Run all examples."""
    print("Academic Figure Generator - Example Usage")
    print("="*60)

    try:
        example_basic_charts()
        example_heatmap()
        example_multi_panel()
        example_document_integration()
        example_data_processing()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nGenerated files are in:")
        print("  - outputs/figures/ (all chart types)")
        print("  - outputs/data/ (sample datasets)")
        print("\nSupported formats: PDF, SVG")
        print("="*60)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()