#!/usr/bin/env python3
"""t-SNE visualization for multi-omics data in BRCA molecular subtype classification."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path


LABEL_MAP = {
    'LumA': 0,
    'LumB': 1,
    'HER2': 2,
    'Basal': 3,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def plot_tsne(X, y, class_names=None, title='t-SNE Visualization', save_path=None, perplexity=30, n_iter=1000):
    """Plot t-SNE visualization of high-dimensional data.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
        class_names: list of class names
        title: plot title
        save_path: path to save figure
        perplexity: t-SNE perplexity parameter
        n_iter: t-SNE number of iterations
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']

    unique_labels = np.unique(y)

    for idx, label in enumerate(unique_labels):
        mask = y == label
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            label=class_name,
            alpha=0.7,
            s=80,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot: {save_path}")
    plt.close()


def plot_tsne_comparison(X_dict, y, method_names=None, title='t-SNE Comparison', save_path=None):
    """Plot t-SNE comparison for multiple feature sets.

    Args:
        X_dict: dict of {method_name: feature_matrix}
        y: True labels (n_samples,)
        method_names: list of method names corresponding to X_dict keys
        title: plot title
        save_path: path to save figure
    """
    n_methods = len(X_dict)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_methods > 1 else np.array([axes])

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']
    unique_labels = np.unique(y)

    for idx, (method_name, X) in enumerate(X_dict.items()):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        X_2d = tsne.fit_transform(X)

        ax = axes[idx]

        for label_idx, label in enumerate(unique_labels):
            mask = y == label
            label_name = ['LumA', 'LumB', 'HER2', 'Basal'][label] if label < 4 else f"Class {label}"
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colors[label_idx % len(colors)],
                marker=markers[label_idx % len(markers)],
                label=label_name,
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidth=0.5
            )

        display_name = method_name if method_names is None else method_names[idx]
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)

    for ax in axes[n_methods:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE comparison: {save_path}")
    plt.close()


def plot_tsne_with_centroids(X, y, class_names=None, title='t-SNE with Class Centroids', save_path=None):
    """Plot t-SNE with class centroids marked.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: True labels (n_samples,)
        class_names: list of class names
        title: plot title
        save_path: path to save figure
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']
    unique_labels = np.unique(y)

    for idx, label in enumerate(unique_labels):
        mask = y == label
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"

        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            label=class_name,
            alpha=0.6,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

        centroid_x = X_2d[mask, 0].mean()
        centroid_y = X_2d[mask, 1].mean()
        ax.scatter(
            [centroid_x], [centroid_y],
            c=colors[idx % len(colors)],
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            zorder=10
        )
        ax.annotate(
            f'{class_name} center',
            (centroid_x, centroid_y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )

    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE with centroids: {save_path}")
    plt.close()


def main():
    import argparse
    import json
    import glob

    parser = argparse.ArgumentParser(description='Generate t-SNE visualizations')
    parser.add_argument('--results-dir', type=str, default='outputs/logs',
                        help='Directory containing CV result JSON files')
    parser.add_argument('--output-dir', type=str, default='outputs/figures',
                        help='Output directory for figures')
    parser.add_argument('--method', type=str, default=None,
                        help='Specific method to visualize (default: all)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method:
        json_files = glob.glob(str(Path(args.results_dir) / f'*{args.method}*.json'))
    else:
        json_files = glob.glob(str(Path(args.results_dir) / '*.json'))

    all_X = []
    all_y = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        fold_metrics = metrics.get('fold_metrics', [])

        for fold in fold_metrics:
            X = fold.get('X_test')
            y_true = fold.get('y_true')

            if X is not None and y_true is not None:
                all_X.append(np.array(X))
                all_y.extend(y_true)

    if not all_X:
        print("No valid results found")
        return

    X_concat = np.vstack(all_X) if all_X else None
    y_arr = np.array(all_y)

    if X_concat is not None and len(X_arr := X_concat.shape) == 2:
        plot_tsne(
            X_concat, y_arr,
            class_names=['LumA', 'LumB', 'HER2', 'Basal'],
            title='BRCA Molecular Subtypes - t-SNE Visualization',
            save_path=output_dir / 'tsne_brca_subtypes.png'
        )
        plot_tsne_with_centroids(
            X_concat, y_arr,
            class_names=['LumA', 'LumB', 'HER2', 'Basal'],
            title='BRCA Molecular Subtypes - t-SNE with Centroids',
            save_path=output_dir / 'tsne_brca_subtypes_with_centroids.png'
        )
        print(f"Generated t-SNE plots for {X_concat.shape[0]} samples")
    else:
        print(f"Could not process data shape: {X_arr if 'X_arr' in dir() else 'N/A'}")


if __name__ == '__main__':
    main()