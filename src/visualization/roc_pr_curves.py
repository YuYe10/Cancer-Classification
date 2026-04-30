#!/usr/bin/env python3
"""Multi-class ROC and PR curve visualization for BRCA molecular subtype classification."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


LABEL_MAP = {
    'LumA': 0,
    'LumB': 1,
    'HER2': 2,
    'Basal': 3,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def compute_roc_pr_curves(y_true, y_pred_prob, n_classes=4):
    """Compute ROC and PR curves for multi-class classification.

    Args:
        y_true: True labels (0, 1, 2, 3)
        y_pred_prob: Predicted probabilities (n_samples, n_classes)
        n_classes: Number of classes

    Returns:
        dict with roc_data and pr_data for each class
    """
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    roc_data = {}
    pr_data = {}

    for i in range(n_classes):
        if y_true_bin[:, i].sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[i] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])
        pr_data[i] = {'precision': precision, 'recall': recall, 'ap': avg_precision}

    return roc_data, pr_data


def plot_multiclass_roc_curves(roc_data, class_names, save_path):
    """Plot multi-class ROC curves (one-vs-rest).

    Args:
        roc_data: dict with class_id -> {'fpr', 'tpr', 'auc'}
        class_names: list of class names
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    for class_id, data in roc_data.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        ax.plot(
            data['fpr'], data['tpr'],
            color=colors[class_id % len(colors)],
            lw=2.5,
            label=f'{class_name} (AUC = {data["auc"]:.3f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('Multi-Class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves: {save_path}")


def plot_multiclass_pr_curves(pr_data, class_names, save_path):
    """Plot multi-class PR curves (one-vs-rest).

    Args:
        pr_data: dict with class_id -> {'precision', 'recall', 'ap'}
        class_names: list of class names
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    for class_id, data in pr_data.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        ax.plot(
            data['recall'], data['precision'],
            color=colors[class_id % len(colors)],
            lw=2.5,
            label=f'{class_name} (AP = {data["ap"]:.3f})'
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Multi-Class Precision-Recall Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curves: {save_path}")


def plot_roc_pr_combined(roc_data, pr_data, class_names, save_path):
    """Plot combined ROC and PR curves side by side.

    Args:
        roc_data: dict with class_id -> {'fpr', 'tpr', 'auc'}
        pr_data: dict with class_id -> {'precision', 'recall', 'ap'}
        class_names: list of class names
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    ax_roc = axes[0]
    for class_id, data in roc_data.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        ax_roc.plot(
            data['fpr'], data['tpr'],
            color=colors[class_id % len(colors)],
            lw=2.5,
            label=f'{class_name} (AUC = {data["auc"]:.3f})'
        )
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves', fontsize=13, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.grid(True, alpha=0.3)

    ax_pr = axes[1]
    for class_id, data in pr_data.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        ax_pr.plot(
            data['recall'], data['precision'],
            color=colors[class_id % len(colors)],
            lw=2.5,
            label=f'{class_name} (AP = {data["ap"]:.3f})'
        )
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
    ax_pr.legend(loc='lower left', fontsize=10)
    ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined ROC/PR curves: {save_path}")


def main():
    import argparse
    import json
    import glob

    parser = argparse.ArgumentParser(description='Generate ROC and PR curves from CV results')
    parser.add_argument('--results-dir', type=str, default='outputs/logs',
                        help='Directory containing CV result JSON files')
    parser.add_argument('--method', type=str, default='concat',
                        help='Method name pattern to match')
    parser.add_argument('--output-dir', type=str, default='outputs/figures',
                        help='Output directory for figures')
    parser.add_argument('--class-names', type=str,
                        default='LumA,LumB,HER2,Basal',
                        help='Comma-separated class names')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = args.class_names.split(',')

    json_files = glob.glob(str(Path(args.results_dir) / f'*{args.method}*.json'))

    all_y_true = []
    all_y_pred_prob = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        fold_metrics = metrics.get('fold_metrics', [])

        for fold in fold_metrics:
            y_true = fold.get('y_true')
            y_pred_prob = fold.get('y_pred_prob')

            if y_true is not None and y_pred_prob is not None:
                all_y_true.extend(y_true)
                all_y_pred_prob.extend(y_pred_prob)

    if not all_y_true:
        print(f"No valid results found for method pattern: {args.method}")
        return

    y_true_arr = np.array(all_y_true)
    y_pred_prob_arr = np.array(all_y_pred_prob)

    n_classes = y_pred_prob_arr.shape[1] if len(y_pred_prob_arr.shape) > 1 else 4

    roc_data, pr_data = compute_roc_pr_curves(y_true_arr, y_pred_prob_arr, n_classes=n_classes)

    plot_multiclass_roc_curves(roc_data, class_names, output_dir / 'roc_curves_multiclass.png')
    plot_multiclass_pr_curves(pr_data, class_names, output_dir / 'pr_curves_multiclass.png')
    plot_roc_pr_combined(roc_data, pr_data, class_names, output_dir / 'roc_pr_combined.png')

    print(f"Generated {len(roc_data)} class ROC curves and {len(pr_data)} class PR curves")


if __name__ == '__main__':
    main()