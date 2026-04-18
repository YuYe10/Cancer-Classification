"""Model performance visualization."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import pandas as pd


def plot_confusion_matrices(y_true, y_pred_dict, class_names=None, save_dir='.'):
    """Plot confusion matrices for multiple models.
    
    Args:
        y_true: True labels
        y_pred_dict: Dict of {model_name: predictions}
        class_names: List of class names
        save_dir: Save directory
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    
    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold')
    
    for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[idx].set_title(f"{model_name}")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_pred_prob_dict, save_dir='.'):
    """Plot ROC curves for multiple models.
    
    Args:
        y_true: True binary labels (0/1)
        y_pred_prob_dict: Dict of {model_name: predicted_probabilities}
        save_dir: Save directory
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, y_pred_prob in y_pred_prob_dict.items():
        # If y_pred_prob is 2D, use probability for class 1
        if len(y_pred_prob.shape) > 1:
            y_prob = y_pred_prob[:, 1] if y_pred_prob.shape[1] > 1 else y_pred_prob[:, 0]
        else:
            y_prob = y_pred_prob
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_performance_comparison(metrics_dict, save_dir='.'):
    """Plot model performance metrics comparison.
    
    Args:
        metrics_dict: Dict of {model_name: {metric_name: value}}
        save_dir: Save directory
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    fig, axes = plt.subplots(1, len(df.columns), figsize=(5*len(df.columns), 5))
    
    if len(df.columns) == 1:
        axes = [axes]
    
    fig.suptitle("Model Performance Metrics", fontsize=16, fontweight='bold')
    
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    
    for idx, metric in enumerate(df.columns):
        values = df[metric].values
        x_pos = np.arange(len(values))
        axes[idx].bar(x_pos, values, color=colors[:len(values)], alpha=0.8, edgecolor='black')
        axes[idx].set_ylabel(metric, fontsize=11)
        axes[idx].set_title(f"{metric}", fontsize=12)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(df.index, rotation=45, ha='right')
        axes[idx].set_ylim([0, 1.0])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_results(ablation_results, save_dir='.'):
    """Plot ablation experiment results.
    
    Args:
        ablation_results: Dict of {ablation_name: accuracy}
        save_dir: Save directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(ablation_results.keys())
    accs = list(ablation_results.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    bars = ax.bar(names, accs, color=colors[:len(names)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Ablation Study Results", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_results.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_sample_counts(sample_count_dict, save_dir='.'):
    """Plot sample counts for each class.
    
    Args:
        sample_count_dict: Dict of {class_name: count}
        save_dir: Save directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(sample_count_dict.keys())
    counts = list(sample_count_dict.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = ax.bar(classes, counts, color=colors[:len(classes)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel("Sample Count", fontsize=12, fontweight='bold')
    ax.set_title("Dataset Sample Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
