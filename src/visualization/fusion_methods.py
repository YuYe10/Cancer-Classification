"""Fusion method comparison visualization."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_fusion_comparison(fusion_methods, metrics, save_dir='.'):
    """Plot fusion methods comparison.
    
    Args:
        fusion_methods: List of method names
        metrics: Dict of {method_name: {metric: value}}
        save_dir: Save directory
    """
    df = pd.DataFrame(metrics).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df.index))
    width = 0.2
    
    for idx, metric in enumerate(df.columns):
        offset = (idx - len(df.columns) / 2 + 0.5) * width
        ax.bar(x + offset, df[metric], width, label=metric, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Fusion Methods Performance Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fusion_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap_correlation(data, method_names, save_dir='.'):
    """Plot correlation heatmap between modalities.
    
    Args:
        data: Feature correlation matrix
        method_names: Names of methods/modalities
        save_dir: Save directory
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=method_names, yticklabels=method_names,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_factor_variance(explained_variance, save_dir='.'):
    """Plot explained variance of latent factors.
    
    Args:
        explained_variance: Array of explained variance ratios
        save_dir: Save directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(len(explained_variance)), explained_variance, 
            color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel("Factor Index", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title("Individual Factor Variance", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cumulative variance
    cumsum = np.cumsum(explained_variance)
    ax2.plot(range(len(cumsum)), cumsum, marker='o', linewidth=2, markersize=6, color='darkorange')
    ax2.fill_between(range(len(cumsum)), cumsum, alpha=0.3, color='darkorange')
    ax2.set_xlabel("Number of Factors", fontsize=12)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax2.set_title("Cumulative Explained Variance", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_factor_variance.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_convergence(train_loss, val_loss, save_dir='.'):
    """Plot training and validation loss convergence.
    
    Args:
        train_loss: Training loss history
        val_loss: Validation loss history
        save_dir: Save directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'o-', linewidth=2, markersize=4, label='Training Loss', color='blue')
    ax.plot(epochs, val_loss, 's-', linewidth=2, markersize=4, label='Validation Loss', color='red')
    
    ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=12, fontweight='bold')
    ax.set_title("Training Convergence", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
