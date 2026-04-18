"""Feature distribution visualization."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def plot_feature_statistics(rna_data, meth_data, save_dir):
    """Plot feature statistics for RNA and methylation data."""
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Feature Distribution Statistics", fontsize=16, fontweight='bold')
    
    # RNA mean and std
    rna_mean = np.mean(rna_data, axis=0)
    rna_std = np.std(rna_data, axis=0)
    axes[0, 0].hist(rna_mean, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel("Mean Expression Level")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("RNA-seq Mean Distribution")
    
    axes[0, 1].hist(rna_std, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel("Standard Deviation")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("RNA-seq Std Distribution")
    
    # Methylation mean and std
    meth_mean = np.mean(meth_data, axis=0)
    meth_std = np.std(meth_data, axis=0)
    axes[1, 0].hist(meth_mean, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel("Mean Methylation Level")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Methylation Mean Distribution")
    
    axes[1, 1].hist(meth_std, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel("Standard Deviation")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Methylation Std Distribution")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_variance(rna_data, meth_data, n_features=50, save_dir='.'):
    """Plot top features by variance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Top Features by Variance", fontsize=16, fontweight='bold')
    
    rna_var = np.var(rna_data, axis=0)
    rna_top_idx = np.argsort(rna_var)[-n_features:][::-1]
    rna_top_var = rna_var[rna_top_idx]
    
    axes[0].barh(range(n_features), rna_top_var, color='steelblue')
    axes[0].set_xlabel("Variance")
    axes[0].set_ylabel("Feature Index (Top)")
    axes[0].set_title(f"Top {n_features} RNA-seq Features")
    axes[0].invert_yaxis()
    
    meth_var = np.var(meth_data, axis=0)
    meth_top_idx = np.argsort(meth_var)[-n_features:][::-1]
    meth_top_var = meth_var[meth_top_idx]
    
    axes[1].barh(range(n_features), meth_top_var, color='darkorange')
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Feature Index (Top)")
    axes[1].set_title(f"Top {n_features} Methylation Features")
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_variance.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_data_distribution(rna_data, meth_data, save_dir='.'):
    """Plot raw data distribution (samples)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sample Distribution", fontsize=16, fontweight='bold')
    
    axes[0].hist(rna_data.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel("Expression Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"RNA-seq Data Distribution ({rna_data.shape[0]} samples × {rna_data.shape[1]} features)")
    axes[0].set_yscale('log')
    
    axes[1].hist(meth_data.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel("Methylation Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Methylation Data Distribution ({meth_data.shape[0]} samples × {meth_data.shape[1]} features)")
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/data_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
