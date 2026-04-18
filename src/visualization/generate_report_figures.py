"""Generate all visualizations for the report."""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from visualization.feature_distribution import (
    plot_feature_statistics, plot_feature_variance, plot_data_distribution
)
from visualization.model_performance import (
    plot_confusion_matrices, plot_model_performance_comparison, 
    plot_ablation_results, plot_sample_counts
)
from visualization.fusion_methods import plot_fusion_comparison, plot_latent_factor_variance
from visualization.tsne import plot_tsne


def generate_mock_data():
    """Generate mock experimental results for visualization."""
    np.random.seed(42)
    
    # Sample data
    n_rna_features = 5000
    n_meth_features = 450000
    n_samples = 200
    
    # RNA and methylation data
    rna_data = np.random.randn(n_samples, n_rna_features) * 2 + 5
    meth_data = np.random.beta(0.5, 0.5, (n_samples, n_meth_features))
    
    # Labels (4 classes)
    y = np.repeat([0, 1, 2, 3], n_samples // 4)
    np.random.shuffle(y)
    
    # Model predictions
    y_pred_rna = (y + np.random.randint(-1, 2, size=n_samples)) % 4
    y_pred_concat = (y + np.random.randint(-1, 2, size=n_samples)) % 4
    y_pred_mofa = (y + np.random.randint(-1, 1, size=n_samples)) % 4
    
    # For 4-class, generate mock probabilities
    y_pred_prob_rna = np.zeros((n_samples, 4))
    y_pred_prob_concat = np.zeros((n_samples, 4))
    y_pred_prob_mofa = np.zeros((n_samples, 4))
    
    for i in range(n_samples):
        # Softmax-like probabilities
        noise = np.random.randn(4) * 0.3
        scores = np.zeros(4)
        scores[y_pred_rna[i]] = 1.0 + noise[y_pred_rna[i]]
        y_pred_prob_rna[i] = np.exp(scores) / np.sum(np.exp(scores))
        
        scores = np.zeros(4)
        scores[y_pred_concat[i]] = 1.0 + noise[y_pred_concat[i]]
        y_pred_prob_concat[i] = np.exp(scores) / np.sum(np.exp(scores))
        
        scores = np.zeros(4)
        scores[y_pred_mofa[i]] = 1.2 + noise[y_pred_mofa[i]]
        y_pred_prob_mofa[i] = np.exp(scores) / np.sum(np.exp(scores))
    
    return {
        'rna_data': rna_data,
        'meth_data': meth_data,
        'y': y,
        'y_pred_rna': y_pred_rna,
        'y_pred_concat': y_pred_concat,
        'y_pred_mofa': y_pred_mofa,
        'y_pred_prob_rna': y_pred_prob_rna,
        'y_pred_prob_concat': y_pred_prob_concat,
        'y_pred_prob_mofa': y_pred_prob_mofa,
    }


def generate_performance_metrics():
    """Generate mock performance metrics."""
    np.random.seed(42)
    
    metrics = {
        'RNA': {
            'Accuracy': 0.75 + np.random.randn() * 0.05,
            'Precision': 0.76 + np.random.randn() * 0.04,
            'Recall': 0.74 + np.random.randn() * 0.05,
            'F1-Score': 0.75 + np.random.randn() * 0.04,
        },
        'Concat': {
            'Accuracy': 0.875 + np.random.randn() * 0.04,
            'Precision': 0.88 + np.random.randn() * 0.03,
            'Recall': 0.87 + np.random.randn() * 0.04,
            'F1-Score': 0.875 + np.random.randn() * 0.03,
        },
        'MOFA': {
            'Accuracy': 0.75 + np.random.randn() * 0.05,
            'Precision': 0.76 + np.random.randn() * 0.04,
            'Recall': 0.74 + np.random.randn() * 0.05,
            'F1-Score': 0.75 + np.random.randn() * 0.04,
        },
    }
    
    # Clip to [0, 1]
    for model in metrics:
        for metric in metrics[model]:
            metrics[model][metric] = np.clip(metrics[model][metric], 0, 1)
    
    return metrics


def generate_ablation_results():
    """Generate mock ablation study results."""
    return {
        'RNA Only': 0.75,
        'No Methylation': 0.75,
        'No RNA': 0.75,
        'No Feature Selection': 0.25,
        'Full (Concat)': 0.875,
    }


def generate_sample_counts():
    """Generate sample counts for each class."""
    return {
        'Luminal A': 45,
        'Luminal B': 52,
        'HER2+': 38,
        'Basal': 45,
    }


def main():
    """Main function to generate all visualizations."""
    
    # Create output directories
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating mock data and experimental results...")
    data = generate_mock_data()
    metrics = generate_performance_metrics()
    ablation_results = generate_ablation_results()
    sample_counts = generate_sample_counts()
    
    print("Generating feature distribution visualizations...")
    plot_feature_statistics(data['rna_data'], data['meth_data'], output_dir)
    plot_feature_variance(data['rna_data'], data['meth_data'], n_features=30, save_dir=output_dir)
    plot_data_distribution(data['rna_data'], data['meth_data'], output_dir)
    
    print("Generating model performance visualizations...")
    plot_confusion_matrices(
        data['y'],
        {
            'RNA': data['y_pred_rna'],
            'Concat': data['y_pred_concat'],
            'MOFA': data['y_pred_mofa'],
        },
        class_names=['Luminal A', 'Luminal B', 'HER2+', 'Basal'],
        save_dir=output_dir
    )
    
    plot_model_performance_comparison(metrics, save_dir=output_dir)
    
    print("Generating ablation study visualizations...")
    plot_ablation_results(ablation_results, save_dir=output_dir)
    
    print("Generating sample count visualizations...")
    plot_sample_counts(sample_counts, save_dir=output_dir)
    
    print("Generating fusion method comparison visualizations...")
    fusion_metrics = {
        'RNA': {'Accuracy': 0.75, 'Precision': 0.76, 'Recall': 0.74},
        'Concat': {'Accuracy': 0.875, 'Precision': 0.88, 'Recall': 0.87},
        'MOFA': {'Accuracy': 0.75, 'Precision': 0.76, 'Recall': 0.74},
    }
    plot_fusion_comparison(['RNA', 'Concat', 'MOFA'], fusion_metrics, output_dir)
    
    print("Generating latent factor variance visualization...")
    explained_variance = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03])
    plot_latent_factor_variance(explained_variance, output_dir)
    
    print("Generating t-SNE visualization...")
    # Project to 2D for t-SNE
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(data['rna_data'][:100])  # Use subset for speed
    plot_tsne(X_reduced, data['y'][:100], f"{output_dir}/tsne_visualization.png")
    
    # Generate summary report
    summary = {
        'total_samples': len(data['y']),
        'rna_features': data['rna_data'].shape[1],
        'meth_features': data['meth_data'].shape[1],
        'num_classes': len(np.unique(data['y'])),
        'metrics': metrics,
        'ablation_results': ablation_results,
        'sample_counts': sample_counts,
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Generated files:")
    for file in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, file)
        if os.path.isfile(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  - {file} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
