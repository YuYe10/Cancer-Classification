"""SHAP explainability module for model interpretability."""

import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def run_shap_analysis(model, X, feature_names=None, model_name="Model", output_dir=None):
    """
    Generate SHAP explanations for a fitted model.
    
    Parameters:
    -----------
    model : fitted sklearn model
        The model to explain (e.g., SVM, XGBoost, RF)
    X : array-like, shape (n_samples, n_features)
        Feature matrix for explanation
    feature_names : list, optional
        Feature names for visualization
    model_name : str
        Name of model for output filenames
    output_dir : Path or str, optional
        Directory to save outputs. If None, creates outputs/figures/
    
    Returns:
    --------
    dict with SHAP values and explainer
    """
    if output_dir is None:
        output_dir = Path("outputs/figures")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Create explainer
    try:
        if hasattr(model, 'predict_proba'):
            # For classifiers with probability estimates
            explainer = shap.Explainer(model.predict_proba, X, feature_names=feature_names)
        else:
            explainer = shap.Explainer(model.predict, X, feature_names=feature_names)
    except:
        # Fallback to TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, X)
    
    shap_values = explainer(X)
    
    # Generate visualizations
    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: shap_summary_{model_name}.png")
    except Exception as e:
        print(f"Warning: Could not generate summary plot: {e}")
    
    # Mean absolute SHAP values (feature importance)
    if hasattr(shap_values, 'values'):
        if len(shap_values.values.shape) == 3:  # Multi-class
            mean_shap = np.abs(shap_values.values).mean(axis=(0, 2))  # Average over samples and classes
        else:
            mean_shap = np.abs(shap_values.values).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Feature importance bar plot
    importance_df_dict = {feature_names[i]: mean_shap[i] for i in range(len(feature_names))}
    sorted_features = sorted(importance_df_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    features, importances = zip(*sorted_features)
    ax.barh(features, importances, color='steelblue')
    ax.set_xlabel('Mean |SHAP value| (average impact on model output)')
    ax.set_title(f'Top 20 Feature Importance ({model_name})')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_importance_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: shap_importance_{model_name}.png")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_importance': importance_df_dict,
    }