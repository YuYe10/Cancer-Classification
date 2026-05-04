#!/usr/bin/env python3
"""
Generate SHAP explanations for Concat model.
Simplified version that trains on full data and generates visualizations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_shap_concat():
    """Generate SHAP explanations for Concat model."""
    
    from src.data.loader import load_data
    from src.data.preprocess import preprocess_rna, preprocess_meth
    from src.explain.shap_explain import run_shap_analysis
    
    print("=" * 80)
    print("SHAP EXPLAINABILITY ANALYSIS FOR CONCAT MODEL")
    print("=" * 80)
    
    # Load config
    config_path = Path('config/exp_concat_cv.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Ensure data paths are correct
    config['data'] = {
        'clinical_path': 'datasets/brca_pam50.csv',
        'rna_path': 'datasets/TCGA-BRCA.star_counts.csv',
        'meth_path': 'datasets/TCGA-BRCA.methylation450.csv',
    }
    
    print("\nLoading data...")
    try:
        rna, meth, clinical = load_data(config)
        print(f"  RNA shape: {rna.shape}")
        print(f"  Methylation shape: {meth.shape}")
        print(f"  Samples: {len(clinical)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Preprocess
    print("\nPreprocessing features...")
    preprocess_config = config.get('preprocess', {})
    rna_top_k = preprocess_config.get('rna_top_var', 500)
    meth_top_k = preprocess_config.get('meth_top_var', 500)
    
    print(f"  RNA top-k: {rna_top_k}")
    print(f"  Meth top-k: {meth_top_k}")
    
    try:
        X_rna = preprocess_rna(rna, top_k=rna_top_k)
        X_meth = preprocess_meth(meth, top_k=meth_top_k)
    except Exception as e:
        print(f"Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"  RNA after preprocess: {X_rna.shape}")
    print(f"  Meth after preprocess: {X_meth.shape}")
    
    # Concatenate
    X_concat = np.hstack([X_rna, X_meth])
    feature_names_rna = [f'RNA_{i}' for i in range(X_rna.shape[1])]
    feature_names_meth = [f'Meth_{i}' for i in range(X_meth.shape[1])]
    feature_names = feature_names_rna + feature_names_meth
    
    # Get labels - assuming rna and meth have same samples in same order
    # For simplicity, use the first len(X_rna) samples
    y_data = clinical['label'].iloc[:len(X_rna)]
    
    # Drop HER2 if present
    valid_mask = y_data.isin(['LumA', 'LumB', 'Basal'])
    X_concat = X_concat[valid_mask]
    y_data = y_data[valid_mask].reset_index(drop=True)
    
    print(f"\nFinal data for SHAP:")
    print(f"  X shape: {X_concat.shape}")
    print(f"  y shape: {y_data.shape}")
    print(f"  Classes: {y_data.unique()}")
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_data)
    
    # Train SVM (no need for extra standardization since preprocess already does it)
    print("\nTraining SVM on full data...")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_concat, y_encoded)
    print(f"  Training accuracy: {model.score(X_concat, y_encoded):.4f}")
    
    # Generate SHAP
    print("\nGenerating SHAP explanations...")
    output_dir = Path('outputs/figures')
    
    try:
        shap_result = run_shap_analysis(
            model=model,
            X=X_concat,
            feature_names=feature_names,
            model_name='concat_svm',
            output_dir=output_dir
        )
        
        print(f"\n✓ SHAP analysis complete!")
        print(f"  Plots saved to: {output_dir}/")
        
        # Print top features
        print("\nTop 15 important features:")
        importance = shap_result['feature_importance']
        sorted_feat = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_feat[:15], 1):
            print(f"  {i:2}. {feat:30} | {imp:.6f}")
        
        return shap_result
    
    except Exception as e:
        print(f"Error generating SHAP: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    result = generate_shap_concat()
    sys.exit(0 if result else 1)
