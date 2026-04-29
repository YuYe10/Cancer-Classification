#!/usr/bin/env python3
"""Train a Concat model (XGBoost) and generate SHAP explanations.

Saves:
- outputs/figures/shap_summary.png
- outputs/figures/shap_class_<label>_dependence.png (top features)
"""
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.data.loader import load_data
from src.data.align import align_samples
from src.data.preprocess import RNAPreprocessor, MethPreprocessor
from src.pipeline import _select_features, _preprocess_view
from src.models.train import train_xgboost
from src.pipeline import LABEL_MAP, INV_LABEL_MAP

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config' / 'exp_concat_cv.yaml'
OUT_DIR = ROOT / 'outputs' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(CONFIG_PATH)

    # Load and align
    rna, meth, clinical = load_data(cfg)
    rna, meth, clinical = align_samples(rna, meth, clinical)

    # Map labels
    y = clinical['label'].map(LABEL_MAP)

    # Drop rare classes (<2)
    counts = y.value_counts()
    keep = counts[counts >= 2].index.tolist()
    mask = y.isin(keep)
    rna = rna.loc[:, mask.index[mask]]
    meth = meth.loc[:, mask.index[mask]]
    y = y.loc[mask.index[mask]]

    # For SHAP, train on full data (no leakage concerns here as this is explanation)
    use_rna = True
    use_meth = True
    use_feature_selection = True

    # Select features
    rna_features, meth_features = _select_features(
        rna_train=rna, meth_train=meth, config=cfg, use_rna=use_rna, use_meth=use_meth, use_feature_selection=use_feature_selection
    )

    # Fit preprocessors on full data (we treat full data as training here)
    rna_pp = RNAPreprocessor(top_k=len(rna_features), selected_features=rna_features)
    rna_pp.fit(rna)
    X_rna = rna_pp.transform(rna)
    meth_pp = MethPreprocessor(top_k=len(meth_features), selected_features=meth_features)
    meth_pp.fit(meth)
    X_meth = meth_pp.transform(meth)

    X = np.concatenate([X_rna, X_meth], axis=1)
    feature_names = list(rna_pp.fitted_features) + list(meth_pp.fitted_features)

    # Train XGBoost model for explainability
    model = train_xgboost(X, y, cfg)
    # If wrapper, get underlying model
    xgb_model = getattr(model, 'model', model)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    # Normalize shap_values to a list of arrays with shape (n_samples, n_features)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # try common orderings
            if shap_values.shape[0] == X.shape[0]:
                # (n_samples, n_features, n_classes)
                shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
            elif shap_values.shape[0] == len(LABEL_MAP):
                # (n_classes, n_samples, n_features)
                shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
            else:
                shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
        else:
            shap_values = [shap_values]

    # Use DataFrame for features to keep column names
    X_df = pd.DataFrame(X, columns=feature_names)
    summary_path = OUT_DIR / 'shap_summary.png'
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, features=X_df, feature_names=feature_names, show=False)
    plt.savefig(summary_path, bbox_inches='tight', dpi=160)
    plt.close()

    # Per-class dependence plots: pick top 5 features by mean(|shap|) per class and plot
    class_idx_map = {v: k for k, v in LABEL_MAP.items()}
    print('X shape:', X.shape)
    print('Number of shap outputs:', len(shap_values))
    for i, sv_check in enumerate(shap_values):
        try:
            print(f'shap_values[{i}] shape: {np.array(sv_check).shape}')
        except Exception:
            print(f'shap_values[{i}] shape: unknown')

    for class_idx in range(len(shap_values)):
        sv = np.array(shap_values[class_idx])
        # If shape appears swapped (n_features, n_samples), transpose
        if sv.shape[0] == X.shape[1] and sv.shape[1] == X.shape[0]:
            sv = sv.T
        mean_abs = np.mean(np.abs(sv), axis=0)
        top_idx = np.argsort(mean_abs)[-5:][::-1]
        for fi in top_idx:
            feat_name = feature_names[fi]
            fig = plt.figure(figsize=(6,4))
            try:
                shap.dependence_plot(ind=fi, shap_values=sv, features=X_df, feature_names=feature_names, show=False)
            except Exception:
                shap.dependence_plot(ind=fi, shap_values=sv, features=X, feature_names=feature_names, show=False)
            outp = OUT_DIR / f'shap_class_{class_idx_map[class_idx]}_{feat_name}.png'
            fig.tight_layout()
            fig.savefig(outp, dpi=160, bbox_inches='tight')
            plt.close(fig)

    print('Saved SHAP summary to', summary_path)


if __name__ == '__main__':
    main()
