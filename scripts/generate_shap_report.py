#!/usr/bin/env python3
"""Generate SHAP top-features CSV and a short markdown report.

Outputs:
- outputs/logs/shap_top_features.csv
- outputs/logs/shap_report.md
"""
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import shap

from src.data.loader import load_data


def _load_data_robust(cfg):
    """Fallback loader that forces the Python CSV engine for large files."""
    import pandas as pd
    clinical = pd.read_csv(cfg['data']['clinical_path'], sep='\t', engine='python')
    clinical.columns = clinical.columns.str.strip()

    if 'PAM50' in clinical.columns and 'label' not in clinical.columns:
        clinical = clinical.rename(columns={'PAM50': 'label'})

    if 'sample' not in clinical.columns:
        raise KeyError("Clinical file must contain a 'sample' column")
    if 'label' not in clinical.columns:
        raise KeyError("Clinical file must contain a 'label' or 'PAM50' column")

    clinical['label'] = clinical['label'].replace({'Her2': 'HER2', 'her2': 'HER2'})
    clinical = clinical[clinical['label'].isin(['LumA', 'LumB', 'HER2', 'Basal'])]
    clinical = clinical.drop_duplicates(subset='sample', keep='first').set_index('sample')

    common_samples = clinical.index.tolist()

    # read headers using python engine to be robust
    rna_header = pd.read_csv(cfg['data']['rna_path'], nrows=0, engine='python')
    meth_header = pd.read_csv(cfg['data']['meth_path'], nrows=0, engine='python')

    rna_samples = [sample for sample in rna_header.columns if sample in common_samples]
    meth_samples = [sample for sample in meth_header.columns if sample in common_samples]

    rna = pd.read_csv(
        cfg['data']['rna_path'],
        index_col=0,
        usecols=lambda col: col == rna_header.columns[0] or col in rna_samples,
        low_memory=False,
        engine='python',
    )
    meth = pd.read_csv(
        cfg['data']['meth_path'],
        index_col=0,
        usecols=lambda col: col == meth_header.columns[0] or col in meth_samples,
        low_memory=False,
        engine='python',
    )

    return rna, meth, clinical
from src.data.align import align_samples
from src.data.preprocess import RNAPreprocessor, MethPreprocessor
from src.pipeline import _select_features
from src.models.train import train_xgboost
from src.pipeline import LABEL_MAP

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config' / 'exp_concat_cv.yaml'
LOGS = ROOT / 'outputs' / 'logs'
FIGS = ROOT / 'outputs' / 'figures'
LOGS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(CONFIG_PATH)

    # Load and align (use robust loader if necessary)
    try:
        rna, meth, clinical = load_data(cfg)
    except Exception:
        rna, meth, clinical = _load_data_robust(cfg)
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

    use_rna = True
    use_meth = True
    use_feature_selection = True

    # Select features
    rna_features, meth_features = _select_features(
        rna_train=rna, meth_train=meth, config=cfg, use_rna=use_rna, use_meth=use_meth, use_feature_selection=use_feature_selection
    )

    # Fit preprocessors on full data
    rna_pp = RNAPreprocessor(top_k=len(rna_features), selected_features=rna_features)
    rna_pp.fit(rna)
    X_rna = rna_pp.transform(rna)
    meth_pp = MethPreprocessor(top_k=len(meth_features), selected_features=meth_features)
    meth_pp.fit(meth)
    X_meth = meth_pp.transform(meth)

    X = np.concatenate([X_rna, X_meth], axis=1)
    feature_names = list(rna_pp.fitted_features) + list(meth_pp.fitted_features)
    X_df = pd.DataFrame(X, columns=feature_names)

    # Train model
    model = train_xgboost(X, y, cfg)
    xgb_model = getattr(model, 'model', model)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    # Normalize
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            if shap_values.shape[0] == X.shape[0]:
                shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
            elif shap_values.shape[0] == len(LABEL_MAP):
                shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
            else:
                shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
        else:
            shap_values = [shap_values]

    records = []
    for class_idx, sv in enumerate(shap_values):
        sv = np.array(sv)
        if sv.shape[0] == X.shape[1] and sv.shape[1] == X.shape[0]:
            sv = sv.T
        mean_abs = np.mean(np.abs(sv), axis=0)
        for feat_idx, val in enumerate(mean_abs):
            records.append({'class_idx': class_idx, 'class_name': LABEL_MAP[class_idx] if class_idx in LABEL_MAP else str(class_idx), 'feature': feature_names[feat_idx], 'mean_abs_shap': float(val)})

    df = pd.DataFrame.from_records(records)
    # Map LABEL_MAP is int mapping; invert
    inv_label = {v:k for k,v in LABEL_MAP.items()}
    df['class_name'] = df['class_idx'].map(inv_label)

    # Save top features per class
    out_csv = LOGS / 'shap_top_features.csv'
    df_sorted = df.sort_values(['class_idx', 'mean_abs_shap'], ascending=[True, False])
    df_sorted.to_csv(out_csv, index=False)

    # Create markdown report
    report_lines = [
        '# SHAP Top Features Report',
        '',
        f'Generated from config: {CONFIG_PATH}',
        '',
    ]
    for class_idx in sorted(df_sorted['class_idx'].unique()):
        cname = inv_label.get(class_idx, str(class_idx))
        report_lines.append(f'## Class: {cname} (index {class_idx})')
        topk = df_sorted[df_sorted['class_idx'] == class_idx].head(10)
        report_lines.append('| Rank | Feature | mean_abs_shap |')
        report_lines.append('|---:|---|---:|')
        for i, row in enumerate(topk.itertuples(), start=1):
            report_lines.append(f'| {i} | {row.feature} | {row.mean_abs_shap:.6f} |')
        report_lines.append('')
        # link to images if exist
        report_lines.append('Related figures:')
        for feat in topk['feature'].head(5):
            # sanitize filename
            fname = f'shap_class_{cname}_{feat}.png'
            fpath = FIGS / fname
            if fpath.exists():
                report_lines.append(f'- ![]({fpath.relative_to(ROOT)})')
        report_lines.append('')

    out_md = LOGS / 'shap_report.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('Wrote', out_csv, out_md)


if __name__ == '__main__':
    main()
