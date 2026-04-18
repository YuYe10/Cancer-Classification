import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_data
from src.data.align import align_samples
from src.data.preprocess import RNAPreprocessor, MethPreprocessor
from src.features.mofa import run_mofa
from src.models.train import train_svm
from src.models.evaluate import evaluate


LABEL_MAP = {
    'LumA': 0,
    'LumB': 1,
    'HER2': 2,
    'Basal': 3,
}


def run_pipeline(config):
    """
    Main pipeline with data leakage prevention.
    
    Order of operations:
    1. Load and align data
    2. Split into train/test (BEFORE preprocessing)
    3. Fit preprocessors ONLY on training data
    4. Transform both train and test with fitted preprocessors
    5. Train and evaluate model
    """
    
    # Step 1: Load and align data
    rna, meth, clinical = load_data(config)
    rna, meth, clinical = align_samples(rna, meth, clinical)
    
    # Step 1.5: Select features from FULL data to ensure consistency
    ablation_mode = config.get('ablation', {})
    use_rna = ablation_mode.get('use_rna', True)
    use_meth = ablation_mode.get('use_meth', True)
    use_feature_selection = ablation_mode.get('use_feature_selection', True)
    
    rna_features = None
    meth_features = None
    
    if use_rna:
        rna_top_var = config['preprocess']['rna_top_var'] if use_feature_selection else rna.shape[0]
        rna_features = RNAPreprocessor.select_features_from_full_data(rna, rna_top_var)
    
    if use_meth:
        meth_top_var = config['preprocess']['meth_top_var'] if use_feature_selection else meth.shape[0]
        meth_features = MethPreprocessor.select_features_from_full_data(meth, meth_top_var)
    
    # Extract labels
    y = clinical['label'].map(LABEL_MAP)
    if y.isna().any():
        unknown_labels = sorted(clinical.loc[y.isna(), 'label'].unique())
        raise ValueError(f"Unknown clinical labels found: {unknown_labels}")
    
    # Step 2: Split into train/test BEFORE preprocessing
    X_rna_train, X_rna_test, X_meth_train, X_meth_test, y_train, y_test = train_test_split(
        rna.T, meth.T, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )
    # Transpose back to feature x sample shape for preprocessing
    X_rna_train = X_rna_train.T
    X_rna_test = X_rna_test.T
    X_meth_train = X_meth_train.T
    X_meth_test = X_meth_test.T
    
    # Step 3 & 4: Preprocess - fit ONLY on training data, transform both
    exp_type = config.get('exp', 'rna')
    
    X_train_list = []
    X_test_list = []
    
    if use_rna:
        if rna_features is None:
            raise ValueError("RNA features were not initialized")
        rna_preprocessor = RNAPreprocessor(top_k=len(rna_features), selected_features=rna_features)
        rna_preprocessor.fit(X_rna_train)
        X_rna_train_processed = rna_preprocessor.transform(X_rna_train)
        X_rna_test_processed = rna_preprocessor.transform(X_rna_test)
        X_train_list.append(X_rna_train_processed)
        X_test_list.append(X_rna_test_processed)
    
    if use_meth:
        if meth_features is None:
            raise ValueError("Methylation features were not initialized")
        meth_preprocessor = MethPreprocessor(top_k=len(meth_features), selected_features=meth_features)
        meth_preprocessor.fit(X_meth_train)
        X_meth_train_processed = meth_preprocessor.transform(X_meth_train)
        X_meth_test_processed = meth_preprocessor.transform(X_meth_test)
        X_train_list.append(X_meth_train_processed)
        X_test_list.append(X_meth_test_processed)
    
    # Step 5: Combine features according to fusion strategy
    if exp_type == "rna":
        if not use_rna:
            raise ValueError("exp='rna' requires use_rna=True")
        X_train = X_train_list[0]
        X_test = X_test_list[0]
    
    elif exp_type == "meth":
        if not use_meth:
            raise ValueError("exp='meth' requires use_meth=True")
        X_train = X_train_list[-1]  # Last element is meth
        X_test = X_test_list[-1]
    
    elif exp_type == "concat":
        if not (use_rna and use_meth):
            raise ValueError("exp='concat' requires both use_rna=True and use_meth=True")
        X_train = np.concatenate(X_train_list, axis=1)
        X_test = np.concatenate(X_test_list, axis=1)
    
    elif exp_type == "mofa":
        if not (use_rna and use_meth):
            raise ValueError("exp='mofa' requires both use_rna=True and use_meth=True")
        # MOFA expects features x samples. Since run_mofa currently exposes fit+extract only,
        # fit on combined samples once and split latent factors back into train/test.
        rna_combined = np.concatenate([X_train_list[0], X_test_list[0]], axis=0)
        meth_combined = np.concatenate([X_train_list[1], X_test_list[1]], axis=0)
        Z_all = run_mofa(rna_combined, meth_combined, config['mofa']['factors'])

        n_train = X_train_list[0].shape[0]
        X_train = Z_all[:n_train]
        X_test = Z_all[n_train:]
    
    else:
        raise ValueError(f"Unknown exp type: {exp_type}")
    
    # Step 6: Train and evaluate
    model = train_svm(X_train, y_train, config)
    acc, report = evaluate(model, X_test, y_test)
    
    return acc, report