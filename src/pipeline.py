import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
import copy

from src.data.loader import load_data
from src.data.align import align_samples
from src.data.preprocess import RNAPreprocessor, MethPreprocessor
from src.features.mofa import fit_mofa, project_mofa_latent, run_mofa
from src.models.train import train_model, train_classifier
from src.models.evaluate import evaluate, summarize_cv_metrics


LABEL_MAP = {
    'LumA': 0,
    'LumB': 1,
    'HER2': 2,
    'Basal': 3,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


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
    
    ablation_mode = config.get('ablation', {})
    use_rna = ablation_mode.get('use_rna', True)
    use_meth = ablation_mode.get('use_meth', True)
    use_feature_selection = ablation_mode.get('use_feature_selection', True)
    
    # Extract labels
    y = clinical['label'].map(LABEL_MAP)
    if y.isna().any():
        unknown_labels = sorted(clinical.loc[y.isna(), 'label'].unique())
        raise ValueError(f"Unknown clinical labels found: {unknown_labels}")

    evaluation_cfg = config.get('evaluation', {})
    evaluation_mode = evaluation_cfg.get('mode', 'holdout')

    rna, meth, clinical, y, dropped_labels = _drop_rare_classes(
        rna=rna,
        meth=meth,
        clinical=clinical,
        y=y,
        min_count=2,
    )

    if evaluation_mode in {'cv', 'repeated_cv'}:
        metrics = _run_cross_validation(
            rna=rna,
            meth=meth,
            y=y,
            config=config,
            use_rna=use_rna,
            use_meth=use_meth,
            use_feature_selection=use_feature_selection,
            evaluation_cfg=evaluation_cfg,
        )
        metrics['dropped_labels'] = dropped_labels
        return metrics
    
    # Step 2: Split into train/test BEFORE preprocessing
    stratify_target = y if y.value_counts().min() >= 2 else None
    X_rna_train, X_rna_test, X_meth_train, X_meth_test, y_train, y_test = train_test_split(
        rna.T, meth.T, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state'],
        stratify=stratify_target,
    )
    # Transpose back to feature x sample shape for preprocessing
    X_rna_train = X_rna_train.T
    X_rna_test = X_rna_test.T
    X_meth_train = X_meth_train.T
    X_meth_test = X_meth_test.T
    
    exp_type = config.get('exp', 'rna')
    if exp_type == 'stacking':
        return _run_stacking_holdout(
            X_rna_train=X_rna_train,
            X_rna_test=X_rna_test,
            X_meth_train=X_meth_train,
            X_meth_test=X_meth_test,
            y_train=y_train,
            y_test=y_test,
            config=config,
            use_rna=use_rna,
            use_meth=use_meth,
            use_feature_selection=use_feature_selection,
        )

    X_train, X_test = _build_feature_matrices(
        X_rna_train=X_rna_train,
        X_rna_test=X_rna_test,
        X_meth_train=X_meth_train,
        X_meth_test=X_meth_test,
        config=config,
        use_rna=use_rna,
        use_meth=use_meth,
        use_feature_selection=use_feature_selection,
        exp_type=exp_type,
        allow_mofa_full_fit=True,
    )

    model = train_model(X_train, y_train, config)
    metrics = evaluate(model, X_test, y_test, labels=list(LABEL_MAP.values()))
    metrics['mode'] = 'holdout'
    metrics['train_size'] = len(y_train)
    metrics['test_size'] = len(y_test)
    metrics['dropped_labels'] = dropped_labels
    return metrics


def _drop_rare_classes(rna, meth, clinical, y, min_count=2):
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < min_count].index.tolist()
    if not rare_classes:
        return rna, meth, clinical, y, []

    keep_mask = ~y.isin(rare_classes)
    if keep_mask.sum() == 0:
        raise ValueError("All classes were filtered out due to insufficient sample counts")

    y_filtered = y.loc[keep_mask]
    if y_filtered.nunique() < 2:
        raise ValueError("Need at least two classes after filtering rare classes")

    kept_samples = y_filtered.index.tolist()
    rna_filtered = rna.loc[:, kept_samples]
    meth_filtered = meth.loc[:, kept_samples]
    clinical_filtered = clinical.loc[kept_samples]

    dropped_labels = [INV_LABEL_MAP.get(code, str(code)) for code in rare_classes]
    return rna_filtered, meth_filtered, clinical_filtered, y_filtered, dropped_labels


def _select_features(rna_train=None, meth_train=None, config=None, use_rna=True, use_meth=True, use_feature_selection=True):
    rna_features = None
    meth_features = None

    if use_rna:
        if use_feature_selection:
            rna_top_var = config['preprocess']['rna_top_var']
            rna_features = RNAPreprocessor.select_features_from_data(rna_train, rna_top_var)
        else:
            rna_features = rna_train.index.tolist()

    if use_meth:
        if use_feature_selection:
            meth_top_var = config['preprocess']['meth_top_var']
            meth_features = MethPreprocessor.select_features_from_data(meth_train, meth_top_var)
        else:
            meth_features = meth_train.index.tolist()

    return rna_features, meth_features


def _preprocess_view(preprocessor_cls, selected_features, fit_data, transform_data):
    if not selected_features:
        raise ValueError("No features selected for preprocessing")

    preprocessor = preprocessor_cls(top_k=len(selected_features), selected_features=selected_features)
    preprocessor.fit(fit_data)
    return preprocessor.transform(transform_data)


def _build_feature_matrices(
    X_rna_train,
    X_rna_test,
    X_meth_train,
    X_meth_test,
    config,
    use_rna,
    use_meth,
    use_feature_selection,
    exp_type,
    allow_mofa_full_fit=False,
):
    rna_features = None
    meth_features = None

    if use_rna or use_meth:
        rna_features, meth_features = _select_features(
            rna_train=X_rna_train if use_rna else None,
            meth_train=X_meth_train if use_meth else None,
            config=config,
            use_rna=use_rna,
            use_meth=use_meth,
            use_feature_selection=use_feature_selection,
        )

    X_train_list = []
    X_test_list = []

    if use_rna:
        X_rna_train_processed = _preprocess_view(
            preprocessor_cls=RNAPreprocessor,
            selected_features=rna_features,
            fit_data=X_rna_train,
            transform_data=X_rna_train,
        )
        X_rna_test_processed = _preprocess_view(
            preprocessor_cls=RNAPreprocessor,
            selected_features=rna_features,
            fit_data=X_rna_train,
            transform_data=X_rna_test,
        )
        X_train_list.append(X_rna_train_processed)
        X_test_list.append(X_rna_test_processed)

    if use_meth:
        X_meth_train_processed = _preprocess_view(
            preprocessor_cls=MethPreprocessor,
            selected_features=meth_features,
            fit_data=X_meth_train,
            transform_data=X_meth_train,
        )
        X_meth_test_processed = _preprocess_view(
            preprocessor_cls=MethPreprocessor,
            selected_features=meth_features,
            fit_data=X_meth_train,
            transform_data=X_meth_test,
        )
        X_train_list.append(X_meth_train_processed)
        X_test_list.append(X_meth_test_processed)

    if exp_type == "rna":
        if not use_rna:
            raise ValueError("exp='rna' requires use_rna=True")
        return X_train_list[0], X_test_list[0]

    if exp_type == "meth":
        if not use_meth:
            raise ValueError("exp='meth' requires use_meth=True")
        return X_train_list[-1], X_test_list[-1]

    if exp_type == "concat":
        if not (use_rna and use_meth):
            raise ValueError("exp='concat' requires both use_rna=True and use_meth=True")
        return np.concatenate(X_train_list, axis=1), np.concatenate(X_test_list, axis=1)

    if exp_type == "mofa":
        if not (use_rna and use_meth):
            raise ValueError("exp='mofa' requires both use_rna=True and use_meth=True")
        if not allow_mofa_full_fit:
            mofa_model = fit_mofa(
                X_train_list[0],
                X_train_list[1],
                factors=config['mofa']['factors'],
                seed=config['model']['random_state'],
            )
            Z_train = np.asarray(mofa_model.model.nodes["Z"].getExpectation())
            Z_test = project_mofa_latent(
                mofa_model.model,
                X_test_list[0],
                X_test_list[1],
            )
            return Z_train, Z_test

        rna_combined = np.concatenate([X_train_list[0], X_test_list[0]], axis=0)
        meth_combined = np.concatenate([X_train_list[1], X_test_list[1]], axis=0)
        Z_all = run_mofa(
            rna_combined,
            meth_combined,
            factors=config['mofa']['factors'],
            seed=config['model']['random_state'],
        )
        n_train = X_train_list[0].shape[0]
        return Z_all[:n_train], Z_all[n_train:]

    raise ValueError(f"Unknown exp type: {exp_type}")


def _run_cross_validation(rna, meth, y, config, use_rna, use_meth, use_feature_selection, evaluation_cfg):
    exp_type = config.get('exp', 'rna')
    n_splits = evaluation_cfg.get('folds', 5)
    n_repeats = evaluation_cfg.get('repeats', 1)
    random_state = config['model']['random_state']

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    effective_splits = min(n_splits, min_class_count)
    if effective_splits < 2:
        raise ValueError(
            f"Cannot run cross-validation because the smallest class has only {min_class_count} sample(s)."
        )

    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(n_splits=effective_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        X_rna_train = rna.iloc[:, train_idx]
        X_rna_test = rna.iloc[:, test_idx]
        X_meth_train = meth.iloc[:, train_idx]
        X_meth_test = meth.iloc[:, test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if exp_type == 'stacking':
            view_map = _preprocess_modalities(
                X_rna_train=X_rna_train,
                X_rna_test=X_rna_test,
                X_meth_train=X_meth_train,
                X_meth_test=X_meth_test,
                config=config,
                use_rna=use_rna,
                use_meth=use_meth,
                use_feature_selection=use_feature_selection,
            )

            stacking_cfg = config.get('stacking', {})
            base_model_type = stacking_cfg.get('base_model_type', config.get('model', {}).get('type', 'xgboost'))
            meta_model_type = stacking_cfg.get('meta_model_type', 'xgboost')
            meta_train, meta_test = _build_meta_features_from_views(
                view_map=view_map,
                model_type=base_model_type,
                config=config,
                y_train=y_train,
                meta_cv_splits=stacking_cfg.get('meta_cv_splits', 5),
            )
            model = _fit_named_model(meta_train, y_train, config, model_type=meta_model_type)
            fold_result = evaluate(model, meta_test, y_test, labels=list(LABEL_MAP.values()))
            fold_result['fold'] = fold_id
            fold_result['base_model_type'] = base_model_type
            fold_result['meta_model_type'] = meta_model_type
            fold_metrics.append(fold_result)
            continue

        X_train, X_test = _build_feature_matrices(
            X_rna_train=X_rna_train,
            X_rna_test=X_rna_test,
            X_meth_train=X_meth_train,
            X_meth_test=X_meth_test,
            config=config,
            use_rna=use_rna,
            use_meth=use_meth,
            use_feature_selection=use_feature_selection,
            exp_type=exp_type,
            allow_mofa_full_fit=False,
        )

        model = train_model(X_train, y_train, config)
        fold_result = evaluate(model, X_test, y_test, labels=list(LABEL_MAP.values()))
        fold_result['fold'] = fold_id
        fold_metrics.append(fold_result)

    summary = summarize_cv_metrics(fold_metrics)
    summary['mode'] = 'cv'
    summary['exp'] = exp_type
    summary['fold_count'] = len(fold_metrics)
    summary['requested_folds'] = n_splits
    summary['effective_folds'] = effective_splits
    return summary


def _preprocess_modalities(X_rna_train, X_rna_test, X_meth_train, X_meth_test, config, use_rna, use_meth, use_feature_selection):
    rna_features, meth_features = _select_features(
        rna_train=X_rna_train if use_rna else None,
        meth_train=X_meth_train if use_meth else None,
        config=config,
        use_rna=use_rna,
        use_meth=use_meth,
        use_feature_selection=use_feature_selection,
    )

    views = {}
    if use_rna:
        rna_train_processed = _preprocess_view(
            preprocessor_cls=RNAPreprocessor,
            selected_features=rna_features,
            fit_data=X_rna_train,
            transform_data=X_rna_train,
        )
        rna_test_processed = _preprocess_view(
            preprocessor_cls=RNAPreprocessor,
            selected_features=rna_features,
            fit_data=X_rna_train,
            transform_data=X_rna_test,
        )
        views['rna'] = (rna_train_processed, rna_test_processed)

    if use_meth:
        meth_train_processed = _preprocess_view(
            preprocessor_cls=MethPreprocessor,
            selected_features=meth_features,
            fit_data=X_meth_train,
            transform_data=X_meth_train,
        )
        meth_test_processed = _preprocess_view(
            preprocessor_cls=MethPreprocessor,
            selected_features=meth_features,
            fit_data=X_meth_train,
            transform_data=X_meth_test,
        )
        views['meth'] = (meth_train_processed, meth_test_processed)

    return views


def _predict_proba_or_transform(model, X, target_labels=None):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if target_labels is not None and hasattr(model, 'classes_'):
            target_idx = {label: i for i, label in enumerate(target_labels)}
            aligned = np.zeros((proba.shape[0], len(target_labels)))
            for col_idx, class_label in enumerate(model.classes_):
                if class_label in target_idx:
                    aligned[:, target_idx[class_label]] = proba[:, col_idx]
            return aligned
        return proba
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        if target_labels is not None:
            if hasattr(model, 'classes_'):
                classes = list(model.classes_)
            else:
                classes = list(range(proba.shape[1]))
            target_idx = {label: i for i, label in enumerate(target_labels)}
            aligned = np.zeros((proba.shape[0], len(target_labels)))
            for col_idx, class_label in enumerate(classes):
                if class_label in target_idx:
                    aligned[:, target_idx[class_label]] = proba[:, col_idx]
            return aligned
        return proba
    raise ValueError('Model does not support probability-like outputs')


def _fit_named_model(X, y, config, model_type=None):
    config_copy = copy.deepcopy(config)
    if model_type is not None:
        config_copy.setdefault('model', {})['type'] = model_type
    return train_classifier(X, y, config_copy, model_type=model_type)


def _build_meta_features_from_views(view_map, model_type, config, y_train, meta_cv_splits=5):
    class_counts = y_train.value_counts()
    inner_splits = min(meta_cv_splits, class_counts.min())
    if inner_splits < 2:
        raise ValueError('Not enough samples per class to build stacking meta-features')

    target_labels = sorted(y_train.unique().tolist())

    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=config['model']['random_state'])
    meta_train_parts = []

    for view_name in sorted(view_map.keys()):
        X_view_train, _ = view_map[view_name]
        oof_proba = np.zeros((X_view_train.shape[0], len(target_labels)))
        for inner_train_idx, inner_val_idx in splitter.split(X_view_train, y_train):
            base_model = _fit_named_model(X_view_train[inner_train_idx], y_train.iloc[inner_train_idx], config, model_type=model_type)
            oof_proba[inner_val_idx] = _predict_proba_or_transform(
                base_model,
                X_view_train[inner_val_idx],
                target_labels=target_labels,
            )
        meta_train_parts.append(oof_proba)

    meta_train = np.concatenate(meta_train_parts, axis=1)

    meta_test_parts = []
    for view_name in sorted(view_map.keys()):
        X_view_train, X_view_test = view_map[view_name]
        base_model = _fit_named_model(X_view_train, y_train, config, model_type=model_type)
        meta_test_parts.append(
            _predict_proba_or_transform(base_model, X_view_test, target_labels=target_labels)
        )

    meta_test = np.concatenate(meta_test_parts, axis=1)
    return meta_train, meta_test


def _run_stacking_holdout(X_rna_train, X_rna_test, X_meth_train, X_meth_test, y_train, y_test, config, use_rna, use_meth, use_feature_selection):
    view_map = _preprocess_modalities(
        X_rna_train=X_rna_train,
        X_rna_test=X_rna_test,
        X_meth_train=X_meth_train,
        X_meth_test=X_meth_test,
        config=config,
        use_rna=use_rna,
        use_meth=use_meth,
        use_feature_selection=use_feature_selection,
    )

    stacking_cfg = config.get('stacking', {})
    base_model_type = stacking_cfg.get('base_model_type', config.get('model', {}).get('type', 'xgboost'))
    meta_model_type = stacking_cfg.get('meta_model_type', 'xgboost')

    meta_train, meta_test = _build_meta_features_from_views(
        view_map=view_map,
        model_type=base_model_type,
        config=config,
        y_train=y_train,
        meta_cv_splits=stacking_cfg.get('meta_cv_splits', 5),
    )

    meta_model = _fit_named_model(meta_train, y_train, config, model_type=meta_model_type)
    metrics = evaluate(meta_model, meta_test, y_test, labels=list(LABEL_MAP.values()))
    metrics['mode'] = 'holdout'
    metrics['exp'] = 'stacking'
    metrics['base_model_type'] = base_model_type
    metrics['meta_model_type'] = meta_model_type
    return metrics