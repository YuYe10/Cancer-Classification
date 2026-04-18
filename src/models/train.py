from sklearn.svm import SVC


def train_svm(X_train, y_train, config):
    """
    Train SVM on already-split training data.
    
    Args:
        X_train: Training features (already split, already preprocessed)
        y_train: Training labels (already split)
        config: Config dict (used for SVM hyperparameters if needed)
    
    Returns:
        model: Trained SVM model
    """
    model_cfg = config.get('model', {})
    model = SVC(
        C=model_cfg.get('svm_c', 1.0),
        kernel=model_cfg.get('svm_kernel', 'rbf'),
        probability=True,
        random_state=model_cfg.get('random_state', 42),
    )
    model.fit(X_train, y_train)
    return model