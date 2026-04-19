from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from xgboost import XGBClassifier


class XGBRemappedModel:
    def __init__(self, model, classes):
        self.model = model
        self.classes_ = np.array(classes)

    def predict(self, X):
        pred_idx = self.model.predict(X).astype(int)
        return self.classes_[pred_idx]

    def predict_proba(self, X):
        return self.model.predict_proba(X)


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


def train_xgboost(X_train, y_train, config):
    model_cfg = config.get('model', {})
    y_arr = np.asarray(y_train)
    classes = np.sort(np.unique(y_arr))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_encoded = np.array([class_to_idx[label] for label in y_arr], dtype=int)

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(classes),
        n_estimators=model_cfg.get('xgb_n_estimators', 300),
        max_depth=model_cfg.get('xgb_max_depth', 4),
        learning_rate=model_cfg.get('xgb_learning_rate', 0.05),
        subsample=model_cfg.get('xgb_subsample', 0.9),
        colsample_bytree=model_cfg.get('xgb_colsample_bytree', 0.9),
        reg_lambda=model_cfg.get('xgb_reg_lambda', 1.0),
        min_child_weight=model_cfg.get('xgb_min_child_weight', 1.0),
        random_state=model_cfg.get('random_state', 42),
        n_jobs=model_cfg.get('n_jobs', -1),
        eval_metric='mlogloss',
        tree_method=model_cfg.get('xgb_tree_method', 'hist'),
    )
    model.fit(X_train, y_encoded)
    return XGBRemappedModel(model, classes)


def train_random_forest(X_train, y_train, config):
    model_cfg = config.get('model', {})
    model = RandomForestClassifier(
        n_estimators=model_cfg.get('rf_n_estimators', 500),
        max_depth=model_cfg.get('rf_max_depth', None),
        min_samples_split=model_cfg.get('rf_min_samples_split', 2),
        min_samples_leaf=model_cfg.get('rf_min_samples_leaf', 1),
        max_features=model_cfg.get('rf_max_features', 'sqrt'),
        random_state=model_cfg.get('random_state', 42),
        n_jobs=model_cfg.get('n_jobs', -1),
        class_weight=model_cfg.get('rf_class_weight', None),
    )
    model.fit(X_train, y_train)
    return model


def train_classifier(X_train, y_train, config, model_type=None):
    model_cfg = config.get('model', {})
    resolved_type = (model_type or model_cfg.get('type', 'svm')).lower()

    if resolved_type == 'svm':
        return train_svm(X_train, y_train, config)
    if resolved_type == 'xgboost':
        return train_xgboost(X_train, y_train, config)
    if resolved_type in {'rf', 'random_forest'}:
        return train_random_forest(X_train, y_train, config)

    raise ValueError(f"Unknown model type: {resolved_type}")


def train_model(X_train, y_train, config):
    return train_classifier(X_train, y_train, config)