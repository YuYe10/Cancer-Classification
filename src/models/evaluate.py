import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(model, X_test, y_test, labels=None):
    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist() if labels is not None else confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
    }

    return metrics


def summarize_cv_metrics(fold_metrics):
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]

    summary = {}
    lines = []
    for name in metric_names:
        values = np.array([metrics[name] for metrics in fold_metrics], dtype=float)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))
        lines.append(f"{name}: {values.mean():.4f} ± {values.std(ddof=0):.4f}")

    summary["fold_count"] = len(fold_metrics)
    summary["fold_metrics"] = fold_metrics
    summary["report"] = "\n".join(lines)

    return summary