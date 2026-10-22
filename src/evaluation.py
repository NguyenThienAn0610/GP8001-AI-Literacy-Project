"""Shared classification metrics and fold summaries (no training side effects)."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score


def metric_values(y_true, predictions, probabilities, labels):
    values = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
    }
    for k in (2, 3, 4):
        values[f"top_{k}_accuracy"] = float(
            top_k_accuracy_score(y_true, probabilities, k=k, labels=labels)
        )
    return values


def neural_scores(y, p):
    pred = p.argmax(axis=1)
    labels = np.arange(p.shape[1])
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            'top_3_accuracy': float(top_k_accuracy_score(y, p, k=3, labels=labels)),
            'log_loss': float(log_loss(y, p, labels=labels))}


def validation_key(values):
    return (values['accuracy'], values['macro_f1'], -values['log_loss'])


def summarize_folds(rows, metrics, prefix=""):
    """Return mean and population standard deviation for each fold metric."""
    result = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        result[f"{prefix}{metric}_mean"] = float(np.mean(values))
        result[f"{prefix}{metric}_std"] = float(np.std(values))
    return result
