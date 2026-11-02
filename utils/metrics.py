"""
utils/metrics.py
────────────────
Classification metric helpers used by both the server (global evaluation)
and the training script (final report generation).

All functions are stateless — pass in lists/arrays, get back dicts.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    all_probs: np.ndarray,
    class_names: List[str],
) -> dict:
    """
    Compute a full classification metric suite.

    Parameters
    ----------
    all_preds   : Flat list of predicted class indices.
    all_labels  : Flat list of ground-truth class indices.
    all_probs   : (N, C) softmax probability array — one row per sample.
    class_names : Ordered list of class name strings matching column order.

    Returns
    -------
    dict with keys:
        accuracy                 — overall accuracy (float)
        classification_report    — per-class precision / recall / F1 (dict)
        confusion_matrix         — (C, C) numpy array
        roc_auc_macro            — macro-averaged one-vs-rest AUC (float)
    """
    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        # Can happen if a class has no samples in the test set
        auc = float("nan")

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc_macro": auc,
    }


def weighted_average_accuracy(
    client_accuracies: List[float],
    client_sizes: List[int],
) -> float:
    """
    Compute the dataset-size-weighted average of per-client accuracies.
    Used to report a single representative client-side metric per round.
    """
    total = sum(client_sizes)
    if total == 0:
        return 0.0
    return sum(a * s for a, s in zip(client_accuracies, client_sizes)) / total
