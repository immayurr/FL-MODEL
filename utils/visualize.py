"""
utils/visualize.py
──────────────────
All plotting lives here. Every function accepts data, saves a file,
and closes the figure. Nothing is shown interactively.

Functions
---------
plot_round_accuracy       — accuracy curves per round (clients + global)
plot_confusion_matrix     — normalised heatmap
plot_client_distribution  — grouped bar chart from manifest.json
plot_roc_curve            — one-vs-rest ROC per class + macro average
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


# ── Shared helper ────────────────────────────────────────────────────── #

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ── Per-round accuracy ───────────────────────────────────────────────── #

def plot_round_accuracy(
    global_accs: List[float],
    client_accs: Dict[int, List[float]],
    save_path: str | Path,
) -> None:
    """
    One line per client (local val accuracy, dashed) plus one solid line
    for the global model on the held-out test set.
    """
    rounds = list(range(1, len(global_accs) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.45, len(client_accs)))
    for (client_id, accs), color in zip(client_accs.items(), colors):
        ax.plot(
            rounds, accs,
            marker="o", linewidth=1.5, markersize=4,
            linestyle="--", color=color,
            label=f"Client {client_id} (local val)",
        )

    ax.plot(
        rounds, global_accs,
        marker="s", linewidth=2.5, markersize=7,
        color="black", label="Global model (held-out test)",
    )

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Federated Learning — Accuracy per Round", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(rounds)
    _save(fig, Path(save_path))


# ── Confusion matrix ─────────────────────────────────────────────────── #

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
    title: str = "Confusion Matrix — Global Model (Held-Out Test)",
) -> None:
    """
    Row-normalised confusion matrix so cell values are recall rates,
    making class-level performance easy to compare regardless of class size.
    """
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=20, ha="right")
    _save(fig, Path(save_path))


# ── Client data distribution ─────────────────────────────────────────── #

def plot_client_distribution(
    manifest_path: str | Path,
    save_path: str | Path,
) -> None:
    """
    Grouped bar chart: per-client training image count per class.
    Loaded from the manifest.json written by prepare_data.py so the
    plot always reflects the actual partition, not assumptions.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    class_names = manifest["classes"]
    clients = sorted(manifest["clients"].keys(), key=int)
    x = np.arange(len(class_names))
    width = 0.8 / len(clients)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(clients)))

    for i, (client_id, color) in enumerate(zip(clients, colors)):
        counts = [
            manifest["clients"][client_id]["train"].get(cls, 0)
            for cls in class_names
        ]
        offset = (i - len(clients) / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=f"Client {client_id}", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylabel("Number of Training Images", fontsize=11)
    ax.set_title("IID Data Distribution Across Clients", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, Path(save_path))


# ── ROC curves ───────────────────────────────────────────────────────── #

def plot_roc_curve(
    all_probs: np.ndarray,
    all_labels: List[int],
    class_names: List[str],
    save_path: str | Path,
) -> None:
    """
    One-vs-rest ROC curve per class plus an interpolated macro-average.
    Requires at least one positive and one negative sample per class.
    """
    n_classes = len(class_names)
    labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.55, n_classes))

    fprs: dict[int, np.ndarray] = {}
    tprs: dict[int, np.ndarray] = {}
    aucs: dict[int, float] = {}

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        try:
            fprs[i], tprs[i], _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            aucs[i] = auc(fprs[i], tprs[i])
            ax.plot(
                fprs[i], tprs[i],
                color=color, lw=1.8,
                label=f"{cls}  (AUC = {aucs[i]:.3f})",
            )
        except ValueError:
            pass  # skip class if only one label present

    # Macro-average by interpolation
    if fprs:
        all_fpr = np.unique(np.concatenate(list(fprs.values())))
        mean_tpr = np.mean(
            [np.interp(all_fpr, fprs[i], tprs[i]) for i in fprs], axis=0
        )
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr, mean_tpr,
            color="black", lw=2.5, linestyle="--",
            label=f"Macro average  (AUC = {macro_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Global Model (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, Path(save_path))
