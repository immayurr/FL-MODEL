"""
federated/server.py
────────────────────
FederatedServer — owns the global model, FedAvg aggregation, global
evaluation on the held-out test set, and round-level logging.

No MPI logic lives here.  All send/recv calls stay in mpi_train.py so
this class is independently testable without spawning MPI processes.

Responsibilities
────────────────
- get_global_params()   — expose current global weights (CPU tensors)
- set_global_params()   — load an incoming state dict into the model
- aggregate()           — weighted FedAvg over client updates
- evaluate_global()     — run inference on a DataLoader, return full stats
- save_global_model()   — persist weights with optional round suffix
- log_round()           — append a structured JSON line per round
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.cnn_model import build_model


class FederatedServer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        self.global_model: nn.Module = build_model(
            num_classes=cfg.model.num_classes,
            input_size=cfg.model.input_size,
        ).to(self.device)

        Path(cfg.results.model_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.results.log_dir).mkdir(parents=True, exist_ok=True)

        self._ts = datetime.now().strftime("%H:%M:%S")

    # ── Logging helper ───────────────────────────────────────────────── #

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Server | {msg}", flush=True)

    # ── Parameter access ─────────────────────────────────────────────── #

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Return the current global model state dict as CPU tensors."""
        return {k: v.cpu() for k, v in self.global_model.state_dict().items()}

    def set_global_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load a state dict into the global model (moves tensors to device)."""
        self.global_model.load_state_dict(
            {k: v.to(self.device) for k, v in params.items()}
        )

    # ── FedAvg aggregation ───────────────────────────────────────────── #

    def aggregate(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted average of client state dicts (FedAvg).

        Each parameter tensor is averaged across clients weighted by the
        number of training samples that client holds.  This is the only
        implementation of FedAvg in the project — there is no duplicate.

        Parameters
        ----------
        client_params_list : List of state dicts received from clients.
        client_sizes       : Number of training samples per client,
                             used as aggregation weights.

        Returns
        -------
        The new global parameters dict (also updates self.global_model).
        """
        total_samples = sum(client_sizes)

        aggregated: Dict[str, torch.Tensor] = {}
        for key in client_params_list[0]:
            aggregated[key] = sum(
                params[key].float() * (size / total_samples)
                for params, size in zip(client_params_list, client_sizes)
            )

        self.set_global_params(aggregated)
        self._log(
            f"FedAvg complete — "
            f"{len(client_sizes)} clients | "
            f"total samples = {total_samples}"
        )
        return self.get_global_params()

    # ── Global evaluation ────────────────────────────────────────────── #

    @torch.no_grad()
    def evaluate_global(self, test_loader: DataLoader) -> dict:
        """
        Evaluate the current global model on a DataLoader.

        Returns a dict with:
            accuracy    — float
            num_samples — int
            preds       — List[int]  (predicted class indices)
            labels      — List[int]  (ground-truth class indices)
            probs       — np.ndarray shape (N, C)  (softmax probabilities)
        """
        self.global_model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs_list: List[np.ndarray] = []

        for images, labels in test_loader:
            images = images.to(self.device)
            logits = self.global_model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs_list.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

        all_probs = np.vstack(all_probs_list)
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        acc = correct / len(all_labels) if all_labels else 0.0

        return {
            "accuracy": acc,
            "num_samples": len(all_labels),
            "preds": all_preds,
            "labels": all_labels,
            "probs": all_probs,
        }

    # ── Persistence ──────────────────────────────────────────────────── #

    def save_global_model(self, round_idx: int | None = None) -> Path:
        """
        Save global model weights.
        round_idx=None  → global_model.pt   (final, after all rounds)
        round_idx=N     → global_model_roundNN.pt  (checkpoint per round)
        """
        fname = (
            f"global_model_round{round_idx:02d}.pt"
            if round_idx is not None
            else "global_model.pt"
        )
        path = Path(self.cfg.results.model_dir) / fname
        torch.save(self.global_model.state_dict(), path)
        self._log(f"Model saved → {path}")
        return path

    def log_round(
        self,
        round_idx: int,
        global_acc: float,
        client_accuracies: List[float],
        client_sizes: List[int],
        weighted_client_acc: float,
    ) -> None:
        """Append one JSON line per round to <log_dir>/global_rounds.jsonl."""
        entry = {
            "round": round_idx,
            "global_test_accuracy": round(global_acc, 4),
            "weighted_client_val_accuracy": round(weighted_client_acc, 4),
            "client_val_accuracies": [round(a, 4) for a in client_accuracies],
            "client_sizes": client_sizes,
        }
        log_path = Path(self.cfg.results.log_dir) / "global_rounds.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
