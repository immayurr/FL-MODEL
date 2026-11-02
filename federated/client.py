"""
federated/client.py
────────────────────
FederatedClient — manages local training, validation, and parameter exchange
for a single MPI worker process.

No MPI logic lives here.  All send/recv calls stay in mpi_train.py so
this class is independently testable without spawning MPI processes.

Key changes from the original
──────────────────────────────
- Constructor accepts a config namespace instead of individual arguments.
  Changing any hyperparameter in config.yaml propagates automatically.
- train() returns per-epoch losses so the caller can log or plot them.
- evaluate() returns a dict {accuracy, num_samples} — num_samples is
  needed for weighted FedAvg aggregation on the server side.
- get/set_parameters() explicitly handle device placement so CPU-trained
  params are safe to send over MPI and GPU-trained params are loaded
  correctly on the receiving side.
- Logs are written as JSON lines (one dict per round) to
  <log_dir>/client_N.jsonl for easy post-hoc analysis.
- save_model() takes an explicit path argument so the caller decides
  naming and location.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from data.dataset import build_loaders
from models.cnn_model import build_model


class FederatedClient:

    def __init__(self, client_id: int, cfg):
        self.client_id = client_id
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        # Model, loss, optimiser — all from config
        self.model: nn.Module = build_model(
            num_classes=cfg.model.num_classes,
            input_size=cfg.model.input_size,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
        )

        # Data loaders
        self.train_loader, self.val_loader = build_loaders(
            client_id=client_id,
            cfg=cfg,
        )

        Path(cfg.results.log_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.results.model_dir).mkdir(parents=True, exist_ok=True)

        self._log(
            f"Initialised | "
            f"train={len(self.train_loader.dataset)} samples | "
            f"val={len(self.val_loader.dataset)} samples"
        )

    # ── Logging ──────────────────────────────────────────────────────── #

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Client {self.client_id} | {msg}", flush=True)

    def _write_jsonl(self, entry: dict) -> None:
        path = Path(self.cfg.results.log_dir) / f"client_{self.client_id}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Local training ───────────────────────────────────────────────── #

    def train(self, num_epochs: int, round_idx: int) -> List[float]:
        """
        Run local SGD for `num_epochs` on this client's training data.

        Returns
        -------
        List of per-epoch average losses (length = num_epochs).
        """
        self.model.train()
        epoch_losses: List[float] = []

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(self.train_loader))
            epoch_losses.append(avg_loss)
            self._log(
                f"Round {round_idx} | Epoch {epoch}/{num_epochs} | "
                f"avg_loss={avg_loss:.4f}"
            )

        return epoch_losses

    # ── Local validation ─────────────────────────────────────────────── #

    @torch.no_grad()
    def evaluate(self, round_idx: int) -> Dict[str, float | int]:
        """
        Evaluate on the client's local validation split.

        Returns
        -------
        dict with keys:
            accuracy    — float  (correct / total)
            num_samples — int    (total val samples; used for weighted avg)
        """
        self.model.eval()
        correct = total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0.0
        self._log(
            f"Round {round_idx} | val_accuracy={acc:.4f} "
            f"({correct}/{total})"
        )

        self._write_jsonl({
            "round": round_idx,
            "val_accuracy": round(acc, 4),
            "num_samples": total,
        })
        return {"accuracy": acc, "num_samples": total}

    # ── Parameter exchange ───────────────────────────────────────────── #

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return model weights as CPU tensors.
        Always CPU — safe to serialise over MPI regardless of training device.
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Load a state dict received from the server.
        Moves each tensor to the client's device.
        """
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in params.items()}
        )

    # ── Persistence ──────────────────────────────────────────────────── #

    def save_model(self, round_idx: int | None = None) -> Path:
        """Save final local model weights."""
        fname = (
            f"client_{self.client_id}_round{round_idx:02d}.pt"
            if round_idx is not None
            else f"client_{self.client_id}_final.pt"
        )
        path = Path(self.cfg.results.model_dir) / fname
        torch.save(self.model.state_dict(), path)
        self._log(f"Model saved → {path}")
        return path
