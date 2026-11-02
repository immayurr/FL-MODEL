"""
mpi_train.py
─────────────
Federated learning orchestrator using MPI (mpi4py).

Process layout
──────────────
  Rank 0  →  server  (FederatedServer)
  Rank 1  →  client 1
  Rank 2  →  client 2
  ...
  Rank N  →  client N

Run with
─────────
  mpirun -n 4 python mpi_train.py            # 1 server + 3 clients
  mpirun -n 4 python mpi_train.py --config config.yaml

Communication protocol (fixed deadlock from original)
──────────────────────────────────────────────────────
Each round follows an identical, symmetric structure:

  Server                          Client
  ──────────────────────────────────────
  isend(global_params) ─────────► recv(global_params)
                                  set_parameters()
                                  train()
                                  evaluate()
  recv(update)         ◄───────── send(update)
  aggregate()
  evaluate_global()

Key fix: there is NO pre-loop receive on the client side.
Every round starts with the server sending and the client receiving.
Server sends exactly NUM_ROUNDS times; client receives exactly NUM_ROUNDS
times → no off-by-one → no deadlock.

Tags
────
  TAG_PARAMS = 10   server → client  (global model state dict)
  TAG_UPDATE = 20   client → server  (local params + stats)
"""


from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from mpi4py import MPI

from data.dataset import build_test_loader
from federated.client import FederatedClient
from federated.server import FederatedServer
from utils.config import load_config
from utils.metrics import compute_metrics, weighted_average_accuracy
from utils.seed import set_all_seeds
from utils.visualize import (
    plot_client_distribution,
    plot_confusion_matrix,
    plot_round_accuracy,
    plot_roc_curve,
)


import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── MPI setup ────────────────────────────────────────────────────────── #
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

TAG_PARAMS = 10   # server → client: global parameters
TAG_UPDATE = 20   # client → server: local update


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    role = "Server" if RANK == 0 else f"Client {RANK}"
    print(f"[{ts}] {role} | {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════ #
# SERVER                                                               #
# ═══════════════════════════════════════════════════════════════════ #

def run_server(cfg) -> None:
    server = FederatedServer(cfg)
    test_loader = build_test_loader(cfg)
    num_clients = SIZE - 1
    num_rounds = cfg.training.num_rounds

    # Accumulators for post-training plots
    global_accs: List[float] = []
    client_accs: Dict[int, List[float]] = {r: [] for r in range(1, SIZE)}

    _log(f"Ready — {num_clients} client(s), {num_rounds} rounds")

    for round_idx in range(1, num_rounds + 1):
        _log(f"{'-'*10} Round {round_idx}/{num_rounds} {'-'*10}")

        # ── 1. Send global params to all clients (non-blocking) ───────
        global_params = server.get_global_params()
        send_reqs = [
            COMM.isend(global_params, dest=r, tag=TAG_PARAMS)
            for r in range(1, SIZE)
        ]
        MPI.Request.Waitall(send_reqs)
        _log(f"Global params broadcast to {num_clients} client(s).")

        # ── 2. Receive client updates ─────────────────────────────────
        updates: List[dict] = [COMM.recv(source=r, tag=TAG_UPDATE)
                               for r in range(1, SIZE)]

        client_params_list = [u["params"]      for u in updates]
        client_sizes       = [u["num_samples"] for u in updates]
        client_val_accs    = [u["val_accuracy"] for u in updates]

        for r, u in zip(range(1, SIZE), updates):
            client_accs[r].append(u["val_accuracy"])
            _log(
                f"  ← Client {r} | "
                f"samples={u['num_samples']} | "
                f"val_acc={u['val_accuracy']:.4f}"
            )

        # ── 3. FedAvg aggregation ──────────────────────────────────────
        server.aggregate(client_params_list, client_sizes)

        # ── 4. Evaluate aggregated global model on held-out test set ───
        result = server.evaluate_global(test_loader)
        global_accs.append(result["accuracy"])
        _log(
            f"Global test accuracy = {result['accuracy']:.4f} "
            f"({result['num_samples']} samples)"
        )

        # ── 5. Logging ────────────────────────────────────────────────
        w_acc = weighted_average_accuracy(client_val_accs, client_sizes)
        server.log_round(
            round_idx=round_idx,
            global_acc=result["accuracy"],
            client_accuracies=client_val_accs,
            client_sizes=client_sizes,
            weighted_client_acc=w_acc,
        )

        # ── 6. Save per-round checkpoint ───────────────────────────────
        server.save_global_model(round_idx=round_idx)

    # ═══════════════════════════════════════════════════════════════ #
    # Post-training: final model + full metrics + plots               #
    # ═══════════════════════════════════════════════════════════════ #

    _log("All rounds complete.  Generating final outputs...")

    # Save final model (clean filename, no round suffix)
    server.save_global_model()

    # Run full evaluation for metrics and plots
    final = server.evaluate_global(test_loader)
    metrics = compute_metrics(
        all_preds=final["preds"],
        all_labels=final["labels"],
        all_probs=final["probs"],
        class_names=cfg.data.classes,
    )

    _log(f"Final test accuracy  : {metrics['accuracy']:.4f}")
    _log(f"Final macro ROC-AUC  : {metrics['roc_auc_macro']:.4f}")

    # Pretty-print per-class metrics to console
    report = metrics["classification_report"]
    _log("Per-class metrics:")
    for cls in cfg.data.classes:
        r = report.get(cls, {})
        _log(
            f"  {cls:<14s}  "
            f"precision={r.get('precision', 0):.3f}  "
            f"recall={r.get('recall', 0):.3f}  "
            f"f1={r.get('f1-score', 0):.3f}  "
            f"support={r.get('support', 0)}"
        )

    # Save classification report as JSON
    report_path = Path(cfg.results.log_dir) / "final_classification_report.json"
    serialisable = {
        k: (
            {sk: float(sv) if not isinstance(sv, str) else sv
             for sk, sv in v.items()}
            if isinstance(v, dict) else float(v)
        )
        for k, v in report.items()
    }
    with open(report_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    _log(f"Classification report → {report_path}")

    # Plots
    plot_dir = Path(cfg.results.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_round_accuracy(
        global_accs=global_accs,
        client_accs=client_accs,
        save_path=plot_dir / "round_accuracy.png",
    )
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=cfg.data.classes,
        save_path=plot_dir / "confusion_matrix.png",
    )
    plot_roc_curve(
        all_probs=final["probs"],
        all_labels=final["labels"],
        class_names=cfg.data.classes,
        save_path=plot_dir / "roc_curves.png",
    )

    manifest_path = Path(cfg.data.partitions_dir) / "manifest.json"
    if manifest_path.exists():
        plot_client_distribution(
            manifest_path=manifest_path,
            save_path=plot_dir / "client_distribution.png",
        )

    _log("Done.  All outputs written to results/")


# ═══════════════════════════════════════════════════════════════════ #
# CLIENT                                                               #
# ═══════════════════════════════════════════════════════════════════ #

def run_client(cfg) -> None:
    client = FederatedClient(client_id=RANK, cfg=cfg)
    num_rounds = cfg.training.num_rounds

    for round_idx in range(1, num_rounds + 1):

        # ── 1. Receive current global params ──────────────────────────
        global_params = COMM.recv(source=0, tag=TAG_PARAMS)
        client.set_parameters(global_params)
        _log(f"Round {round_idx} | Global model received.")

        # ── 2. Local training ──────────────────────────────────────────
        losses = client.train(
            num_epochs=cfg.training.local_epochs,
            round_idx=round_idx,
        )
        mean_loss = sum(losses) / len(losses)
        _log(f"Round {round_idx} | Training done | mean_loss={mean_loss:.4f}")

        # ── 3. Local validation ────────────────────────────────────────
        eval_result = client.evaluate(round_idx=round_idx)

        # ── 4. Send update to server ───────────────────────────────────
        update = {
            "params":       client.get_parameters(),
            "num_samples":  eval_result["num_samples"],
            "val_accuracy": eval_result["accuracy"],
        }
        COMM.send(update, dest=0, tag=TAG_UPDATE)
        _log(f"Round {round_idx} | Update sent.")

    # Save final local model after all rounds
    client.save_model()
    _log("All rounds complete.  Final model saved.")


# ═══════════════════════════════════════════════════════════════════ #
# Entry point                                                          #
# ═══════════════════════════════════════════════════════════════════ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated learning for brain tumor MRI classification."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.data.seed)

    # Validate MPI setup
    if SIZE < 2:
        raise RuntimeError(
            "At least 2 MPI processes required (1 server + N clients).\n"
            f"Run with:  mpirun -n {cfg.data.num_clients + 1} python mpi_train.py"
        )

    num_clients_expected = cfg.data.num_clients
    if SIZE - 1 != num_clients_expected:
        _log(
            f"Warning: config expects {num_clients_expected} client(s) but "
            f"{SIZE - 1} MPI worker(s) detected.  Proceeding with {SIZE - 1}."
        )

    # Create result dirs on every rank (all ranks may need to write logs)
    for d in [cfg.results.log_dir, cfg.results.model_dir, cfg.results.plot_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if RANK == 0:
        run_server(cfg)
    else:
        run_client(cfg)


if __name__ == "__main__":
    main()
