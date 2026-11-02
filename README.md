# Federated Learning — Brain Tumor MRI Classification

Simulates federated learning across multiple clients using MPI.  
4-class classification: **glioma · meningioma · pituitary · no_tumor**

---

## Project structure

```
fl_project/
├── config.yaml           ← all constants live here
├── prepare_data.py       ← run once before training
├── mpi_train.py          ← main training entry point
├── predict.py            ← inference on a single image
├── data/
│   ├── dataset.py        ← MRIDataset + DataLoader factories
│   └── raw/brain_tumor/  ← place downloaded dataset here
│       ├── train/
│       └── test/
├── models/
│   └── cnn_model.py      ← BrainTumorCNN
├── federated/
│   ├── client.py         ← FederatedClient
│   └── server.py         ← FederatedServer (FedAvg)
└── utils/
    ├── config.py          ← YAML loader
    ├── seed.py            ← reproducibility seeds
    ├── metrics.py         ← accuracy, F1, ROC-AUC
    └── visualize.py       ← four plot functions
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Workflow

### 1. Place data

[Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

Download the Brain Tumor MRI dataset from link above and extract it so the layout is:

```
data/raw/brain_tumor/
    train/  glioma/  meningioma/  pituitary/  no_tumor/
    test/   glioma/  meningioma/  pituitary/  no_tumor/
```

### 2. Prepare partitions (run once)

```bash
python prepare_data.py
```

This creates:
- `data/raw/brain_tumor/val/`  — global validation split (copied from train)
- `data/partitions/client_{1,2,3}/train|val/`  — per-client data splits
- `data/partitions/manifest.json`  — records all counts and the seed

Re-running is a no-op if `manifest.json` already exists.

### 3. Train

```bash
chcp 65001
mpiexec -n 3 python mpi_train.py
```

4 processes = 1 server (rank 0) + 3 clients (ranks 1-3).

To change the number of clients, update `data.num_clients` in `config.yaml`
and adjust the `-n` argument accordingly: `-n (num_clients + 1)`.

### 4. Predict

```bash
python predict.py --model results/models/global_model.pt --image data/raw/brain_tumor/train/glioma/0011.jpg
```

---

## Outputs (written to `results/`)

| Path | Contents |
|---|---|
| `results/models/global_model.pt` | Final global model weights |
| `results/models/global_model_roundNN.pt` | Per-round checkpoints |
| `results/models/client_N_final.pt` | Final local client weights |
| `results/logs/global_rounds.jsonl` | Per-round accuracy (JSON lines) |
| `results/logs/client_N.jsonl` | Per-client per-round loss + accuracy |
| `results/logs/final_classification_report.json` | Precision/recall/F1 |
| `results/plots/round_accuracy.png` | Client + global accuracy curves |
| `results/plots/confusion_matrix.png` | Normalised confusion matrix |
| `results/plots/roc_curves.png` | One-vs-rest ROC per class |
| `results/plots/client_distribution.png` | Class counts per client |

---

## Configuration

All tunable parameters are in `config.yaml`.  
**Nothing is hardcoded** in any other file.

Key parameters:

| Key | Default | Meaning |
|---|---|---|
| `data.num_clients` | 3 | Number of federated clients |
| `data.val_split_ratio` | 0.2 | Fraction of train used for validation |
| `data.seed` | 42 | Random seed for partitioning + training |
| `training.num_rounds` | 10 | Federated communication rounds |
| `training.local_epochs` | 2 | Local epochs per client per round |
| `training.batch_size` | 32 | Batch size for all loaders |
| `training.learning_rate` | 0.001 | Adam learning rate |
| `training.device` | cpu | `cpu` or `cuda` |

---

## Federated protocol (MPI)

Each round follows a symmetric, deadlock-free structure:

```
Server                          Client
──────────────────────────────────────
isend(global_params) ─────────► recv(global_params)
                                set_parameters()
                                train(local_epochs)
                                evaluate()
recv(update)         ◄───────── send({params, accuracy, n_samples})
aggregate()  ← FedAvg
evaluate_global()  ← held-out test
```

No pre-loop receive on the client side.  
Server sends exactly `num_rounds` times; client receives exactly `num_rounds` times.