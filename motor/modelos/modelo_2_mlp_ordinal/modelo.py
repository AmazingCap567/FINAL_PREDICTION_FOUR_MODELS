# -*- coding: utf-8 -*-
"""
training.py
===========
Entrenamiento del MLP ordinal multi-task: preparacion de tensores, calculo
de pesos de clase (a partir de TRAIN), bucle de entrenamiento con early
stopping por Weighted Kappa promedio de validacion, y utilidades de
prediccion.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ... import config
from ...utils import log
from .arquitectura import MultiTaskOrdinalMLP, MaskedMultiTaskOrdinalLoss, decode_predictions
from ...evaluacion.metrics import compute_metrics


class SupervisedTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, entity_type_idx: np.ndarray, targets: dict[str, np.ndarray]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.entity_type_idx = torch.tensor(entity_type_idx, dtype=torch.long)
        self.targets = {ecr: torch.tensor(np.nan_to_num(t, nan=-1.0), dtype=torch.float32)
                         for ecr, t in targets.items()}
        self.masks = {ecr: torch.tensor(~np.isnan(t), dtype=torch.bool) for ecr, t in targets.items()}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        y = {ecr: self.targets[ecr][idx] for ecr in config.ECR_LIST}
        m = {ecr: self.masks[ecr][idx] for ecr in config.ECR_LIST}
        return self.X[idx], self.entity_type_idx[idx], y, m


def _collate(batch):
    X = torch.stack([b[0] for b in batch])
    etype = torch.stack([b[1] for b in batch])
    y = {ecr: torch.stack([b[2][ecr] for b in batch]) for ecr in config.ECR_LIST}
    m = {ecr: torch.stack([b[3][ecr] for b in batch]) for ecr in config.ECR_LIST}
    return X, etype, y, m


def compute_class_weights(train_targets: dict[str, np.ndarray], k_per_ecr: dict[str, int]) -> dict[str, torch.Tensor]:
    """Peso inverso a la frecuencia de cada clase local, calculado solo con TRAIN."""
    weights = {}
    for ecr in config.ECR_LIST:
        y = train_targets[ecr]
        valid = y[~np.isnan(y)].astype(int)
        k = k_per_ecr[ecr]
        counts = np.array([max((valid == c).sum(), 1) for c in range(k)], dtype=float)
        w = counts.sum() / (k * counts)
        weights[ecr] = torch.tensor(w, dtype=torch.float32)
    return weights


def train_model(
    X_train: np.ndarray, etype_train: np.ndarray, y_train: dict[str, np.ndarray],
    X_val: np.ndarray, etype_val: np.ndarray, y_val: dict[str, np.ndarray],
    n_entity_types: int, k_per_ecr: dict[str, int],
) -> tuple[MultiTaskOrdinalMLP, list[dict]]:

    cfg = config.MODEL_CONFIG
    torch.manual_seed(config.SEED)

    train_ds = SupervisedTensorDataset(X_train, etype_train, y_train)
    val_ds = SupervisedTensorDataset(X_val, etype_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=_collate)

    model = MultiTaskOrdinalMLP(n_features=X_train.shape[1], n_entity_types=n_entity_types, k_per_ecr=k_per_ecr)
    class_weights = compute_class_weights(y_train, k_per_ecr)
    loss_fn = MaskedMultiTaskOrdinalLoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_score = -np.inf
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        epoch_losses = []
        for X_b, etype_b, y_b, m_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b, etype_b)
            loss, _ = loss_fn(outputs, y_b, m_b)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_ds.X, val_ds.entity_type_idx)
            val_loss, per_ecr_val_loss = loss_fn(val_outputs, val_ds.targets, val_ds.masks)

            kappas = []
            for ecr in config.ECR_LIST:
                mask = val_ds.masks[ecr].numpy()
                if mask.sum() < 5:
                    continue
                preds = decode_predictions(val_outputs[ecr])
                y_true = val_ds.targets[ecr].numpy().astype(int)
                m = compute_metrics(y_true[mask], preds[mask])
                if not np.isnan(m["weighted_kappa"]):
                    kappas.append(m["weighted_kappa"])
            mean_kappa = float(np.mean(kappas)) if kappas else -1.0

        history.append({
            "epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
            "val_loss": float(val_loss.item()), "val_mean_kappa": mean_kappa,
        })

        improved = mean_kappa > best_score + cfg["min_delta"]
        if improved:
            best_score = mean_kappa
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            log(f"  Epoch {epoch:03d} | train_loss={np.mean(epoch_losses):.4f} "
                f"| val_loss={val_loss.item():.4f} | val_mean_kappa={mean_kappa:.4f}"
                f"{'  <- mejor' if improved else ''}")

        if epochs_no_improve >= cfg["patience"]:
            log(f"  Early stopping en epoch {epoch} (sin mejora en {cfg['patience']} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict(model: MultiTaskOrdinalMLP, X: np.ndarray, etype: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        etype_t = torch.tensor(etype, dtype=torch.long)
        outputs = model(X_t, etype_t)
        preds = {ecr: decode_predictions(outputs[ecr]) for ecr in config.ECR_LIST}
    return preds


def decode_absolute_from_delta(
    model: MultiTaskOrdinalMLP, X: np.ndarray, etype: np.ndarray,
    prev_ratings: dict[str, np.ndarray], k_local: dict[str, int], majority_fallback: dict[str, int],
) -> dict[str, np.ndarray]:
    """Reconstruye el rating absoluto: prev_rating + delta_decodificado, acotado a la escala real."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        etype_t = torch.tensor(etype, dtype=torch.long)
        outputs = model(X_t, etype_t)

    abs_preds = {}
    for ecr in config.ECR_LIST:
        delta_class = decode_predictions(outputs[ecr])
        delta_value = delta_class - config.DELTA_OFFSET
        prev = prev_ratings[ecr]
        abs_pred = prev + delta_value
        abs_pred = np.where(np.isnan(prev), majority_fallback[ecr], abs_pred)
        abs_pred = np.clip(abs_pred, 0, k_local[ecr] - 1)
        abs_preds[ecr] = abs_pred.astype(int)
    return abs_preds
