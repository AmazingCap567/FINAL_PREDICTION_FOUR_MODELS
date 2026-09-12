# -*- coding: utf-8 -*-
"""
ordinal_model.py
================
Red neuronal MLP ordinal multi-task en PyTorch.

Arquitectura:
    [features financieras escaladas] + [embedding de ENTITY_TYPE]
        -> Dense(64) -> ReLU -> Dropout
        -> Dense(32) -> ReLU -> Dropout
        -> Dense(16) -> ReLU
        -> 6 cabezas independientes (una por ECR)

Cada cabeza usa la formulacion de "extended binary classification" para
regresion ordinal (Frank & Hall, 2001): para una escala de K_ecr categorias,
la cabeza produce K_ecr - 1 logits, uno por cada umbral k=0..K-2, que
representan P(rating_local > k). En inferencia, el rating predicho es el
numero de umbrales superados (sum(sigmoid(logit) > 0.5)). Esta formulacion
penaliza de forma creciente los errores mas alejados de la clase real
(equivocarse por 2 umbrales cuesta mas que por 1), cumpliendo el requisito
de que errores grandes sean mas costosos que errores pequenos.

La perdida:
  - Se calcula independientemente por ECR y solo sobre las filas donde esa
    ECR tiene target valido (mascara), nunca se descarta una fila completa
    por faltarle el target de una sola agencia.
  - Admite ponderacion por clase (pesos inversos a la frecuencia en TRAIN).
  - La perdida total es la suma de las perdidas promedio (por fila valida)
    de las 6 ECR.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ... import config


class MultiTaskOrdinalMLP(nn.Module):
    def __init__(self, n_features: int, n_entity_types: int, k_per_ecr: dict[str, int]):
        super().__init__()
        cfg = config.MODEL_CONFIG
        embed_dim = cfg["entity_type_embed_dim"]
        self.entity_embedding = nn.Embedding(n_entity_types, embed_dim)

        input_dim = n_features + embed_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, cfg["hidden_1"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_1"], cfg["hidden_2"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_2"], cfg["hidden_3"]),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            ecr: nn.Linear(cfg["hidden_3"], max(k_per_ecr[ecr] - 1, 1))
            for ecr in config.ECR_LIST
        })
        self.k_per_ecr = k_per_ecr

    def forward(self, x_num: torch.Tensor, entity_type_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.entity_embedding(entity_type_idx)
        x = torch.cat([x_num, emb], dim=1)
        h = self.shared(x)
        return {ecr: head(h) for ecr, head in self.heads.items()}


def ordinal_targets_to_threshold_matrix(y_local: torch.Tensor, n_thresholds: int) -> torch.Tensor:
    """
    y_local: tensor (N,) con el ordinal local (entero, >=0) o NaN para filas
    invalidas (ya deben venir filtradas/mascaradas por separado).
    Devuelve matriz (N, n_thresholds) con 1.0 si y_local > k, 0.0 si no.
    """
    n = y_local.shape[0]
    thresholds = torch.arange(n_thresholds, device=y_local.device).unsqueeze(0)  # (1, T)
    y = y_local.unsqueeze(1)  # (N, 1)
    return (y > thresholds).float()


class MaskedMultiTaskOrdinalLoss(nn.Module):
    """
    class_weights: dict {ecr: tensor(K_ecr,)} con el peso de cada clase local,
    calculado en TRAIN (ver training.compute_class_weights).
    """

    def __init__(self, class_weights: dict[str, torch.Tensor]):
        super().__init__()
        self.class_weights = class_weights
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = 0.0
        per_ecr_loss = {}
        n_active_ecrs = 0

        for ecr in config.ECR_LIST:
            logits = outputs[ecr]              # (N, T)
            y_local = targets[ecr]             # (N,) puede tener valores basura donde mask=0
            mask = masks[ecr]                  # (N,) bool

            if mask.sum() == 0:
                per_ecr_loss[ecr] = float("nan")
                continue

            n_thresholds = logits.shape[1]
            y_thresh = ordinal_targets_to_threshold_matrix(y_local, n_thresholds)  # (N, T)

            raw = self.bce(logits, y_thresh)  # (N, T)

            # Peso por clase real de cada fila (broadcast a las T columnas)
            weights_per_class = self.class_weights[ecr]  # (K,)
            row_weight = weights_per_class[y_local.clamp(min=0).long()]  # (N,)
            row_weight = row_weight * mask.float()

            weighted = raw.mean(dim=1) * row_weight  # promedio de umbrales, ponderado por fila
            valid_weight_sum = row_weight.sum().clamp(min=1e-8)
            ecr_loss = weighted.sum() / valid_weight_sum

            per_ecr_loss[ecr] = float(ecr_loss.detach().cpu().item())
            total_loss = total_loss + ecr_loss
            n_active_ecrs += 1

        if n_active_ecrs == 0:
            # No deberia ocurrir en un batch bien construido, pero se evita division por cero.
            return torch.tensor(0.0, requires_grad=True), per_ecr_loss

        return total_loss / n_active_ecrs, per_ecr_loss


def decode_predictions(logits: torch.Tensor) -> np.ndarray:
    """Decodifica logits (N, T) -> ordinal local predicho (N,) contando umbrales superados."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).sum(dim=1)
    return preds.detach().cpu().numpy().astype(int)
