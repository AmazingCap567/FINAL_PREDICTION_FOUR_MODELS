# -*- coding: utf-8 -*-
"""
metrics.py
==========
Metricas de evaluacion para clasificacion ordinal, calculadas por ECR y de
forma agregada. La metrica principal NO es Accuracy (poco informativa con
clases desbalanceadas), sino Weighted Cohen's Kappa (penaliza mas los
errores grandes que los pequenos, respetando la naturaleza ordinal del
problema).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, f1_score, balanced_accuracy_score,
    accuracy_score, confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true, y_pred: arrays de enteros (ordinal). Deben tener la misma
    longitud y no contener NaN (filtrar antes de llamar).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) == 0:
        return {k: np.nan for k in (
            "weighted_kappa", "f1_macro", "f1_weighted", "ordinal_mae",
            "accuracy", "accuracy_pm1", "balanced_accuracy", "n",
        )}

    abs_err = np.abs(y_true - y_pred)
    out = {
        "weighted_kappa": cohen_kappa_score(y_true, y_pred, weights="linear") if len(set(y_true)) > 1 else np.nan,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ordinal_mae": float(abs_err.mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_pm1": float((abs_err <= 1).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "n": len(y_true),
    }
    return out


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """results: {nombre_modelo_o_ecr: metrics_dict} -> DataFrame legible."""
    df = pd.DataFrame(results).T
    cols_order = ["n", "accuracy", "accuracy_pm1", "balanced_accuracy",
                  "f1_macro", "f1_weighted", "ordinal_mae", "weighted_kappa"]
    cols = [c for c in cols_order if c in df.columns]
    return df[cols].round(4)


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"real_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
