# -*- coding: utf-8 -*-
"""
explainability.py
==================
Explicabilidad del modelo final, por ECR.

Prioriza:
  1. Permutation importance (model-agnostic, siempre disponible): mide la
     caida en Weighted Kappa al permutar aleatoriamente cada variable, sobre
     el conjunto de VALIDACION/HOLDOUT (nunca sobre TRAIN, para no medir
     memorizacion).
  2. SHAP (si la libreria esta disponible y es estable para este tipo de
     modelo): se intenta con KernelExplainer sobre una funcion de prediccion
     que envuelve el modelo PyTorch. Si falla o no esta instalada, se omite
     silenciosamente y se deja constancia en el reporte de que no se pudo
     calcular.

IMPORTANTE: la importancia reportada es PREDICTIVA, no causal. No debe
interpretarse como que una variable "causa" un cambio de rating.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config
from ..utils import log
from .metrics import compute_metrics
# torch y el modulo del MLP se importan de forma perezosa dentro de las
# funciones que los usan, para que este modulo (y todo lo que lo importe)
# siga siendo utilizable en un entorno sin PyTorch instalado.


def permutation_importance_by_ecr(
    model, X: np.ndarray, etype: np.ndarray, targets: dict[str, np.ndarray],
    feature_names: list[str], n_repeats: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Calcula, para cada ECR, la caida promedio en Weighted Kappa al permutar
    cada variable `n_repeats` veces. Devuelve {ecr: DataFrame(feature, importancia)}.
    """
    import torch
    from ..modelos.modelo_2_mlp_ordinal.arquitectura import decode_predictions

    rng = np.random.RandomState(config.SEED)
    model.eval()

    results = {}
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        etype_t = torch.tensor(etype, dtype=torch.long)
        base_outputs = model(X_t, etype_t)

    for ecr in config.ECR_LIST:
        y = targets[ecr]
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            log(f"Explicabilidad: ECR {ecr} tiene muy pocas observaciones validas ({mask.sum()}), se omite.")
            continue

        y_true = y[mask].astype(int)
        base_preds = decode_predictions(base_outputs[ecr])[mask]
        base_metric = compute_metrics(y_true, base_preds)["weighted_kappa"]
        base_metric = 0.0 if np.isnan(base_metric) else base_metric

        importances = []
        for j, fname in enumerate(feature_names):
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                rng.shuffle(X_perm[:, j])
                with torch.no_grad():
                    X_perm_t = torch.tensor(X_perm, dtype=torch.float32)
                    out_perm = model(X_perm_t, etype_t)
                preds_perm = decode_predictions(out_perm[ecr])[mask]
                m = compute_metrics(y_true, preds_perm)["weighted_kappa"]
                m = 0.0 if np.isnan(m) else m
                drops.append(base_metric - m)
            importances.append(np.mean(drops))

        imp_df = pd.DataFrame({"feature": feature_names, "importancia_permutacion": importances})
        imp_df = imp_df.sort_values("importancia_permutacion", ascending=False).reset_index(drop=True)
        results[ecr] = imp_df
        log(f"Explicabilidad ({ecr}): top variable = {imp_df.iloc[0]['feature']} "
            f"(caida kappa={imp_df.iloc[0]['importancia_permutacion']:.4f})")

    return results


def try_shap_summary(model, X: np.ndarray, etype: np.ndarray, feature_names: list[str],
                      ecr: str, sample_size: int = 30) -> pd.DataFrame | None:
    """
    Intento best-effort de calcular SHAP para una ECR sobre una muestra
    pequena (KernelExplainer es costoso). Si shap no esta disponible o el
    calculo falla/es inestable, devuelve None sin interrumpir el pipeline.
    """
    try:
        import shap
        import torch
    except ImportError:
        log("Libreria 'shap' no disponible; se omite SHAP y se reporta solo permutation importance.", level="WARN")
        return None

    try:
        model.eval()
        n = min(sample_size, X.shape[0])
        idx = np.random.RandomState(config.SEED).choice(X.shape[0], size=n, replace=False)
        X_sample = X[idx]
        etype_sample = etype[idx]

        etype_fixed = torch.tensor(etype_sample, dtype=torch.long)

        def predict_fn(x_num: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x_t = torch.tensor(x_num, dtype=torch.float32)
                out = model(x_t, etype_fixed[: x_num.shape[0]])
                probs = torch.sigmoid(out[ecr])
                # Valor escalar resumen: numero esperado de umbrales superados
                return probs.sum(dim=1).numpy()

        background = X_sample[: max(5, n // 4)]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100, silent=True)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:  # pragma: no cover - best effort, no debe tumbar el pipeline
        log(f"SHAP fallo para {ecr} ({type(e).__name__}: {e}); se omite y se usa solo permutation importance.",
            level="WARN")
        return None
