# -*- coding: utf-8 -*-
"""
modelo_1_hist_gradient_boosting
================================
Modelo 1 de 4 del comparativo. Arboles de gradient boosting con soporte
nativo de NaN (no requiere imputacion completa), uno por ECR.

Ventajas: robusto a variables ruidosas/correlacionadas, rapido de
entrenar, buen desempeno de referencia "no lineal" sin ajustar mucho
hiperparametro.
Limitaciones: no modela explicitamente la naturaleza ORDINAL del target
(trata las clases como nominales), por lo que puede predecir "saltos"
poco plausibles entre categorias no adyacentes.

Interfaz comun con los otros 3 modelos (ver motor/entrenamiento/orquestador.py):
    entrenar(X_train, y_train, **kwargs) -> modelo_entrenado
    predecir(modelo_entrenado, X) -> np.ndarray de clases ordinales predichas
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ... import config

NOMBRE_MODELO = "HistGradientBoosting"


# ---------------------------------------------------------------------------
# entrenar
# ---------------------------------------------------------------------------
# Que hace: ajusta un HistGradientBoostingClassifier de sklearn sobre las
# variables ya seleccionadas para una ECR especifica.
# Recibe: X_train (DataFrame de features), y_train (Serie de clase ordinal).
# Devuelve: el modelo entrenado (objeto sklearn, ya tiene .predict()).
# Nota metodologica: max_depth=4 y max_iter=250 se mantienen conservadores
# a proposito dado el tamano de muestra pequeno (decenas/pocos cientos de
# filas por tipo de entidad); valores mas altos sobreajustan con estos
# volumenes de datos.
# ---------------------------------------------------------------------------
def entrenar(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> HistGradientBoostingClassifier:
    modelo = HistGradientBoostingClassifier(
        max_depth=kwargs.get("max_depth", 4),
        learning_rate=kwargs.get("learning_rate", 0.08),
        max_iter=kwargs.get("max_iter", 250),
        random_state=config.SEED,
    )
    modelo.fit(X_train, y_train.astype(int))
    return modelo


# ---------------------------------------------------------------------------
# predecir
# ---------------------------------------------------------------------------
# Que hace: genera las predicciones de clase ordinal para X.
# Devuelve: np.ndarray de enteros (rating_ordinal predicho).
# ---------------------------------------------------------------------------
def predecir(modelo: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    return modelo.predict(X).astype(int)
