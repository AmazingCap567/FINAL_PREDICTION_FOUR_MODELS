# -*- coding: utf-8 -*-
"""
modelo_4_lightgbm
==================
Modelo 4 de 4 del comparativo. LightGBM (gradient boosting por hojas), uno
por ECR, con `class_weight="balanced"` para compensar el desbalance de
clases (la mayoria de observaciones son "sin cambio de rating").

Por que se agrega junto al modelo 1 (HistGradientBoosting): ambos son
gradient boosting de arboles, pero difieren en el tratamiento del
desbalance (LightGBM aqui pondera clases explicitamente; HistGB no) y en
la estrategia de crecimiento del arbol (leaf-wise en LightGBM vs
level-wise en HistGB), lo que en la practica les da sesgos distintos ante
clases raras (como downgrades fuertes) — util precisamente para el
objetivo de riesgo/deterioro, no solo para el rating.

Ventajas: rapido, maneja bien variables categoricas/numericas mixtas,
ponderacion de clases nativa.
Limitaciones: mas hiperparametros que ajustar que HistGB; con muestras muy
chicas (< ~50 filas) puede sobreajustar si no se limita la profundidad.

Interfaz comun con los otros 3 modelos (ver motor/entrenamiento/orquestador.py):
    entrenar(X_train, y_train, **kwargs) -> modelo_entrenado
    predecir(modelo_entrenado, X) -> np.ndarray de clases ordinales predichas
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ... import config

NOMBRE_MODELO = "LightGBM"


# ---------------------------------------------------------------------------
# entrenar
# ---------------------------------------------------------------------------
# Que hace: ajusta un LGBMClassifier multiclase con pesos de clase
# balanceados (compensa que la mayoria de las observaciones son "sin
# cambio de rating" y los downgrades fuertes son raros).
# Recibe: X_train (DataFrame de features), y_train (Serie de clase ordinal).
# Devuelve: el modelo entrenado (objeto LightGBM, ya tiene .predict()).
# Nota metodologica: num_leaves y max_depth se mantienen bajos (8 y 4) por
# el mismo motivo que en el modelo 1: el tamano de muestra por tipo de
# entidad es pequeno y un arbol profundo memoriza en vez de generalizar.
# ---------------------------------------------------------------------------
def entrenar(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> LGBMClassifier:
    modelo = LGBMClassifier(
        n_estimators=kwargs.get("n_estimators", 200),
        num_leaves=kwargs.get("num_leaves", 8),
        max_depth=kwargs.get("max_depth", 4),
        learning_rate=kwargs.get("learning_rate", 0.05),
        class_weight="balanced",
        random_state=config.SEED,
        verbosity=-1,
    )
    modelo.fit(X_train, y_train.astype(int))
    return modelo


# ---------------------------------------------------------------------------
# predecir
# ---------------------------------------------------------------------------
# Que hace: genera las predicciones de clase ordinal para X.
# Devuelve: np.ndarray de enteros (rating_ordinal predicho).
# ---------------------------------------------------------------------------
def predecir(modelo: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return modelo.predict(X).astype(int)
