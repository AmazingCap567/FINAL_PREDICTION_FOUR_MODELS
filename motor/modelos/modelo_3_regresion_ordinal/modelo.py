# -*- coding: utf-8 -*-
"""
modelo_3_regresion_ordinal
===========================
Modelo 3 de 4 del comparativo. Regresion logistica ORDINAL (mord.LogisticIT,
"Immediate-Threshold" / proportional odds), uno por ECR.

Por que se agrega: a diferencia de HistGradientBoosting (modelo 1) y LightGBM
(modelo 4), que tratan el rating como una clase nominal cualquiera, este
modelo asume explicitamente que las clases estan ORDENADAS (A+ > A > A- >
... > E) y penaliza mas los errores que "saltan" varios escalones que los
que se equivocan por un escalon. Es el modelo mas simple e interpretable de
los 4 (coeficientes lineales por variable), y sirve como referencia de
"cuanto aporta" la no linealidad de los modelos de arboles/red neuronal.

Ventajas: interpretable, pocos parametros (bajo riesgo de sobreajuste con
muestras chicas), respeta el orden.
Limitaciones: asume relaciones lineales entre variables y logit acumulado
(supuesto de "odds proporcionales"); puede quedarse corto si la relacion
real es muy no lineal.

Interfaz comun con los otros 3 modelos (ver motor/entrenamiento/orquestador.py):
    entrenar(X_train, y_train, **kwargs) -> modelo_entrenado
    predecir(modelo_entrenado, X) -> np.ndarray de clases ordinales predichas
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mord

NOMBRE_MODELO = "RegresionLogisticaOrdinal"


# ---------------------------------------------------------------------------
# entrenar
# ---------------------------------------------------------------------------
# Que hace: imputa (mediana, por si llega algun NaN residual), escala
# (StandardScaler; mord es sensible a la escala de las variables al ser un
# modelo lineal) y ajusta mord.LogisticIT sobre las clases ordinales.
# Recibe: X_train (DataFrame de features ya seleccionadas), y_train (Serie
# de clase ordinal; NO necesita ser consecutiva sin huecos -- eso lo
# resuelve internamente esta funcion, ver nota below).
# Devuelve: un dict {"pipeline": Pipeline entrenado, "clases_originales":
# array} -- se envuelve en dict (en vez de devolver solo el Pipeline) para
# poder reconstruir las clases ordinales originales al predecir.
# Nota metodologica: mord.LogisticIT exige que las clases de y sean
# enteros CONSECUTIVOS empezando en 0 (0,1,2,...), sin huecos. Como el
# universo real de ratings por ECR puede tener huecos (p.ej. ninguna
# entidad calificada como "C-" en TRAIN para esa ECR), se remapean las
# clases presentes en TRAIN a un rango denso 0..K-1 antes de entrenar, y
# se revierte el mapeo en `predecir`.
# ---------------------------------------------------------------------------
def entrenar(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> dict:
    clases_originales = np.sort(y_train.unique())
    remap = {orig: denso for denso, orig in enumerate(clases_originales)}
    y_denso = y_train.map(remap).astype(int)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("modelo", mord.LogisticIT(alpha=kwargs.get("alpha", 1.0))),
    ])
    pipeline.fit(X_train, y_denso)
    return {"pipeline": pipeline, "clases_originales": clases_originales}


# ---------------------------------------------------------------------------
# predecir
# ---------------------------------------------------------------------------
# Que hace: genera las predicciones de clase ordinal para X, revirtiendo
# el remapeo denso->clase original hecho en entrenar().
# Devuelve: np.ndarray de enteros (rating_ordinal predicho, en la escala
# ORIGINAL, no la densa interna).
# ---------------------------------------------------------------------------
def predecir(modelo: dict, X: pd.DataFrame) -> np.ndarray:
    preds_densas = modelo["pipeline"].predict(X).astype(int)
    return modelo["clases_originales"][preds_densas]
