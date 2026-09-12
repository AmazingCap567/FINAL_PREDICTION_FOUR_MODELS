# -*- coding: utf-8 -*-
"""
baselines_piso.py
==================
"Piso" de referencia contra el que se comparan los 4 modelos reales.

Estos NO cuentan como modelos comparables (punto 7 del proyecto): son
reglas triviales que cualquier modelo de verdad debe superar para
justificar su complejidad. Si un modelo no le gana a la persistencia,
no aporta valor frente a "asumir que no cambia nada".

  - Clase mayoritaria: predice siempre la clase mas frecuente de TRAIN
    (por ECR). Sirve de piso absoluto.
  - Persistencia: predice el ULTIMO rating publicado disponible para esa
    misma entidad y esa misma ECR en un corte ANTERIOR (no necesariamente
    el corte inmediatamente previo, sino el mas reciente disponible). Si
    la entidad nunca fue clasificada antes por esa ECR, no se puede aplicar
    persistencia y esa fila se excluye de la evaluacion de este baseline
    especifico.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from ... import config


# ---------------------------------------------------------------------------
# build_persistence_lookup
# ---------------------------------------------------------------------------
# Que hace: indexa TODO el historico de clasificaciones resueltas (no solo
# las filas que terminaron en el dataset supervisado) en un diccionario
# {(entity_id, ecr): [(orden_corte, rating_ordinal), ...]} ordenado
# cronologicamente.
# Por que asi: predict_persistence necesita poder "mirar hacia atras" en el
# tiempo sin depender de que exista una fila financiera para ese periodo
# (una entidad puede tener rating conocido en un corte aunque ese mes no
# haya datos financieros utilizables).
# Recibe: DataFrame largo de clasificaciones ya resueltas a ENTITY_ID.
# Devuelve: dict de listas (orden_corte, rating_ordinal) por (entidad, ECR).
# ---------------------------------------------------------------------------
def build_persistence_lookup(classifications_resolved: pd.DataFrame) -> dict:
    corte_order = {c: i for i, c in enumerate(config.CORTES_CLASIFICACION.keys())}
    lookup: dict[tuple, list[tuple[int, int]]] = {}
    for _, r in classifications_resolved.iterrows():
        if pd.isna(r.get("rating_ordinal")):
            continue
        key = (r["ENTITY_ID"], r["ECR"])
        lookup.setdefault(key, []).append((corte_order[r["CORTE"]], r["rating_ordinal"]))
    for key in lookup:
        lookup[key] = sorted(lookup[key], key=lambda x: x[0])
    return lookup


# ---------------------------------------------------------------------------
# predict_persistence
# ---------------------------------------------------------------------------
# Que hace: busca, para la fila `row` (una observacion entidad-corte), el
# rating_ordinal MAS RECIENTE que exista estrictamente ANTES de row["CORTE"]
# para esa entidad y esa ECR.
# Por que asi: es la definicion operativa de "persistencia" -> "va a seguir
# igual que la ultima vez que se supo algo de ella", sin asumir que el
# corte anterior inmediato tiene dato (puede no tenerlo).
# Devuelve: el rating_ordinal previo, o None si no hay historia anterior.
# ---------------------------------------------------------------------------
def predict_persistence(row: pd.Series, ecr: str, lookup: dict) -> float | None:
    corte_order = {c: i for i, c in enumerate(config.CORTES_CLASIFICACION.keys())}
    current_order = corte_order[row["CORTE"]]
    key = (row["ENTITY_ID"], ecr)
    history = lookup.get(key, [])
    prior = [val for (order, val) in history if order < current_order]
    if not prior:
        return None
    return prior[-1]


# ---------------------------------------------------------------------------
# majority_class_baseline
# ---------------------------------------------------------------------------
# Que hace: calcula la clase (rating ordinal) mas frecuente dentro de TRAIN
# para una ECR dada.
# Por que asi: es el piso absoluto de comparacion (ignora toda la
# informacion de la entidad). Si algun modelo no le gana, no sirve.
# Devuelve: entero (rating_ordinal mas frecuente en TRAIN).
# ---------------------------------------------------------------------------
def majority_class_baseline(train_df: pd.DataFrame, ecr: str) -> int:
    y = train_df[f"TARGET_ORD__{ecr}"].dropna()
    if len(y) == 0:
        return int(np.median(list(config.RATING_MASTER_ORDINAL.values())))
    return int(y.mode().iloc[0])
