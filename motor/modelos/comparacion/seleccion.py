# -*- coding: utf-8 -*-
"""
seleccion.py
============
Metodologia REPRODUCIBLE para elegir, por ECR (y opcionalmente por tipo de
entidad), cual de los 4 modelos comparables "gana". No se selecciona por
una sola metrica (accuracy serial mata la informacion de calibracion y de
desbalance), sino por un puntaje compuesto explicito.

Criterios usados (ver punto 15 del proyecto) y su peso:

  1. Weighted Kappa (Cohen, pesos cuadraticos)  -> 0.35
     Mide acierto ordinal: penaliza mas los errores lejanos (A+ predicho
     como E) que los cercanos (A predicho como A-). Es la metrica principal
     porque el problema es fundamentalmente ordinal.
  2. MAE ordinal (distancia media en notches)   -> 0.25
     Complementa al Kappa con una lectura directamente interpretable
     ("en promedio nos equivocamos por X escalones").
  3. F1 macro (por clase, sin ponderar por frecuencia) -> 0.20
     Evita que el puntaje lo domine la clase mayoritaria ("sin cambio"),
     relevante porque el desbalance es fuerte.
  4. Brier score (calibracion de probabilidad, si el modelo la expone) -> 0.20
     Solo se incluye si el modelo devuelve predict_proba; si no, este peso
     se redistribuye proporcionalmente entre los otros tres criterios.

Todas las metricas se calculan siempre sobre el mismo split HOLDOUT
(nunca sobre TRAIN), y todos los modelos se comparan sobre exactamente las
mismas filas.

Este modulo NO decide arbitrariamente con un numero magico: expone
`construir_ranking()` para que el resultado (tabla ordenada + ganador) se
pueda inspeccionar y cuestionar, no solo un string con el nombre ganador.
"""

from __future__ import annotations
import pandas as pd


PESOS_DEFAULT = {
    "weighted_kappa": 0.35,
    "mae_ordinal_inv": 0.25,   # se usa 1/(1+MAE) para que "mayor sea mejor" en todos los criterios
    "f1_macro": 0.20,
    "brier_inv": 0.20,         # se usa 1-brier para que "mayor sea mejor"
}


# ---------------------------------------------------------------------------
# construir_ranking
# ---------------------------------------------------------------------------
# Que hace: recibe un DataFrame con una fila por modelo y columnas de
# metricas ya calculadas (weighted_kappa, mae_ordinal, f1_macro, brier o
# NaN si el modelo no expone probabilidades), normaliza cada metrica a una
# escala donde "mayor es mejor" en [0,1] (min-max dentro del propio grupo
# de modelos comparados), calcula el puntaje compuesto ponderado y ordena.
# Por que asi: normalizar dentro del grupo evita que una metrica con rango
# numerico mas amplio (p.ej. MAE en notches, 0-12) domine arbitrariamente
# sobre otra con rango mas chico (p.ej. Kappa, -1 a 1).
# Recibe: metricas_por_modelo (DataFrame, index = nombre del modelo).
# Devuelve: (DataFrame ordenado de mejor a peor con columna 'puntaje',
#            nombre del modelo ganador).
# ---------------------------------------------------------------------------
def construir_ranking(metricas_por_modelo: pd.DataFrame, pesos: dict | None = None) -> tuple[pd.DataFrame, str]:
    pesos = dict(pesos or PESOS_DEFAULT)
    df = metricas_por_modelo.copy()

    df["mae_ordinal_inv"] = 1.0 / (1.0 + df["mae_ordinal"])
    if "brier" in df.columns:
        df["brier_inv"] = 1.0 - df["brier"]
    else:
        df["brier_inv"] = float("nan")

    # Si ningun modelo expone Brier, redistribuir su peso proporcionalmente
    # entre los otros tres criterios en vez de penalizar a todos por igual.
    if df["brier_inv"].isna().all():
        peso_brier = pesos.pop("brier_inv", 0.0)
        restantes = list(pesos.keys())
        for k in restantes:
            pesos[k] += peso_brier * (pesos[k] / sum(pesos[k2] for k2 in restantes))
        df = df.drop(columns=["brier_inv"])

    criterios = [c for c in pesos if c in df.columns or c in ("mae_ordinal_inv", "brier_inv")]
    df["puntaje"] = 0.0
    for criterio, peso in pesos.items():
        if criterio not in df.columns:
            continue
        col = df[criterio]
        rango = col.max() - col.min()
        norm = (col - col.min()) / rango if rango > 0 else pd.Series(0.5, index=col.index)
        df["puntaje"] += peso * norm.fillna(norm.mean())

    df = df.sort_values("puntaje", ascending=False)
    ganador = df.index[0]
    return df, ganador


# ---------------------------------------------------------------------------
# resumen_texto
# ---------------------------------------------------------------------------
# Que hace: arma una linea de consola legible con el ganador y su
# puntaje, para el modo resumido de salida (no verbose) del punto 9 del
# proyecto.
# ---------------------------------------------------------------------------
def resumen_texto(ranking: pd.DataFrame, ganador: str) -> str:
    fila = ranking.loc[ganador]
    return f"Mejor modelo: {ganador} (puntaje compuesto={fila['puntaje']:.3f})"
