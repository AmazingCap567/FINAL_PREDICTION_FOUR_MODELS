# -*- coding: utf-8 -*-
"""
feature_engineering.py
=======================
Construccion de variables temporales (niveles, variaciones y tendencias)
sobre el panel financiero unificado, SIN usar informacion posterior a la
fecha base de cada fila (no hay fuga de informacion porque cada variable de
variacion se calcula unicamente contra observaciones anteriores del mismo
ENTITY_ID).

Decision metodologica documentada:
  Los 4 datasets financieros publican, para muchas partidas, la version en
  Moneda Nacional (sufijo _MN), Moneda Extranjera (sufijo _ME) y el TOTAL
  (sin sufijo). Para evitar una explosion innecesaria de variables (los 3
  componentes estan casi perfectamente relacionados: TOTAL = MN + ME), las
  variables de variacion/tendencia (lags) se calculan SOLO sobre las
  columnas TOTAL. Las columnas MN/ME originales se conservan como niveles
  (no se eliminan aqui; la seleccion de variables decidira si aportan senal
  incremental), pero no se les genera todo el arbol de lags para controlar
  la dimensionalidad, tal como pide el enunciado.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd

from .. import config
from ..utils import log


def _is_id_like(col: str) -> bool:
    return col in config.ID_LIKE_COLUMNS or col in (
        "ENTITY_ID", "ENTITY_NAME_NORM", "SOURCE", "FECHA", "ENTIDAD",
        "TIPO_DE_ENTIDAD", "AÑO", "MES",
    )


# ---------------------------------------------------------------------------
# get_lag_base_columns
# ---------------------------------------------------------------------------
# Que hace: filtra del panel las columnas sobre las que SI se generaran
# variables de lag/tendencia: numericas, no identificadoras, y unicamente
# las versiones TOTAL (excluye _MN, _ME y los indicadores _MISSING).
# Por que asi: ver nota metodologica al inicio del archivo (evitar
# explosion de variables casi redundantes).
# Recibe: panel (DataFrame financiero unificado, con o sin lags todavia).
# Devuelve: lista de nombres de columna candidatas a lag.
# ---------------------------------------------------------------------------
def get_lag_base_columns(panel: pd.DataFrame) -> list[str]:
    """
    Selecciona las columnas TOTAL (sin sufijo _MN/_ME/_MISSING) sobre las que
    se calcularan variaciones/tendencias, evitando explosion de variables.
    """
    candidates = []
    for col in panel.columns:
        if _is_id_like(col):
            continue
        if col.endswith("_MISSING"):
            continue
        if col.endswith("_MN") or col.endswith("_ME"):
            continue
        if not pd.api.types.is_numeric_dtype(panel[col]):
            continue
        candidates.append(col)
    return candidates


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------
# Que hace: agrega, para cada columna base (TOTAL), 3 familias de variables
# derivadas:
#   - <VAR>_DELTA_{k}M : diferencia absoluta vs. k meses atras
#   - <VAR>_PCTCHG_{k}M: variacion porcentual vs. k meses atras
#   - <VAR>_TREND_6M   : pendiente simple (regresion lineal) de los ultimos
#                         hasta-6 puntos disponibles (tendencia de corto plazo)
# Por que asi (anti-fuga): todo se calcula por ENTITY_ID, ordenado por
# FECHA, usando exclusivamente `.shift(k)` (observaciones YA OCURRIDAS de
# la misma entidad). Ninguna de estas columnas puede "ver" el futuro de
# esa entidad, sin importar en que corte se use despues la fila.
# Recibe: panel (DataFrame financiero unificado, sin lags).
# Devuelve: el mismo panel + las columnas nuevas (concatenadas).
# ---------------------------------------------------------------------------
def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega, para cada columna base (TOTAL), variables de:
      - <VAR>_DELTA_{k}M : diferencia absoluta vs. k meses atras
      - <VAR>_PCTCHG_{k}M: variacion porcentual vs. k meses atras
      - <VAR>_TREND_6M   : pendiente simple (regresion lineal) de los ultimos
                            hasta-6 puntos disponibles (tendencia de corto plazo)

    Todo se calcula por ENTITY_ID, ordenado por FECHA, usando exclusivamente
    observaciones pasadas de la misma entidad (shift), por lo que no hay
    fuga de informacion futura.
    """
    panel = panel.sort_values(["ENTITY_ID", "FECHA"]).reset_index(drop=True)
    base_cols = get_lag_base_columns(panel)
    log(f"Feature engineering: generando lags/tendencias sobre {len(base_cols)} variables base (columnas TOTAL).")

    grouped = panel.groupby("ENTITY_ID", sort=False)

    new_cols = {}
    for col in base_cols:
        for k in config.LAG_MONTHS:
            shifted = grouped[col].shift(k)
            new_cols[f"{col}_DELTA_{k}M"] = panel[col] - shifted
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = (panel[col] - shifted) / shifted.abs()
            pct = pct.replace([np.inf, -np.inf], np.nan)
            new_cols[f"{col}_PCTCHG_{k}M"] = pct

    # Tendencia simple: pendiente de regresion lineal sobre la ventana de los
    # ultimos 6 puntos disponibles (incluyendo el actual), por entidad.
    def _rolling_slope(s: pd.Series, window: int = 6) -> pd.Series:
        def slope(values):
            values = values[~np.isnan(values)]
            if len(values) < 2:
                return np.nan
            x = np.arange(len(values))
            return np.polyfit(x, values, 1)[0]
        return s.rolling(window=window, min_periods=2).apply(slope, raw=True)

    for col in base_cols:
        new_cols[f"{col}_TREND_6M"] = grouped[col].transform(lambda s: _rolling_slope(s, 6))

    new_df = pd.DataFrame(new_cols, index=panel.index)
    result = pd.concat([panel, new_df], axis=1)
    log(f"Feature engineering: panel paso de {panel.shape[1]} a {result.shape[1]} columnas.")
    return result
