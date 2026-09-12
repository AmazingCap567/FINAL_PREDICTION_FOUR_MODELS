# -*- coding: utf-8 -*-
"""
financial_loader.py
====================
Carga y estandarizacion de los 4 datasets financieros:
  - Bancos            (CSV, separador ';', PERIODO='YYYY-MM')
  - Caja Municipal    (CSV, separador ';', PERIODO='DD/MM/YYYY')
  - Financieras       (XLSX, hoja 'DATASET_PRELIMINAR_134', columna FECHA ya en ISO)
  - CRAC              (XLSX, hoja 'DATASET')

Cada fuente se estandariza a un esquema comun minimo:
    ENTIDAD, TIPO_DE_ENTIDAD, FECHA (datetime, fin de mes), + variables financieras propias.

No se fuerza un esquema de columnas identico entre fuentes (los 4 datasets
tienen catalogos de variables distintos): se preserva cada variable con su
nombre original y se concatenan con union de columnas (outer), dejando NaN
donde una fuente no tiene esa variable. Esto es intencional: el modelo
tratara esas ausencias como missing (con su propio indicador), no como cero.

IMPORTANTE: las columnas de clasificacion que pudieran existir dentro de
`dataset_financieras.xlsx` (hoja 'CLASIFICACIONES') NUNCA se cargan aqui;
el target proviene exclusivamente de sbs_loader.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from .. import config
from ..utils import log


def _month_end(series_period: pd.Series) -> pd.Series:
    """Convierte una serie de periodos mensuales (cualquier formato parseable) a fin de mes."""
    dt = pd.to_datetime(series_period)
    return dt + pd.offsets.MonthEnd(0)


def load_bancos() -> pd.DataFrame:
    df = pd.read_csv(config.FILE_BANCOS, sep=";")
    df["TIPO_DE_ENTIDAD"] = "BANCO"
    df["FECHA"] = _month_end(df["PERIODO"])
    df["ENTIDAD"] = df["ENTIDAD"].astype(str).str.strip()
    return df


def load_caja_municipal() -> pd.DataFrame:
    df = pd.read_csv(config.FILE_CAJA_MUNICIPAL, sep=";")
    df["TIPO_DE_ENTIDAD"] = "CMAC"
    df["FECHA"] = pd.to_datetime(df["PERIODO"], format="%d/%m/%Y") + pd.offsets.MonthEnd(0)
    df["ENTIDAD"] = df["ENTIDAD"].astype(str).str.strip()
    return df


def load_financieras() -> pd.DataFrame:
    xl = pd.ExcelFile(config.FILE_FINANCIERAS)
    if "DATASET_PRELIMINAR_134" not in xl.sheet_names:
        raise ValueError(
            f"Se esperaba la hoja 'DATASET_PRELIMINAR_134' en {config.FILE_FINANCIERAS}, "
            f"hojas disponibles: {xl.sheet_names}"
        )
    df = xl.parse("DATASET_PRELIMINAR_134")
    df["TIPO_DE_ENTIDAD"] = "FINANCIERA"
    df["FECHA"] = pd.to_datetime(df["FECHA"]) + pd.offsets.MonthEnd(0)
    df["ENTIDAD"] = df["ENTIDAD"].astype(str).str.strip()
    return df


def load_crac() -> pd.DataFrame:
    xl = pd.ExcelFile(config.FILE_CRAC)
    if "DATASET" not in xl.sheet_names:
        raise ValueError(
            f"Se esperaba la hoja 'DATASET' en {config.FILE_CRAC}, hojas disponibles: {xl.sheet_names}"
        )
    df = xl.parse("DATASET")
    df["TIPO_DE_ENTIDAD"] = "CRAC"
    df["FECHA"] = _month_end(df["PERIODO"])
    df["ENTIDAD"] = df["ENTIDAD"].astype(str).str.strip()

    # Requisito explicito: CRAC contiene columnas completamente vacias y
    # constantes; se eliminan del conjunto de modelado.
    nunique_incl_na = df.nunique(dropna=False)
    constant_cols = nunique_incl_na[nunique_incl_na <= 1].index.tolist()
    constant_cols = [c for c in constant_cols if c not in ("ENTIDAD", "TIPO_DE_ENTIDAD")]
    if constant_cols:
        log(f"CRAC: eliminando {len(constant_cols)} columnas constantes/vacias: {constant_cols[:10]}...")
        df = df.drop(columns=constant_cols)
    return df


def _apply_entity_continuity_renames(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Unifica bajo un mismo nombre canonico a entidades que cambiaron de razon
    social durante el periodo cubierto por los datos (ver
    config.FINANCIAL_ENTITY_CONTINUITY_RENAMES). Debe aplicarse ANTES de
    construir el registro de entidades y el panel unificado, para que ambos
    tramos de historial (nombre anterior y nombre nuevo) se traten como una
    unica entidad con ENTITY_ID continuo.

    El matching se hace sobre el nombre NORMALIZADO (mayusculas, sin acentos)
    para no depender de que el archivo fuente use exactamente la misma
    capitalizacion que config.py.
    """
    from ..utils import normalize_entity_name

    renames_norm = {normalize_entity_name(k): v for k, v in config.FINANCIAL_ENTITY_CONTINUITY_RENAMES.items()}
    for name, df in sources.items():
        norm_series = df["ENTIDAD"].apply(normalize_entity_name)
        mask = norm_series.isin(renames_norm.keys())
        if mask.any():
            for norm_nuevo, anterior in renames_norm.items():
                n = int((norm_series == norm_nuevo).sum())
                if n:
                    ejemplo = df.loc[norm_series == norm_nuevo, "ENTIDAD"].iloc[0]
                    log(f"Fuente '{name}': {n} filas de '{ejemplo}' homologadas al nombre "
                        f"canonico anterior '{anterior}' (cambio de razon social, misma entidad).")
            df.loc[mask, "ENTIDAD"] = norm_series[mask].map(renames_norm)
    return sources


def load_all_financial_sources() -> dict[str, pd.DataFrame]:
    """Devuelve un dict {nombre_fuente: dataframe estandarizado (sin concatenar)}."""
    sources = {
        "BANCOS": load_bancos(),
        "CAJA_MUNICIPAL": load_caja_municipal(),
        "FINANCIERAS": load_financieras(),
        "CRAC": load_crac(),
    }
    sources = _apply_entity_continuity_renames(sources)
    for name, df in sources.items():
        log(f"Fuente financiera '{name}': {df.shape[0]} filas, {df.shape[1]} columnas, "
            f"{df['ENTIDAD'].nunique()} entidades, rango fechas "
            f"[{df['FECHA'].min().date()} .. {df['FECHA'].max().date()}]")
    return sources


def build_unified_panel(sources: dict[str, pd.DataFrame], entity_id_map: dict[str, str]) -> pd.DataFrame:
    """
    Concatena las 4 fuentes en un unico panel largo (una fila por
    entidad-mes), agregando ENTITY_ID resuelto via entity_id_map
    (clave = ENTITY_NAME_NORM). Las columnas financieras que no existen en
    una fuente quedan como NaN (missing real, no se rellenan con 0).
    """
    from ..utils import normalize_entity_name

    frames = []
    for source_name, df in sources.items():
        d = df.copy()
        d["SOURCE"] = source_name
        d["ENTITY_NAME_NORM"] = d["ENTIDAD"].apply(normalize_entity_name)
        d["ENTITY_ID"] = d["ENTITY_NAME_NORM"].map(entity_id_map)
        frames.append(d)

    panel = pd.concat(frames, ignore_index=True, sort=False)
    n_sin_id = panel["ENTITY_ID"].isna().sum()
    if n_sin_id > 0:
        log(f"ADVERTENCIA: {n_sin_id} filas del panel financiero sin ENTITY_ID asignado (se descartan).", level="WARN")
        panel = panel[panel["ENTITY_ID"].notna()].copy()

    panel = panel.sort_values(["ENTITY_ID", "FECHA"]).reset_index(drop=True)
    return panel
