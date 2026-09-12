# -*- coding: utf-8 -*-
"""
sbs_loader.py
=============
Carga robusta de los archivos de clasificaciones SBS (.xls), que en la
practica NO son archivos Excel binarios sino documentos HTML con una tabla
(esto se verifico inspeccionando los bytes crudos de los 8 archivos: todos
comienzan con '<table').

El loader:
  1. Detecta si el archivo es realmente un binario Excel (firma OLE2 / ZIP)
     o un documento HTML, sin asumir la extension.
  2. Si es HTML, usa pandas.read_html con encoding explicito 'utf-8'
     (se verifico que omitir el encoding produce mojibake: "Crédito" se
     lee como "CrÃ©dito").
  3. Si es Excel real, usa pandas.read_excel como fallback.
  4. De las multiples tablas que trae el HTML (la tabla principal + tablas
     de 1x2 con notas al pie), selecciona automaticamente la que tiene mas
     columnas (la tabla principal de clasificaciones).
  5. Devuelve un DataFrame "largo" (long format) con una fila por
     (ENTITY_TYPE_RAW, ENTITY_NAME_RAW, ECR, RATING_RAW, FECHA_CORTE).
"""

from __future__ import annotations
import os
import re
import pandas as pd

from .. import config
from ..utils import log, normalize_column_token


def _is_probably_html(path: str) -> bool:
    """Inspecciona los primeros bytes del archivo para decidir el formato real."""
    with open(path, "rb") as f:
        head = f.read(512)
    # Firma OLE2 (.xls binario clasico)
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return False
    # Firma ZIP (.xlsx moderno, a veces mal nombrado .xls)
    if head.startswith(b"PK"):
        return False
    # Si contiene tags HTML tipicos, es HTML
    lowered = head.lower()
    if b"<table" in lowered or b"<html" in lowered or b"<!doctype" in lowered:
        return True
    # Por defecto, intentar HTML (mas comun en exportes SBS)
    return True


def _select_main_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Elige la tabla con mayor numero de columnas Y de filas (la tabla real de datos)."""
    best = max(tables, key=lambda t: (t.shape[1], t.shape[0]))
    return best


def _match_ecr_columns(columns: list[str]) -> dict[str, str]:
    """
    Empareja las columnas reales del archivo (nombres largos de agencia) con
    los codigos canonicos de ECR usando los "hints" definidos en config.
    Lanza error claro si alguna ECR objetivo no aparece en absoluto.
    """
    mapping: dict[str, str] = {}
    for col in columns:
        tok = normalize_column_token(col)
        for ecr, hints in config.ECR_COLUMN_HINTS.items():
            if any(h in tok for h in hints):
                mapping[col] = ecr
                break
    found_ecrs = set(mapping.values())
    missing = set(config.ECR_LIST) - found_ecrs
    if missing:
        raise ValueError(
            f"No se encontraron columnas para las ECR {missing} en columnas: {columns}"
        )
    return mapping


def load_sbs_classification_file(path: str, corte_nombre: str) -> pd.DataFrame:
    """
    Carga un unico archivo de clasificacion SBS y devuelve un DataFrame largo con
    columnas: CORTE, FECHA_CORTE, ENTITY_TYPE_RAW, ENTITY_NAME_RAW, ECR, RATING_RAW
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo de clasificacion: {path}")

    is_html = _is_probably_html(path)
    if is_html:
        tables = pd.read_html(path, encoding="utf-8")
        df = _select_main_table(tables)
    else:
        df = pd.read_excel(path)

    if df.shape[1] < 3:
        raise ValueError(f"El archivo {path} no parece tener la estructura esperada (columnas={df.shape[1]})")

    # Las dos primeras columnas son Tipo de Entidad y Entidad; el resto son ECR.
    cols = list(df.columns)
    tipo_col, entidad_col = cols[0], cols[1]
    ecr_cols = cols[2:]

    ecr_map = _match_ecr_columns(ecr_cols)

    # Elimina filas completamente vacias (el HTML SBS intercala filas de
    # separacion visual sin contenido)
    df = df.dropna(how="all")
    df = df[df[entidad_col].notna()]

    records = []
    fecha_corte = config.CORTES_CLASIFICACION[corte_nombre]
    for _, row in df.iterrows():
        tipo_raw = row[tipo_col]
        nombre_raw = row[entidad_col]
        if pd.isna(tipo_raw) or pd.isna(nombre_raw):
            continue
        for col_original, ecr in ecr_map.items():
            valor = row[col_original]
            if pd.isna(valor):
                continue
            records.append({
                "CORTE": corte_nombre,
                "FECHA_CORTE": fecha_corte,
                "ENTITY_TYPE_RAW": str(tipo_raw).strip(),
                "ENTITY_NAME_RAW": str(nombre_raw).strip(),
                "ECR": ecr,
                "RATING_RAW": str(valor).strip(),
            })

    long_df = pd.DataFrame.from_records(records)
    return long_df


def load_all_sbs_classifications() -> pd.DataFrame:
    """Carga y concatena los 8 cortes disponibles en config.CORTES_CLASIFICACION."""
    frames = []
    for corte_nombre in config.CORTES_CLASIFICACION:
        path = os.path.join(config.CLASIFICACIONES_DIR, f"{corte_nombre}.xls")
        log(f"Cargando clasificacion SBS: {corte_nombre} ({path})")
        frame = load_sbs_classification_file(path, corte_nombre)
        log(f"  -> {len(frame)} pares (entidad, ECR) con valor no nulo")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result
