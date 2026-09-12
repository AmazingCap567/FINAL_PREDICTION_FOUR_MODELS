# -*- coding: utf-8 -*-
"""
ratings.py
==========
Normalizacion de la variable objetivo (rating publicado por cada ECR).

Reglas:
  - Se eliminan los modificadores de perspectiva/movimiento (flechas ↑ / ↓).
    Estos modificadores indican tendencia, no nivel; el nivel es lo unico que
    se modela como target principal.
  - Se descartan tokens que no representan un nivel de rating valido
    (p.ej. 'RET' = rating retirado). Estas observaciones se tratan como
    "sin clasificacion valida" (mascara = 0 en la loss), nunca se inventan.
  - Cada ECR tiene su propio conjunto de categorias observadas, pero todas
    comparten el MISMO orden relativo (config.RATING_MASTER_SCALE). El
    ordinal reportado (rating_ordinal) es la posicion en la escala maestra,
    de modo que "equivocarse por 1" significa lo mismo para todas las ECR.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config
from ..utils import strip_rating_modifiers, log


def normalize_rating(raw: str) -> str | None:
    """
    Convierte un rating crudo SBS ('A-↑', 'C+↓', 'RET↓', ...) en su nivel
    canonico ('A-', 'C+', None). Devuelve None si el token no es un nivel
    de rating valido (p.ej. retirado) o no pertenece a la escala maestra.
    """
    cleaned = strip_rating_modifiers(raw)
    if cleaned in config.INVALID_RATING_TOKENS:
        return None
    if cleaned not in config.RATING_MASTER_ORDINAL:
        # Token desconocido: se registra como invalido en vez de adivinar.
        return None
    return cleaned


def build_ecr_scales(classifications_long: pd.DataFrame) -> dict[str, list[str]]:
    """
    Construye, para cada ECR, la lista ordenada (de mejor a peor) de las
    categorias que esa agencia efectivamente publico en los datos
    disponibles. Se documenta explicitamente en vez de asumir que todas las
    ECR comparten la escala completa.
    """
    scales: dict[str, list[str]] = {}
    for ecr in config.ECR_LIST:
        sub = classifications_long[classifications_long["ECR"] == ecr]
        labels_seen = set()
        for raw in sub["RATING_RAW"]:
            lab = normalize_rating(raw)
            if lab is not None:
                labels_seen.add(lab)
        ordered = [l for l in config.RATING_MASTER_SCALE if l in labels_seen]
        scales[ecr] = ordered
        log(f"Escala ordinal para {ecr}: {ordered}")
    return scales


def annotate_classifications(classifications_long: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas rating_label (nivel limpio) y rating_ordinal (posicion en
    la escala maestra GLOBAL, solo de referencia/auditoria) al dataframe
    largo de clasificaciones. Filas cuyo rating no es valido (retirado /
    desconocido) se marcan y se excluyen del set final (se reportan en
    auditoria, no se descartan silenciosamente).
    """
    df = classifications_long.copy()
    df["rating_label"] = df["RATING_RAW"].apply(normalize_rating)
    df["rating_ordinal"] = df["rating_label"].map(config.RATING_MASTER_ORDINAL)

    n_invalid = df["rating_label"].isna().sum()
    if n_invalid > 0:
        log(f"{n_invalid} registros de clasificacion con rating no valido (retirado/desconocido) "
            f"seran excluidos del target.", level="WARN")

    return df


def add_local_ordinal(classifications_long: pd.DataFrame, scales: dict[str, list[str]]) -> pd.DataFrame:
    """
    Sustituye rating_ordinal (escala global) por el ordinal LOCAL a la escala
    propia de cada ECR (0..K_ecr-1), que es el que efectivamente se usa para
    entrenar y evaluar cada cabeza del modelo. Esto respeta el requisito de
    NO asumir que todas las ECR comparten exactamente la misma escala: cada
    agencia se juzga contra su propio conjunto de categorias observadas.
    """
    df = classifications_long.copy()
    local_maps = {ecr: {label: i for i, label in enumerate(scale)} for ecr, scale in scales.items()}

    def _map_row(row):
        ecr = row["ECR"]
        label = row["rating_label"]
        if label is None or pd.isna(label):
            return np.nan
        return local_maps.get(ecr, {}).get(label, np.nan)

    df["rating_ordinal"] = df.apply(_map_row, axis=1)
    return df
