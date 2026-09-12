# -*- coding: utf-8 -*-
"""
dataset_builder.py
===================
Integra clasificaciones (target) + panel financiero (features) respetando
estrictamente el orden temporal:

    INFORMACION FINANCIERA DISPONIBLE EN t  ->  CLASIFICACION ECR EN t' > t

Para cada (ENTITY_ID, CORTE) se busca la ultima fecha financiera disponible
ESTRICTAMENTE ANTERIOR a FECHA_CORTE, dentro de una ventana maxima de
`config.MAX_LOOKBACK_MONTHS` meses. Si no existe ninguna observacion
financiera dentro de esa ventana, NO se construye la fila (no se inventa
informacion ni se usa informacion demasiado vieja como si fuera reciente).

El resultado es un dataset ancho (una fila por ENTITY_ID x CORTE) con:
  - columnas de identificacion (ENTITY_ID, ENTITY_NAME, ENTITY_TYPE, CORTE,
    FECHA_BASE, FECHA_TARGET)
  - columnas de features financieras (extraidas del panel en FECHA_BASE)
  - columnas de target por ECR: TARGET_ORD__<ECR> (ordinal) y
    TARGET_LABEL__<ECR> (etiqueta), con NaN cuando la entidad no fue
    clasificada por esa ECR en ese corte (fila NO se elimina completa: cada
    ECR se evalua de forma independiente via mascara).
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from .. import config
from ..utils import log


# ---------------------------------------------------------------------------
# build_entity_universe_and_resolve
# ---------------------------------------------------------------------------
# Que hace: recorre cada fila de clasificaciones SBS (una por entidad-ECR-
# corte) y la homologa a un ENTITY_ID del universo financiero via
# `registry.resolve_sbs_name`. Excluye tipos de entidad fuera de alcance
# (Seguros, Fondo, etc., que no tienen dataset financiero por diseno) y
# entidades que no lograron homologarse (quedan en el reporte de
# entidades_no_homologadas.csv del registry).
# Por que asi: el TIPO de entidad final se toma siempre del universo
# financiero (fuente de verdad), no de la etiqueta SBS de ese corte
# puntual, porque la SBS a veces etiqueta el tipo de forma inconsistente
# entre cortes para una misma entidad.
# Recibe: classifications (DataFrame largo SBS), registry (EntityRegistry
# ya construido con build_from_financial_sources).
# Devuelve: DataFrame con las mismas columnas + ENTITY_ID, ENTITY_TYPE
# resuelto, ya filtrado a las filas homologables.
# ---------------------------------------------------------------------------
def build_entity_universe_and_resolve(classifications: pd.DataFrame, registry) -> pd.DataFrame:
    from ..utils import normalize_entity_name

    type_by_entity_id = dict(zip(registry.canonical_df["ENTITY_ID"], registry.canonical_df["ENTITY_TYPE"]))

    rows = []
    for _, r in classifications.iterrows():
        tipo_raw_norm = str(r["ENTITY_TYPE_RAW"]).strip().lower()
        tipo_raw_norm_noacc = (
            tipo_raw_norm.replace("é", "e").replace("í", "i").replace("á", "a")
            .replace("ó", "o").replace("ú", "u")
        )
        canonical_type = config.SBS_ENTITY_TYPE_MAP.get(tipo_raw_norm) or config.SBS_ENTITY_TYPE_MAP.get(tipo_raw_norm_noacc)
        if canonical_type is None:
            # Tipo de entidad fuera del universo de modelado (Seguros, Fondo, etc.)
            continue
        entity_id = registry.resolve_sbs_name(r["ENTITY_NAME_RAW"], r["ENTITY_TYPE_RAW"], r["CORTE"])
        if entity_id is None:
            continue
        # El tipo de entidad final proviene del universo financiero (fuente de
        # verdad), no de la etiqueta SBS del corte puntual, para evitar que una
        # misma entidad cambie de tipo entre cortes por una inconsistencia de
        # etiquetado en la SBS.
        final_type = type_by_entity_id.get(entity_id, canonical_type)
        rows.append({**r.to_dict(), "ENTITY_ID": entity_id, "ENTITY_TYPE": final_type})

    resolved = pd.DataFrame(rows)
    return resolved


# ---------------------------------------------------------------------------
# _find_base_snapshot
# ---------------------------------------------------------------------------
# Que hace: dado el panel financiero de UNA entidad (ya ordenado por
# fecha), busca la observacion mas reciente ESTRICTAMENTE ANTERIOR a
# fecha_corte, dentro de una ventana de MAX_LOOKBACK_MONTHS meses.
# Por que asi: esta es la funcion que materializa la regla anti-fuga del
# proyecto ("para predecir en una fecha, solo se puede usar informacion
# disponible hasta esa fecha"). La ventana de lookback evita usar un dato
# financiero demasiado viejo (p.ej. de hace 2 anios) como si fuera
# reciente solo porque es el ultimo disponible.
# Recibe: entity_panel (DataFrame de UNA entidad, ordenado por FECHA),
# fecha_corte (Timestamp del corte de clasificacion objetivo).
# Devuelve: la fila (Series) mas reciente valida, o None si no hay ninguna
# dentro de la ventana permitida (esa entidad-corte se descarta, no se
# inventa un dato).
# ---------------------------------------------------------------------------
def _find_base_snapshot(entity_panel: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.Series | None:
    min_allowed = fecha_corte - pd.DateOffset(months=config.MAX_LOOKBACK_MONTHS)
    candidates = entity_panel[(entity_panel["FECHA"] < fecha_corte) & (entity_panel["FECHA"] >= min_allowed)]
    if candidates.empty:
        return None
    return candidates.iloc[-1]  # la mas reciente dentro de la ventana (panel ya viene ordenado)


def build_supervised_dataset(
    classifications_resolved: pd.DataFrame,
    panel_features: pd.DataFrame,
    entity_meta: pd.DataFrame,
    persistence_lookup: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye el dataset supervisado ancho.

    Devuelve:
      - supervised_df: dataset final (features + targets por ECR)
      - filas_sin_clasificacion_valida: registro de auditoria de pares
        (entidad, corte, ECR) descartados por rating invalido (ya filtrado
        en ratings.annotate_classifications, aqui solo se re-expone el
        conteo para trazabilidad)
    """
    panel_by_entity = {eid: g.sort_values("FECHA") for eid, g in panel_features.groupby("ENTITY_ID")}

    # Pivot: una fila por (ENTITY_ID, CORTE), columnas por ECR con rating_ordinal
    pivot_target = classifications_resolved.pivot_table(
        index=["ENTITY_ID", "ENTITY_TYPE", "CORTE"],
        columns="ECR",
        values="rating_ordinal",
        aggfunc="first",
    )
    pivot_label = classifications_resolved.pivot_table(
        index=["ENTITY_ID", "ENTITY_TYPE", "CORTE"],
        columns="ECR",
        values="rating_label",
        aggfunc="first",
    )
    pivot_target.columns = [f"TARGET_ORD__{c}" for c in pivot_target.columns]
    pivot_label.columns = [f"TARGET_LABEL__{c}" for c in pivot_label.columns]

    base = pivot_target.join(pivot_label).reset_index()

    # Asegura que existan columnas para las 6 ECR aunque falten en algun corte puntual
    for ecr in config.ECR_LIST:
        col_ord = f"TARGET_ORD__{ecr}"
        col_lab = f"TARGET_LABEL__{ecr}"
        if col_ord not in base.columns:
            base[col_ord] = np.nan
        if col_lab not in base.columns:
            base[col_lab] = np.nan

    rows_out = []
    filas_sin_datos_financieros = []

    for _, row in base.iterrows():
        entity_id = row["ENTITY_ID"]
        corte = row["CORTE"]
        fecha_corte = pd.Timestamp(config.CORTES_CLASIFICACION[corte])

        entity_panel = panel_by_entity.get(entity_id)
        if entity_panel is None:
            filas_sin_datos_financieros.append({"ENTITY_ID": entity_id, "CORTE": corte,
                                                 "MOTIVO": "Entidad sin ninguna fila en el panel financiero"})
            continue

        snapshot = _find_base_snapshot(entity_panel, fecha_corte)
        if snapshot is None:
            filas_sin_datos_financieros.append({
                "ENTITY_ID": entity_id, "CORTE": corte,
                "MOTIVO": f"Sin dato financiero dentro de {config.MAX_LOOKBACK_MONTHS} meses antes del corte",
            })
            continue

        out_row = row.to_dict()
        out_row["FECHA_BASE"] = snapshot["FECHA"]
        out_row["FECHA_TARGET"] = fecha_corte
        # Copia todas las columnas de features del snapshot (excluye columnas
        # de identificacion que ya se manejan aparte)
        for col, val in snapshot.items():
            if col in ("ENTITY_ID", "FECHA", "ENTIDAD", "TIPO_DE_ENTIDAD", "SOURCE", "ENTITY_NAME_NORM"):
                continue
            out_row[col] = val
        rows_out.append(out_row)

    supervised_df = pd.DataFrame(rows_out)

    # Agrega nombre legible de entidad
    name_map = dict(zip(entity_meta["ENTITY_ID"], entity_meta["ENTITY_NAME"]))
    supervised_df["ENTITY_NAME"] = supervised_df["ENTITY_ID"].map(name_map)

    sin_datos_df = pd.DataFrame(filas_sin_datos_financieros)

    # --- Rating anterior (persistencia) y delta ---
    # DELTA_ORD__<ECR> = TARGET_ORD - PREV_RATING_ORD. Como la escala va de
    # 0 (A+, mejor) a 12 (E, peor), un DELTA POSITIVO significa que el
    # ordinal SUBIO -> el rating EMPEORO (downgrade real). Un delta
    # negativo es una mejora (upgrade). Ver motor/features/riesgo.py, que
    # reutiliza esta misma columna para el proxy de deterioro.
    from ..modelos.comparacion.baselines_piso import predict_persistence
    for ecr in config.ECR_LIST:
        supervised_df[f"PREV_RATING_ORD__{ecr}"] = supervised_df.apply(
            lambda row: predict_persistence(row, ecr, persistence_lookup), axis=1
        )
        supervised_df[f"DELTA_ORD__{ecr}"] = (
            supervised_df[f"TARGET_ORD__{ecr}"] - supervised_df[f"PREV_RATING_ORD__{ecr}"]
        )

    log(f"Dataset supervisado construido: {len(supervised_df)} filas "
        f"(de {len(base)} pares entidad-corte con al menos una clasificacion valida).")
    if len(sin_datos_df) > 0:
        log(f"{len(sin_datos_df)} pares entidad-corte descartados por falta de dato financiero "
            f"dentro de la ventana de {config.MAX_LOOKBACK_MONTHS} meses.", level="WARN")

    return supervised_df, sin_datos_df


# ---------------------------------------------------------------------------
# agregar_variables_riesgo
# ---------------------------------------------------------------------------
# Que hace: agrega al dataset supervisado ya construido las columnas del
# objetivo de riesgo/deterioro (adicional al motor de rating):
#   - SCORE_DETERIORO: el peor (mayor) DELTA_ORD__<ECR> entre todas las
#     ECR disponibles para esa fila. Como DELTA positivo = downgrade
#     (ver nota en build_supervised_dataset), esto es "cuantos escalones
#     empeoro, en la peor ECR que la cubre, en este corte".
#   - DOWNGRADE_FUERTE: booleano, True si SCORE_DETERIORO >=
#     config.DOWNGRADE_THRESHOLD_NOTCHES (umbral elegido empiricamente,
#     ver comentario en config.py). Este es el target entrenable del
#     objetivo de riesgo (los 4 modelos tambien pueden usarse aqui).
#   - EVENTO_INTERVENCION_PREVIA: booleano, True si esta entidad tiene un
#     evento de intervencion SBS CONFIRMADO (eventos_intervencion.py) con
#     fecha <= FECHA_TARGET de esta fila. Es informativo/de auditoria, NO
#     un target de entrenamiento (la muestra de 4 eventos es demasiado
#     chica para entrenar un clasificador supervisado serio con ella).
# Recibe: supervised_df (el DataFrame que devuelve build_supervised_dataset).
# Devuelve: el mismo DataFrame con las 3 columnas nuevas agregadas.
# ---------------------------------------------------------------------------
def agregar_variables_riesgo(supervised_df: pd.DataFrame) -> pd.DataFrame:
    from . import eventos_intervencion

    df = supervised_df.copy()
    delta_cols = [f"DELTA_ORD__{ecr}" for ecr in config.ECR_LIST if f"DELTA_ORD__{ecr}" in df.columns]
    df["SCORE_DETERIORO"] = df[delta_cols].max(axis=1, skipna=True)
    df["DOWNGRADE_FUERTE"] = df["SCORE_DETERIORO"] >= config.DOWNGRADE_THRESHOLD_NOTCHES

    fecha_limite = eventos_intervencion.fecha_limite_por_entidad()
    # ENTITY_NAME ya viene homologado al nombre canonico (post renombres de
    # continuidad), por lo que se puede comparar directo contra las claves
    # normalizadas de la tabla de eventos.
    from ..utils import normalize_entity_name
    nombre_norm = df["ENTITY_NAME"].map(normalize_entity_name)
    fecha_evento = nombre_norm.map(fecha_limite)
    df["EVENTO_INTERVENCION_PREVIA"] = (
        fecha_evento.notna() & (df["FECHA_TARGET"] >= fecha_evento)
    )
    return df


# ---------------------------------------------------------------------------
# excluir_posteriores_a_intervencion
# ---------------------------------------------------------------------------
# Que hace: filtra filas de un dataset de PREDICCION FUTURA (no de
# entrenamiento) para no generar una prediccion de rating/riesgo para una
# entidad cuya fecha base ya es posterior a su evento de intervencion SBS
# confirmado. Esto materializa el punto 3.B del proyecto ("no debe
# intentarse generar una prediccion para una entidad que ya habia
# desaparecido antes de la fecha de prediccion").
# Importante: esto NO elimina la entidad del ENTRENAMIENTO (su historial
# previo a la intervencion sigue siendo valido y util); solo bloquea que
# se le genere una prediccion HACIA ADELANTE despues de haber quebrado.
# Recibe: df_prediccion (filas candidatas a prediccion futura, con
# columnas ENTITY_NAME y FECHA_BASE), fecha_col (nombre de la columna de
# fecha a comparar, por defecto "FECHA_BASE").
# Devuelve: (df_filtrado, df_excluidas_con_motivo) para poder auditar
# cuantas y cuales filas se excluyeron y por que.
# ---------------------------------------------------------------------------
def excluir_posteriores_a_intervencion(
    df_prediccion: pd.DataFrame, fecha_col: str = "FECHA_TARGET"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from . import eventos_intervencion
    from ..utils import normalize_entity_name

    fecha_limite = eventos_intervencion.fecha_limite_por_entidad()
    nombre_norm = df_prediccion["ENTITY_NAME"].map(normalize_entity_name)
    fecha_evento = nombre_norm.map(fecha_limite)
    mask_excluir = fecha_evento.notna() & (df_prediccion[fecha_col] >= fecha_evento)

    excluidas = df_prediccion.loc[mask_excluir].copy()
    if len(excluidas) > 0:
        excluidas["MOTIVO"] = "Entidad con evento de intervencion SBS confirmado antes o durante la ventana de prediccion"
        log(f"{len(excluidas)} filas excluidas de la prediccion futura por intervencion SBS previa "
            f"confirmada: {sorted(excluidas['ENTITY_NAME'].unique())}", level="WARN")

    return df_prediccion.loc[~mask_excluir].copy(), excluidas[["ENTITY_NAME", fecha_col, "MOTIVO"]] if len(excluidas) else excluidas
