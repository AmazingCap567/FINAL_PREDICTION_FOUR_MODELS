# -*- coding: utf-8 -*-
"""
scripts/run_prediccion.py
===========================
Punto de entrada de PRODUCCION del proyecto (ex predict_future.py). Genera
una prediccion de clasificaciones ECR para un CORTE FUTURO que todavia no
tiene clasificacion SBS publicada, y escribe los Excel que consumira la
futura interfaz en outputs/interfaz/ (punto 10 del proyecto).

Diferencias frente a scripts/run_evaluacion.py:
  - No hay holdout "oculto": se usa TODO el historico de clasificaciones
    conocido como TRAIN, con el ultimo corte como validacion interna
    (early stopping).
  - Cada entidad usa su propia fecha de "presente" (el ultimo dato
    financiero que exista para ella), sin forzar una fecha unica global.
  - No se calculan metricas de acierto (no existe ground truth todavia).
  - Se excluyen explicitamente las entidades con un evento de intervencion
    SBS confirmado anterior a su fecha base (punto 3.B del proyecto: no se
    genera una prediccion futura para una entidad que ya quebro).
  - Se agrega una columna de alerta de riesgo (ALERTA_DOWNGRADE_FUERTE),
    derivada del mismo cambio ordinal predicho por el modelo de rating con
    el mismo umbral empirico usado en el motor (config.DOWNGRADE_THRESHOLD_NOTCHES),
    para no tener que mantener/entrenar un segundo modelo separado solo
    para esto en esta etapa.

Salida (outputs/interfaz/, un archivo por tipo de entidad + consolidado):
  predicciones_<CORTE_OBJETIVO>_<TIPO>.xlsx
  predicciones_<CORTE_OBJETIVO>_TODAS.xlsx

Ejecucion:
    python scripts/run_prediccion.py
"""

from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motor import config
from motor.utils import log, section, subsection
from motor.features import feature_engineering, selection
from motor.datos import dataset_builder, eventos_intervencion
from motor.modelos.comparacion import baselines_piso
import run_evaluacion as pipeline_fases  # reutiliza load_classifications / load_financials_and_homologate

ENTITY_TYPE_TO_IDX = {t: i for i, t in enumerate(config.ENTITY_TYPES)}

# ---------------------------------------------------------------------------
# PARAMETRO PRINCIPAL: fecha del corte objetivo a predecir.
# Cambiar solo esta linea para proyectar a otro corte futuro.
# ---------------------------------------------------------------------------
CORTE_OBJETIVO_NOMBRE = "SEP2026"
CORTE_OBJETIVO_FECHA = pd.Timestamp("2026-09-30")

OUT_DIR = config.INTERFAZ_DIR


def get_latest_known_rating(entity_id, ecr, persistence_lookup):
    """Ultimo rating ordinal conocido (historico completo) para (entidad, ECR)."""
    history = persistence_lookup.get((entity_id, ecr), [])
    if not history:
        return None
    return history[-1][1]


def build_production_split(supervised_df):
    """Split de PRODUCCION: usa TODO el historico conocido como TRAIN, con el
    ultimo corte como validacion interna (early stopping), sin holdout."""
    cortes_ordenados = list(config.CORTES_CLASIFICACION.keys())
    val_corte = cortes_ordenados[-1]
    train_cortes = cortes_ordenados[:-1]

    train_df = supervised_df[supervised_df["CORTE"].isin(train_cortes)].copy()
    val_df = supervised_df[supervised_df["CORTE"] == val_corte].copy()

    log(f"[PRODUCCION] Train: {len(train_df)} filas | cortes = {train_cortes}")
    log(f"[PRODUCCION] Validacion (early stopping): {len(val_df)} filas | corte = {val_corte}")
    return train_df, val_df


def train_production_model(train_df, val_df, scales):
    from motor.modelos.modelo_2_mlp_ordinal import modelo as modelo2

    pipeline_fe = selection.FeaturePipeline()
    pipeline_fe.fit(train_df, target_cols=[f"TARGET_ORD__{e}" for e in config.ECR_LIST])

    X_train_raw = pipeline_fe.transform(train_df)
    X_val_raw = pipeline_fe.transform(val_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    etype_train = train_df["ENTITY_TYPE"].map(ENTITY_TYPE_TO_IDX).values
    etype_val = val_df["ENTITY_TYPE"].map(ENTITY_TYPE_TO_IDX).values

    def _encode_delta(series):
        clipped = np.clip(series.values.astype(float), -config.MAX_DELTA_CLIP, config.MAX_DELTA_CLIP)
        return clipped + config.DELTA_OFFSET

    y_train_delta = {ecr: _encode_delta(train_df[f"DELTA_ORD__{ecr}"]) for ecr in config.ECR_LIST}
    y_val_delta = {ecr: _encode_delta(val_df[f"DELTA_ORD__{ecr}"]) for ecr in config.ECR_LIST}
    delta_k_per_ecr = {ecr: config.N_DELTA_CLASSES for ecr in config.ECR_LIST}

    log("[PRODUCCION] Entrenando MLP ordinal multi-task (target = DELTA) con todo el historico conocido...")
    model, history = modelo2.train_model(
        X_train, etype_train, y_train_delta, X_val, etype_val, y_val_delta,
        n_entity_types=len(config.ENTITY_TYPES), k_per_ecr=delta_k_per_ecr,
    )

    k_per_ecr_local = {ecr: max(len(scales[ecr]), 2) for ecr in config.ECR_LIST}
    majority_fallback = {ecr: baselines_piso.majority_class_baseline(train_df, ecr) for ecr in config.ECR_LIST}
    return model, pipeline_fe, scaler, k_per_ecr_local, majority_fallback, modelo2


# ---------------------------------------------------------------------------
# build_inference_rows
# ---------------------------------------------------------------------------
# Que hace: para cada entidad del universo financiero, toma el snapshot MAS
# RECIENTE disponible (sin ventana de lookback -- por definicion es el
# ultimo dato que existe) y arma una fila de inferencia con FECHA_TARGET =
# CORTE_OBJETIVO_FECHA. Luego EXCLUYE las entidades con un evento de
# intervencion SBS confirmado anterior a esa fecha base (punto 3.B del
# proyecto: no se predice el futuro de una entidad que ya quebro).
# Recibe: panel_feat (panel con lags), canon (universo de entidades),
# persistence_lookup (historico de ratings).
# Devuelve: (inference_df de entidades vigentes, df_excluidas por intervencion).
# ---------------------------------------------------------------------------
def build_inference_rows(panel_feat, canon, persistence_lookup):
    panel_by_entity = {eid: g.sort_values("FECHA") for eid, g in panel_feat.groupby("ENTITY_ID")}
    name_map = dict(zip(canon["ENTITY_ID"], canon["ENTITY_NAME"]))
    type_map = dict(zip(canon["ENTITY_ID"], canon["ENTITY_TYPE"]))

    rows = []
    sin_dato_financiero = []
    for entity_id, entity_panel in panel_by_entity.items():
        snapshot = entity_panel.iloc[-1]
        fecha_base = snapshot["FECHA"]
        if fecha_base >= CORTE_OBJETIVO_FECHA:
            sin_dato_financiero.append({"ENTITY_ID": entity_id, "MOTIVO": "FECHA_BASE >= FECHA_TARGET"})
            continue
        out_row = {
            "ENTITY_ID": entity_id, "ENTITY_NAME": name_map.get(entity_id, entity_id),
            "ENTITY_TYPE": type_map.get(entity_id), "FECHA_BASE": fecha_base, "FECHA_TARGET": CORTE_OBJETIVO_FECHA,
        }
        for col, val in snapshot.items():
            if col in ("ENTITY_ID", "FECHA", "ENTIDAD", "TIPO_DE_ENTIDAD", "SOURCE", "ENTITY_NAME_NORM"):
                continue
            out_row[col] = val
        rows.append(out_row)

    inference_df = pd.DataFrame(rows)
    inference_df["GAP_MESES"] = (
        (inference_df["FECHA_TARGET"].dt.year - inference_df["FECHA_BASE"].dt.year) * 12
        + (inference_df["FECHA_TARGET"].dt.month - inference_df["FECHA_BASE"].dt.month)
    )
    for ecr in config.ECR_LIST:
        inference_df[f"PREV_RATING_ORD__{ecr}"] = inference_df["ENTITY_ID"].apply(
            lambda eid: get_latest_known_rating(eid, ecr, persistence_lookup))

    if sin_dato_financiero:
        log(f"{len(sin_dato_financiero)} entidades excluidas de la inferencia "
            f"(sin dato financiero valido anterior al corte objetivo).", level="WARN")

    # Exclusion por intervencion SBS confirmada (punto 3.B del proyecto).
    inference_df, excluidas_intervencion = dataset_builder.excluir_posteriores_a_intervencion(inference_df)

    return inference_df, excluidas_intervencion


def predict_all_ecr(model, modelo2, pipeline_fe, scaler, inference_df, k_per_ecr_local, majority_fallback):
    X_infer_raw = pipeline_fe.transform(inference_df)
    X_infer = scaler.transform(X_infer_raw)
    etype_infer = inference_df["ENTITY_TYPE"].map(ENTITY_TYPE_TO_IDX).values
    prev_ratings = {ecr: inference_df[f"PREV_RATING_ORD__{ecr}"].values.astype(float) for ecr in config.ECR_LIST}
    return modelo2.decode_absolute_from_delta(
        model, X_infer, etype_infer, prev_ratings=prev_ratings,
        k_local=k_per_ecr_local, majority_fallback=majority_fallback,
    )


# ---------------------------------------------------------------------------
# build_output_table
# ---------------------------------------------------------------------------
# Que hace: arma la tabla final (una fila por entidad-ECR, solo para ECR
# que efectivamente calificaron antes a esa entidad) con las columnas que
# necesita la futura interfaz: entidad, tipo, ECR, clasificacion actual,
# clasificacion predicha, cambio, fechas, y la alerta de riesgo derivada
# del mismo cambio ordinal (ver docstring del modulo).
# Recibe: inference_df, preds (dict {ecr: array}), scales (escalas locales).
# Devuelve: DataFrame final, ordenado por tipo/entidad/ECR.
# ---------------------------------------------------------------------------
def build_output_table(inference_df, preds, scales):
    local_to_label = {ecr: {i: lab for i, lab in enumerate(scales[ecr])} for ecr in config.ECR_LIST}
    fecha_limite = eventos_intervencion.fecha_limite_por_entidad()
    from motor.utils import normalize_entity_name

    rows = []
    for i in range(len(inference_df)):
        r = inference_df.iloc[i]
        tuvo_intervencion_historica = normalize_entity_name(r["ENTITY_NAME"]) in fecha_limite
        for ecr in config.ECR_LIST:
            prev_ord = r[f"PREV_RATING_ORD__{ecr}"]
            if pd.isna(prev_ord):
                continue
            pred_ord = int(preds[ecr][i])
            cambio = pred_ord - int(prev_ord)
            rows.append({
                "ENTITY_NAME": r["ENTITY_NAME"], "ENTITY_TYPE": r["ENTITY_TYPE"], "ECR": ecr,
                "FECHA_BASE": r["FECHA_BASE"].date(), "FECHA_TARGET": r["FECHA_TARGET"].date(),
                "GAP_MESES": int(r["GAP_MESES"]),
                "RATING_ANTERIOR": local_to_label[ecr].get(int(prev_ord), int(prev_ord)),
                "RATING_PREDICHO": local_to_label[ecr].get(pred_ord, pred_ord),
                "CAMBIO_ORDINAL": cambio,
                "ALERTA_DOWNGRADE_FUERTE": cambio >= config.DOWNGRADE_THRESHOLD_NOTCHES,
                "ALERTA_INTERVENCION_HISTORICA": tuvo_intervencion_historica,
            })
    return pd.DataFrame(rows).sort_values(["ENTITY_TYPE", "ENTITY_NAME", "ECR"]).reset_index(drop=True)


def save_excels(output_df, excluidas_intervencion):
    os.makedirs(OUT_DIR, exist_ok=True)
    saved_paths = []

    consolidado_path = os.path.join(OUT_DIR, f"predicciones_{CORTE_OBJETIVO_NOMBRE}_TODAS.xlsx")
    output_df.to_excel(consolidado_path, index=False, sheet_name="PREDICCIONES")
    saved_paths.append(consolidado_path)

    for tipo in config.ENTITY_TYPES:
        sub = output_df[output_df["ENTITY_TYPE"] == tipo]
        if sub.empty:
            continue
        path = os.path.join(OUT_DIR, f"predicciones_{CORTE_OBJETIVO_NOMBRE}_{tipo}.xlsx")
        sub.to_excel(path, index=False, sheet_name=tipo)
        saved_paths.append(path)

    if len(excluidas_intervencion) > 0:
        path_excl = os.path.join(config.ANALISIS_DIR, f"entidades_excluidas_por_intervencion_{CORTE_OBJETIVO_NOMBRE}.csv")
        excluidas_intervencion.to_csv(path_excl, index=False, encoding="utf-8-sig")
        log(f"Entidades excluidas por intervencion SBS confirmada guardadas en: {path_excl} (NO van a la interfaz).")

    return saved_paths


def main():
    config.set_global_seed()
    section(f"PREDICCION DE CLASIFICACIONES FUTURAS -> {CORTE_OBJETIVO_NOMBRE} ({CORTE_OBJETIVO_FECHA.date()})")

    clas_valid, _, scales_full = pipeline_fases.load_classifications()
    sources, registry, canon, resolved, panel = pipeline_fases.load_financials_and_homologate(clas_valid)
    persistence_lookup = baselines_piso.build_persistence_lookup(resolved)

    section("FEATURE ENGINEERING SOBRE EL PANEL COMPLETO")
    panel_feat = feature_engineering.add_lag_features(panel)
    subsection("Rango de fechas disponible por tipo de entidad (fuente de verdad para el 'presente' de cada una)")
    merged_type = panel_feat.merge(canon[["ENTITY_ID", "ENTITY_TYPE"]], on="ENTITY_ID", how="left", suffixes=("", "_c"))
    type_col = "ENTITY_TYPE_c" if "ENTITY_TYPE_c" in merged_type.columns else "ENTITY_TYPE"
    for tipo, g in merged_type.groupby(type_col):
        print(f"  {tipo:12s}: hasta {g['FECHA'].max().date()}")

    supervised_df, sin_datos_df = dataset_builder.build_supervised_dataset(resolved, panel_feat, canon, persistence_lookup)
    bad = supervised_df[supervised_df["FECHA_BASE"] >= supervised_df["FECHA_TARGET"]]
    if len(bad) > 0:
        raise RuntimeError(f"ERROR CRITICO DE INTEGRACION TEMPORAL en datos historicos: {len(bad)} filas invalidas.")
    log("Chequeo de integridad temporal (historico): OK.")

    train_df, val_df = build_production_split(supervised_df)
    model, pipeline_fe, scaler, k_per_ecr_local, majority_fallback, modelo2 = train_production_model(
        train_df, val_df, scales_full)

    section("CONSTRUCCION DE FILAS DE INFERENCIA (una por entidad, fecha base = ultimo dato disponible)")
    inference_df, excluidas_intervencion = build_inference_rows(panel_feat, canon, persistence_lookup)
    log(f"Entidades con fila de inferencia construida: {len(inference_df)}")
    subsection("Gap en meses (FECHA_BASE -> FECHA_TARGET) por tipo de entidad")
    print(inference_df.groupby("ENTITY_TYPE")["GAP_MESES"].agg(["min", "max", "mean"]).round(1).to_string())

    preds = predict_all_ecr(model, modelo2, pipeline_fe, scaler, inference_df, k_per_ecr_local, majority_fallback)
    output_df = build_output_table(inference_df, preds, scales_full)

    section(f"PREDICCIONES {CORTE_OBJETIVO_NOMBRE} (sin ground truth: corte no publicado por la SBS todavia)")
    print(output_df.to_string(index=False))
    n_alertas = int(output_df["ALERTA_DOWNGRADE_FUERTE"].sum())
    log(f"Filas con ALERTA_DOWNGRADE_FUERTE=True: {n_alertas}")

    saved_paths = save_excels(output_df, excluidas_intervencion)
    section("ARCHIVOS GENERADOS (outputs/interfaz/)")
    for p in saved_paths:
        log(f"Guardado: {p}")

    return output_df, saved_paths


if __name__ == "__main__":
    main()
