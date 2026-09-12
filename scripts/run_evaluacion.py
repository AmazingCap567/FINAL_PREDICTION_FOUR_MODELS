# -*- coding: utf-8 -*-
"""
scripts/run_evaluacion.py
==========================
Punto de entrada de EVALUACION del proyecto (ex main.py). Ejecuta el
pipeline completo en modo "conozco la respuesta correcta" (holdout con
ground truth ya publicado por las ECR), para poder medir que tan bien
funciona cada uno de los 4 modelos comparables antes de usarlos para
predecir el futuro de verdad (eso lo hace scripts/run_prediccion.py).

    diagnostico -> carga -> homologacion -> integracion temporal ->
    feature engineering -> seleccion -> variables de riesgo ->
    4 modelos comparables + baselines de piso -> seleccion del ganador ->
    validacion walk-forward -> explicabilidad -> salidas (analisis + consola)

Ejecucion:
    python scripts/run_evaluacion.py [--verbose]

Requiere que los datos originales esten en ./data/ (ver README.md).
Holdout de evaluacion: el corte mas reciente configurado en
motor.config.CORTES_CLASIFICACION (actualmente MAR2026).
"""

from __future__ import annotations
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor import config
from motor.utils import log, section, subsection
from motor.datos import sbs_loader, entity_homology, financial_loader, ratings, dataset_builder, eventos_intervencion
from motor.features import feature_engineering, selection
from motor.evaluacion import validation, metrics as metrics_mod, explainability, report
from motor.modelos.comparacion import baselines_piso, seleccion
from motor.modelos.modelo_1_hist_gradient_boosting import modelo as modelo1
from motor.modelos.modelo_3_regresion_ordinal import modelo as modelo3
from motor.modelos.modelo_4_lightgbm import modelo as modelo4
# El modelo 2 (MLP ordinal, PyTorch) se importa de forma perezosa dentro de
# train_all_models() para que el resto del pipeline (incluido este script)
# siga siendo utilizable en un entorno sin PyTorch instalado.


ENTITY_TYPE_TO_IDX = {t: i for i, t in enumerate(config.ENTITY_TYPES)}
OUT_ANALISIS = config.ANALISIS_DIR


# =============================================================================
# FASE 0-5: identicas al pipeline original (carga, homologacion, dataset
# supervisado, seleccion de variables) — ver motor/datos y motor/features
# para el detalle de cada paso; aqui solo se orquesta el orden.
# =============================================================================
def run_diagnostics():
    section("FASE 0: DIAGNOSTICO INICIAL DE ARCHIVOS")
    for name, path in [
        ("Bancos", config.FILE_BANCOS), ("Caja Municipal", config.FILE_CAJA_MUNICIPAL),
        ("Financieras", config.FILE_FINANCIERAS), ("CRAC", config.FILE_CRAC),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontro el archivo requerido: {path}")
        log(f"Archivo encontrado: {name} -> {path} ({os.path.getsize(path)/1024:.1f} KB)")
    for corte in config.CORTES_CLASIFICACION:
        p = os.path.join(config.CLASIFICACIONES_DIR, f"{corte}.xls")
        if not os.path.exists(p):
            raise FileNotFoundError(f"No se encontro el archivo de clasificacion: {p}")
    log(f"Los {len(config.CORTES_CLASIFICACION)} archivos de clasificacion SBS estan presentes.")


def load_classifications():
    section("FASE 1: CARGA Y NORMALIZACION DE CLASIFICACIONES SBS")
    clas_raw = sbs_loader.load_all_sbs_classifications()
    log(f"Total de registros (entidad, ECR, corte) con valor no nulo: {len(clas_raw)}")
    clas_annotated = ratings.annotate_classifications(clas_raw)
    n_invalid = clas_annotated["rating_label"].isna().sum()
    filas_sin_clasificacion_valida = clas_annotated[clas_annotated["rating_label"].isna()].copy()
    scales = ratings.build_ecr_scales(clas_annotated)
    clas_local = ratings.add_local_ordinal(clas_annotated, scales)
    clas_valid = clas_local[clas_local["rating_label"].notna()].copy()
    log(f"Registros validos tras limpieza de rating: {len(clas_valid)} (excluidos: {n_invalid})")
    return clas_valid, filas_sin_clasificacion_valida, scales


def load_financials_and_homologate(clas_valid: pd.DataFrame):
    section("FASE 2: CARGA DE DATASETS FINANCIEROS Y HOMOLOGACION DE ENTIDADES")
    sources = financial_loader.load_all_financial_sources()
    registry = entity_homology.EntityRegistry()
    canon = registry.build_from_financial_sources(sources)
    log(f"Universo canonico de entidades (fuente=datasets financieros): {len(canon)}")
    subsection("Entidades por tipo")
    print(canon["ENTITY_TYPE"].value_counts().to_string())
    resolved = dataset_builder.build_entity_universe_and_resolve(clas_valid, registry)
    n_no_homologadas = len(pd.DataFrame(registry.no_homologadas).drop_duplicates())
    log(f"Pares (entidad SBS, corte) resueltos correctamente: {len(resolved)}")
    log(f"Entradas distintas de entidad no homologada / excluida: {n_no_homologadas}")
    registry.save_no_homologadas(os.path.join(config.AUDIT_DIR, "entidades_no_homologadas.csv"))
    entity_id_map = dict(zip(canon["ENTITY_NAME_NORM"], canon["ENTITY_ID"]))
    panel = financial_loader.build_unified_panel(sources, entity_id_map)
    log(f"Panel financiero unificado: {panel.shape[0]} filas x {panel.shape[1]} columnas.")
    return sources, registry, canon, resolved, panel


def build_supervised_dataset(resolved, canon, panel, persistence_lookup):
    section("FASE 3: FEATURE ENGINEERING, INTEGRACION TEMPORAL Y VARIABLES DE RIESGO")
    n_cols_raw = panel.shape[1]
    panel_feat = feature_engineering.add_lag_features(panel)
    n_cols_after_fe = panel_feat.shape[1]

    supervised_df, sin_datos_df = dataset_builder.build_supervised_dataset(
        resolved, panel_feat, canon, persistence_lookup)

    bad = supervised_df[supervised_df["FECHA_BASE"] >= supervised_df["FECHA_TARGET"]]
    if len(bad) > 0:
        raise RuntimeError(
            f"ERROR CRITICO DE INTEGRACION TEMPORAL: {len(bad)} filas tienen FECHA_BASE >= FECHA_TARGET. "
            f"Esto indicaria fuga de informacion futura. Deteniendo ejecucion."
        )
    log("Chequeo de integridad temporal: OK (FECHA_BASE < FECHA_TARGET en el 100% de las filas).")

    # Variables del objetivo de riesgo/deterioro (adicional al rating).
    supervised_df = dataset_builder.agregar_variables_riesgo(supervised_df)
    n_downgrade = int(supervised_df["DOWNGRADE_FUERTE"].sum())
    n_intervencion = int(supervised_df["EVENTO_INTERVENCION_PREVIA"].sum())
    log(f"Filas con downgrade fuerte (>= {config.DOWNGRADE_THRESHOLD_NOTCHES} notches en alguna ECR): {n_downgrade}")
    log(f"Filas posteriores a un evento de intervencion SBS confirmado: {n_intervencion}")

    sin_datos_df.to_csv(os.path.join(config.AUDIT_DIR, "filas_sin_dato_financiero.csv"), index=False, encoding="utf-8-sig")
    return supervised_df, panel_feat, n_cols_raw, n_cols_after_fe


def print_pretrain_diagnostics(supervised_df, n_cols_raw, n_cols_after_fe, scales):
    section("DIAGNOSTICO PRE-ENTRENAMIENTO")
    print(f"Dimensiones del dataset supervisado final: {supervised_df.shape[0]} filas x {supervised_df.shape[1]} columnas")
    print(f"Entidades unicas: {supervised_df['ENTITY_ID'].nunique()}")
    print(f"Variables financieras iniciales (panel crudo): {n_cols_raw}")
    print(f"Variables tras feature engineering (antes de seleccion): {n_cols_after_fe}")
    subsection("Numero de targets validos por ECR")
    for ecr in config.ECR_LIST:
        n = supervised_df[f"TARGET_ORD__{ecr}"].notna().sum()
        print(f"  {ecr:12s}: {n:4d} observaciones validas | escala local: {scales[ecr]}")


def prepare_splits_and_features(supervised_df, scales):
    section("FASE 5: SPLIT TEMPORAL Y SELECCION DE VARIABLES (SOLO EN TRAIN)")
    train_df = supervised_df[supervised_df["CORTE"].isin(validation.FINAL_TRAIN_CORTES)].copy()
    val_df = supervised_df[supervised_df["CORTE"] == validation.FINAL_VAL_CORTE].copy()
    holdout_df = supervised_df[supervised_df["CORTE"] == validation.HOLDOUT_CORTE].copy()
    log(f"Train final: {len(train_df)} filas | cortes = {validation.FINAL_TRAIN_CORTES}")
    log(f"Validacion (early stopping MLP): {len(val_df)} filas | corte = {validation.FINAL_VAL_CORTE}")
    log(f"Holdout final (fuera de muestra): {len(holdout_df)} filas | corte = {validation.HOLDOUT_CORTE}")

    pipeline = selection.FeaturePipeline()
    pipeline.fit(train_df, target_cols=[f"TARGET_ORD__{e}" for e in config.ECR_LIST])
    X_train_raw = pipeline.transform(train_df)
    X_val_raw = pipeline.transform(val_df)
    X_holdout_raw = pipeline.transform(holdout_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_holdout = scaler.transform(X_holdout_raw)
    # Version DataFrame (con nombres de columna) para los modelos 1/3/4, que
    # no necesitan estandarizacion previa de HistGB/LightGBM mas alla de la
    # que ya hace su propio pipeline (modelo 3 si estandariza internamente).
    X_train_df = pd.DataFrame(X_train_raw.values, columns=X_train_raw.columns)
    X_holdout_df = pd.DataFrame(X_holdout_raw.values, columns=X_holdout_raw.columns)

    etype_train = train_df["ENTITY_TYPE"].map(ENTITY_TYPE_TO_IDX).values
    etype_val = val_df["ENTITY_TYPE"].map(ENTITY_TYPE_TO_IDX).values
    etype_holdout = holdout_df["ENTITY_TYPE"].map(ENTITY_TYPE_TO_IDX).values

    y_train = {ecr: train_df[f"TARGET_ORD__{ecr}"].values.astype(float) for ecr in config.ECR_LIST}
    y_holdout = {ecr: holdout_df[f"TARGET_ORD__{ecr}"].values.astype(float) for ecr in config.ECR_LIST}

    def _encode_delta(series):
        clipped = np.clip(series.values.astype(float), -config.MAX_DELTA_CLIP, config.MAX_DELTA_CLIP)
        return clipped + config.DELTA_OFFSET

    y_train_delta = {ecr: _encode_delta(train_df[f"DELTA_ORD__{ecr}"]) for ecr in config.ECR_LIST}
    y_val_delta = {ecr: _encode_delta(val_df[f"DELTA_ORD__{ecr}"]) for ecr in config.ECR_LIST}
    prev_rating_holdout = {ecr: holdout_df[f"PREV_RATING_ORD__{ecr}"].values.astype(float) for ecr in config.ECR_LIST}
    k_per_ecr = {ecr: max(len(scales[ecr]), 2) for ecr in config.ECR_LIST}

    return {
        "train_df": train_df, "val_df": val_df, "holdout_df": holdout_df,
        "X_train": X_train, "X_val": X_val, "X_holdout": X_holdout,
        "X_train_df": X_train_df, "X_holdout_df": X_holdout_df,
        "etype_train": etype_train, "etype_val": etype_val, "etype_holdout": etype_holdout,
        "y_train": y_train, "y_holdout": y_holdout,
        "y_train_delta": y_train_delta, "y_val_delta": y_val_delta,
        "prev_rating_holdout": prev_rating_holdout,
        "pipeline": pipeline, "scaler": scaler, "k_per_ecr": k_per_ecr,
    }


# =============================================================================
# FASE 6: LOS 4 MODELOS COMPARABLES + BASELINES DE PISO
# =============================================================================
# Que hace: entrena, por ECR, los 2 baselines de piso (mayoria, persistencia,
# que NO cuentan como modelos comparables) y los 4 modelos reales (HistGB,
# MLP ordinal multi-task, Regresion Logistica Ordinal, LightGBM), y devuelve
# las predicciones de holdout de cada uno.
# Nota importante: el modelo 2 (MLP) es estructuralmente distinto a los
# otros 3 -- es multi-tarea (comparte capas entre las 6 ECR) y predice
# DELTA de rating, no el nivel absoluto directamente -- por eso se entrena
# una sola vez para todas las ECR, mientras que 1/3/4 se entrenan un
# modelo independiente POR ECR. Ambos enfoques son validos; se documentan
# las diferencias en vez de forzar una interfaz identica que no encaja.
# =============================================================================
def train_all_models(data, persistence_lookup):
    section("FASE 6: ENTRENAMIENTO DE BASELINES DE PISO + LOS 4 MODELOS COMPARABLES")
    train_df, holdout_df = data["train_df"], data["holdout_df"]

    preds_holdout = {"Mayoria": {}, "Persistencia": {}, "1-HistGB": {}, "3-RegresionOrdinal": {}, "4-LightGBM": {}}
    fitted_models = {"1-HistGB": {}, "3-RegresionOrdinal": {}, "4-LightGBM": {}}

    for ecr in config.ECR_LIST:
        y_tr = train_df[f"TARGET_ORD__{ecr}"]
        mask_tr = y_tr.notna()

        maj = baselines_piso.majority_class_baseline(train_df, ecr)
        preds_holdout["Mayoria"][ecr] = np.full(len(holdout_df), maj)
        persist_preds = holdout_df.apply(
            lambda row: baselines_piso.predict_persistence(row, ecr, persistence_lookup), axis=1)
        preds_holdout["Persistencia"][ecr] = persist_preds.values

        if mask_tr.sum() >= 10 and y_tr.loc[mask_tr].nunique() > 1:
            Xtr = data["X_train_df"].loc[mask_tr.values].reset_index(drop=True)
            ytr = y_tr.loc[mask_tr].reset_index(drop=True)
            for nombre, modulo in [("1-HistGB", modelo1), ("3-RegresionOrdinal", modelo3), ("4-LightGBM", modelo4)]:
                m = modulo.entrenar(Xtr, ytr)
                fitted_models[nombre][ecr] = m
                preds_holdout[nombre][ecr] = modulo.predecir(m, data["X_holdout_df"])
        else:
            log(f"ECR {ecr}: TRAIN insuficiente ({int(mask_tr.sum())} filas) para los modelos 1/3/4, "
                f"se usa clase mayoritaria como respaldo.", level="WARN")
            for nombre in ("1-HistGB", "3-RegresionOrdinal", "4-LightGBM"):
                preds_holdout[nombre][ecr] = np.full(len(holdout_df), maj)

    log("Entrenando modelo 2 (MLP ordinal multi-task, target = DELTA de rating)...")
    from motor.modelos.modelo_2_mlp_ordinal import modelo as modelo2
    delta_k_per_ecr = {ecr: config.N_DELTA_CLASSES for ecr in config.ECR_LIST}
    model2, history = modelo2.train_model(
        data["X_train"], data["etype_train"], data["y_train_delta"],
        data["X_val"], data["etype_val"], data["y_val_delta"],
        n_entity_types=len(config.ENTITY_TYPES), k_per_ecr=delta_k_per_ecr,
    )
    majority_fallback = {ecr: baselines_piso.majority_class_baseline(train_df, ecr) for ecr in config.ECR_LIST}
    preds_holdout["2-MLPOrdinal"] = modelo2.decode_absolute_from_delta(
        model2, data["X_holdout"], data["etype_holdout"],
        prev_ratings=data["prev_rating_holdout"], k_local=data["k_per_ecr"], majority_fallback=majority_fallback,
    )

    return preds_holdout, fitted_models, model2, history


# =============================================================================
# FASE 7: EVALUACION HOLDOUT + SELECCION DEL GANADOR (por ECR)
# =============================================================================
def evaluate_and_select(data, preds_holdout):
    section("FASE 7: EVALUACION EN HOLDOUT Y SELECCION DEL MODELO GANADOR POR ECR")
    holdout_df = data["holdout_df"]
    y_holdout = data["y_holdout"]
    nombres_modelos_comparables = ["1-HistGB", "2-MLPOrdinal", "3-RegresionOrdinal", "4-LightGBM"]

    resumen_ganadores = {}
    per_ecr_all_models = {}

    for ecr in config.ECR_LIST:
        y_true_full = y_holdout[ecr]
        mask = ~np.isnan(y_true_full)
        if mask.sum() == 0:
            continue

        model_metrics = {}
        for nombre in ["Mayoria", "Persistencia"] + nombres_modelos_comparables:
            preds_full = np.asarray(preds_holdout[nombre][ecr], dtype=float)
            row_mask = mask & ~np.isnan(preds_full)
            if row_mask.sum() == 0:
                continue
            m = metrics_mod.compute_metrics(y_holdout[ecr][row_mask].astype(int), preds_full[row_mask].astype(int))
            model_metrics[nombre] = m

        per_ecr_all_models[ecr] = metrics_mod.metrics_table(model_metrics)
        subsection(f"Metricas holdout — {ecr}")
        print(per_ecr_all_models[ecr].to_string())

        # Ranking y ganador SOLO entre los 4 modelos comparables (no los
        # baselines de piso, que existen para comparar, no para competir).
        tabla_comparables = pd.DataFrame({
            "weighted_kappa": [model_metrics[n]["weighted_kappa"] for n in nombres_modelos_comparables if n in model_metrics],
            "mae_ordinal": [model_metrics[n]["ordinal_mae"] for n in nombres_modelos_comparables if n in model_metrics],
            "f1_macro": [model_metrics[n].get("f1_macro", np.nan) for n in nombres_modelos_comparables if n in model_metrics],
        }, index=[n for n in nombres_modelos_comparables if n in model_metrics])

        if len(tabla_comparables) > 0:
            ranking, ganador = seleccion.construir_ranking(tabla_comparables)
            resumen_ganadores[ecr] = ganador
            print(seleccion.resumen_texto(ranking, ganador))

    return per_ecr_all_models, resumen_ganadores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Muestra tablas detalladas adicionales de auditoria.")
    args = parser.parse_args()

    config.set_global_seed()
    run_diagnostics()

    clas_valid, filas_sin_clasificacion_valida, scales_full = load_classifications()
    sources, registry, canon, resolved, panel = load_financials_and_homologate(clas_valid)
    persistence_lookup = baselines_piso.build_persistence_lookup(resolved)

    supervised_df, panel_feat, n_cols_raw, n_cols_after_fe = build_supervised_dataset(
        resolved, canon, panel, persistence_lookup)
    scales = scales_full

    print_pretrain_diagnostics(supervised_df, n_cols_raw, n_cols_after_fe, scales)
    data = prepare_splits_and_features(supervised_df, scales)

    preds_holdout, fitted_models, model2, history = train_all_models(data, persistence_lookup)
    per_ecr_all_models, resumen_ganadores = evaluate_and_select(data, preds_holdout)

    section("RESUMEN: MODELO GANADOR POR ECR")
    for ecr, ganador in resumen_ganadores.items():
        print(f"  {ecr:12s}: {ganador}")

    # Salidas de auditoria/analisis (NO son las salidas de la interfaz --
    # esas las genera scripts/run_prediccion.py en outputs/interfaz/).
    os.makedirs(OUT_ANALISIS, exist_ok=True)
    for ecr, tabla in per_ecr_all_models.items():
        tabla.to_csv(os.path.join(OUT_ANALISIS, f"metricas_holdout_{ecr}.csv"), encoding="utf-8-sig")
    pd.DataFrame([{"ECR": e, "MODELO_GANADOR": g} for e, g in resumen_ganadores.items()]).to_csv(
        os.path.join(OUT_ANALISIS, "modelo_ganador_por_ecr.csv"), index=False, encoding="utf-8-sig")

    log(f"Evaluacion completa. Metricas y ganadores guardados en {OUT_ANALISIS}/")
    log(f"Holdout evaluado: corte {validation.HOLDOUT_CORTE}.")


if __name__ == "__main__":
    main()
