# -*- coding: utf-8 -*-
"""
scripts/generar_dataset_variables.py
=====================================
Genera la carpeta dataset_variables/<tipo_entidad>/ con lo que el motor
REALMENTE usa para entrenar cada tipo de entidad (punto 11 del proyecto):

    dataset_variables/
    ├── banco/
    │   ├── variables_entrenamiento.xlsx   (variables seleccionadas + metadata)
    │   ├── variables_descripcion.xlsx     (diccionario de datos)
    │   └── resumen_variables.txt
    ├── cmac/  crac/  financiera/  (idem)

IMPORTANTE: esto NO es una lista teorica de todas las columnas que existen
en los archivos fuente -- es el resultado de correr, POR TIPO DE ENTIDAD,
el mismo FeaturePipeline que usa el motor (motor/features/selection.py)
sobre el split de TRAIN real, y registrar exactamente que selecciono.

Ejecucion:
    python scripts/generar_dataset_variables.py
"""

from __future__ import annotations
import os
import re
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor import config
from motor.utils import log, section
from motor.features import selection
from motor.evaluacion import validation


# ---------------------------------------------------------------------------
# _parse_variable_name
# ---------------------------------------------------------------------------
# Que hace: a partir del nombre final de una variable (tal como sale de
# feature_engineering.py / selection.py), reconstruye de forma
# PROGRAMATICA (no manual) sus metadatos: variable original de la que
# deriva, si es derivada o no, que transformacion se le aplico y sobre que
# ventana temporal (lag en meses), para no tener que mantener a mano una
# tabla separada que se desactualiza cada vez que cambian las columnas.
# Recibe: nombre de columna final (string).
# Devuelve: dict con nombre_original, es_derivada, transformacion, periodo.
# ---------------------------------------------------------------------------
def _parse_variable_name(col: str) -> dict:
    m = re.match(r"^(.*)__WASNULL$", col)
    if m:
        return {"nombre_original": m.group(1), "es_derivada": True,
                "transformacion": "indicador_missing", "periodo_meses": None}

    m = re.match(r"^(.*)_DELTA_(\d+)M$", col)
    if m:
        return {"nombre_original": m.group(1), "es_derivada": True,
                "transformacion": "delta_absoluto", "periodo_meses": int(m.group(2))}

    m = re.match(r"^(.*)_PCTCHG_(\d+)M$", col)
    if m:
        return {"nombre_original": m.group(1), "es_derivada": True,
                "transformacion": "variacion_porcentual", "periodo_meses": int(m.group(2))}

    m = re.match(r"^(.*)_TREND_(\d+)M$", col)
    if m:
        return {"nombre_original": m.group(1), "es_derivada": True,
                "transformacion": "tendencia_pendiente", "periodo_meses": int(m.group(2))}

    if col.startswith("PREV_RATING_ORD__"):
        return {"nombre_original": col.replace("PREV_RATING_ORD__", ""), "es_derivada": True,
                "transformacion": "rating_ordinal_anterior (lag_1_corte)", "periodo_meses": None}

    if col.endswith("_MN"):
        return {"nombre_original": col[:-3], "es_derivada": False,
                "transformacion": "nivel (moneda nacional)", "periodo_meses": None}
    if col.endswith("_ME"):
        return {"nombre_original": col[:-3], "es_derivada": False,
                "transformacion": "nivel (moneda extranjera)", "periodo_meses": None}

    return {"nombre_original": col, "es_derivada": False,
            "transformacion": "nivel (valor contemporaneo)", "periodo_meses": None}


# ---------------------------------------------------------------------------
# generar_para_tipo
# ---------------------------------------------------------------------------
# Que hace: filtra el dataset supervisado a UN tipo de entidad, ajusta un
# FeaturePipeline especificamente sobre ese subconjunto (no reutiliza la
# seleccion global -- cada tipo puede necesitar variables distintas) y
# escribe los 3 archivos de esa carpeta.
# Recibe: supervised_df completo, tipo (string, p.ej. "BANCO"),
# carpeta_salida (ruta de dataset_variables/<tipo_normalizado>/).
# Devuelve: None (efecto: escribe los 3 archivos en disco).
# ---------------------------------------------------------------------------
def generar_para_tipo(supervised_df: pd.DataFrame, tipo: str, carpeta_salida: str) -> None:
    train_df = supervised_df[
        (supervised_df["ENTITY_TYPE"] == tipo) &
        (supervised_df["CORTE"].isin(validation.FINAL_TRAIN_CORTES))
    ].copy()

    n_entidades = train_df["ENTITY_ID"].nunique()
    n_obs = len(train_df)
    log(f"[{tipo}] TRAIN: {n_obs} observaciones, {n_entidades} entidades.")

    if n_obs < 15:
        log(f"[{tipo}] Muestra insuficiente ({n_obs} filas) para una seleccion de variables "
            f"propia; se omite la generacion de dataset_variables para este tipo.", level="WARN")
        with open(os.path.join(carpeta_salida, "resumen_variables.txt"), "w", encoding="utf-8") as f:
            f.write(f"Tipo de entidad: {tipo}\n")
            f.write(f"Observaciones en TRAIN: {n_obs} (insuficiente, < 15)\n")
            f.write("No se genero seleccion de variables propia por este tipo: la muestra es "
                    "demasiado chica para que la importancia de variables sea confiable.\n")
        return

    pipeline = selection.FeaturePipeline()
    pipeline.fit(train_df, target_cols=[f"TARGET_ORD__{e}" for e in config.ECR_LIST])

    # --- variables_entrenamiento.xlsx: lo que REALMENTE entra al modelo ---
    filas = []
    for var in pipeline.selected_features_:
        meta = _parse_variable_name(var)
        importancia_por_ecr = {
            ecr: float(imp.get(var, 0.0)) for ecr, imp in pipeline.feature_importance_by_ecr_.items()
        }
        importancia_prom = (sum(importancia_por_ecr.values()) / len(importancia_por_ecr)) if importancia_por_ecr else 0.0
        modelos_donde_participa = sorted([ecr for ecr, v in importancia_por_ecr.items() if v > 0])
        filas.append({
            "VARIABLE": var,
            "NOMBRE_ORIGINAL_DATASET": meta["nombre_original"],
            "ES_DERIVADA": meta["es_derivada"],
            "TRANSFORMACION": meta["transformacion"],
            "PERIODO_MESES": meta["periodo_meses"],
            "USADA_COMO_PREDICTOR": True,
            "USADA_COMO_TARGET": False,
            "IMPORTANCIA_PROMEDIO_RF": round(importancia_prom, 5),
            "ECR_DONDE_APORTA": ", ".join(modelos_donde_participa) if modelos_donde_participa else "(ninguna con importancia > 0)",
        })
    # Las columnas TARGET_ORD__<ECR> no son features, pero se documentan
    # explicitamente como variables objetivo para que la tabla sea completa.
    for ecr in config.ECR_LIST:
        filas.append({
            "VARIABLE": f"TARGET_ORD__{ecr}", "NOMBRE_ORIGINAL_DATASET": f"Rating publicado por {ecr}",
            "ES_DERIVADA": True, "TRANSFORMACION": "rating_ordinal (variable objetivo)",
            "PERIODO_MESES": None, "USADA_COMO_PREDICTOR": False, "USADA_COMO_TARGET": True,
            "IMPORTANCIA_PROMEDIO_RF": None, "ECR_DONDE_APORTA": ecr,
        })

    df_vars = pd.DataFrame(filas)
    df_vars.to_excel(os.path.join(carpeta_salida, "variables_entrenamiento.xlsx"), index=False)

    # --- variables_descripcion.xlsx: diccionario de datos ---
    descripciones = []
    vistos = set()
    for _, row in df_vars.iterrows():
        base = row["NOMBRE_ORIGINAL_DATASET"]
        if base in vistos:
            continue
        vistos.add(base)
        descripciones.append({
            "NOMBRE_ORIGINAL": base,
            "TIPO_DATO": "numerico" if row["USADA_COMO_TARGET"] is False or True else "ordinal",
            "DESCRIPCION": (
                f"Partida financiera del dataset fuente de {tipo.lower()}, usada como base de "
                f"variables derivadas (lags/tendencias)." if row["ES_DERIVADA"] and row["USADA_COMO_PREDICTOR"]
                else f"Rating ordinal publicado por la ECR {base}." if row["USADA_COMO_TARGET"]
                else f"Variable financiera original del dataset fuente de {tipo.lower()}."
            ),
        })
    pd.DataFrame(descripciones).to_excel(os.path.join(carpeta_salida, "variables_descripcion.xlsx"), index=False)

    # --- resumen_variables.txt ---
    n_derivadas = int(df_vars["ES_DERIVADA"][df_vars["USADA_COMO_PREDICTOR"]].sum())
    n_originales = int((~df_vars["ES_DERIVADA"])[df_vars["USADA_COMO_PREDICTOR"]].sum())
    with open(os.path.join(carpeta_salida, "resumen_variables.txt"), "w", encoding="utf-8") as f:
        f.write(f"Tipo de entidad: {tipo}\n")
        f.write(f"Observaciones en TRAIN usadas para la seleccion: {n_obs} ({n_entidades} entidades)\n")
        f.write(f"Variables candidatas iniciales: {len(pipeline.candidate_columns_)}\n")
        f.write(f"Variables finales seleccionadas como predictoras: {len(pipeline.selected_features_)}\n")
        f.write(f"  - De nivel/original: {n_originales}\n")
        f.write(f"  - Derivadas (lag/tendencia/indicador missing): {n_derivadas}\n")
        f.write(f"Variables objetivo (rating por ECR): {len(config.ECR_LIST)}\n")
        f.write("\nVer variables_entrenamiento.xlsx para el detalle variable por variable "
                "y variables_descripcion.xlsx para el diccionario de datos.\n")

    log(f"[{tipo}] dataset_variables generado en {carpeta_salida} "
        f"({len(pipeline.selected_features_)} variables predictoras).")


def main():
    section("GENERACION DE dataset_variables/ POR TIPO DE ENTIDAD")
    # Importa aqui (no al tope del archivo) las fases de carga para no
    # duplicar el modulo run_evaluacion como dependencia dura si algun dia
    # se llama este script de forma completamente independiente.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_evaluacion as pipeline_fases

    pipeline_fases.config.set_global_seed()
    clas_valid, _, _ = pipeline_fases.load_classifications()
    sources, registry, canon, resolved, panel = pipeline_fases.load_financials_and_homologate(clas_valid)
    from motor.modelos.comparacion import baselines_piso
    persistence_lookup = baselines_piso.build_persistence_lookup(resolved)
    supervised_df, _, _, _ = pipeline_fases.build_supervised_dataset(resolved, canon, panel, persistence_lookup)

    tipo_a_carpeta = {"BANCO": "banco", "CMAC": "cmac", "CRAC": "crac", "FINANCIERA": "financiera"}
    tipos_presentes = sorted(supervised_df["ENTITY_TYPE"].dropna().unique())
    for tipo in tipos_presentes:
        carpeta = tipo_a_carpeta.get(tipo, tipo.lower())
        carpeta_salida = os.path.join(config.DATASET_VARIABLES_DIR, carpeta)
        os.makedirs(carpeta_salida, exist_ok=True)
        generar_para_tipo(supervised_df, tipo, carpeta_salida)

    log(f"Listo. Carpetas generadas en {config.DATASET_VARIABLES_DIR}/ para: {tipos_presentes}")


if __name__ == "__main__":
    main()
