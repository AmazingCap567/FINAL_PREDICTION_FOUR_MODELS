# -*- coding: utf-8 -*-
"""
selection.py
============
Seleccion y preparacion de variables predictoras.

REGLA DE ORO (se cumple estrictamente en todo este modulo):
    Toda operacion que "aprende" de los datos (que columnas son casi
    constantes, medianas de imputacion, pares correlacionados, importancia
    de variables) se ajusta UNICAMENTE sobre el conjunto de TRAIN que se le
    pasa a `FeaturePipeline.fit`. `transform` se usa despues, sin refit,
    sobre validacion/test/holdout.

Pasos (en orden):
  1. Excluir columnas identificadoras, de texto y de target.
  2. Eliminar columnas casi-constantes (proporcion del valor mas frecuente
     >= NEAR_CONSTANT_THRESHOLD, calculado sobre TRAIN).
  3. Crear indicadores de missing para columnas con NaN en TRAIN (antes de
     imputar), y respetar los indicadores *_MISSING ya existentes en los
     datasets originales (no se duplican, no se sobreescriben).
  4. Imputar con la mediana de TRAIN (nunca con 0 por defecto).
  5. Eliminar variables redundantes por alta correlacion (> CORR_THRESHOLD),
     ajustado sobre TRAIN.
  6. Seleccionar el subconjunto final de variables mediante importancia de
     un RandomForest ligero entrenado en TRAIN (uno por cada ECR, y se toma
     la union para no perder senal especifica de una agencia), respetando
     el limite MAX_SELECTED_FEATURES.

No se usa PCA (para conservar interpretabilidad), tal como pide el enunciado.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .. import config
from ..utils import log


class FeaturePipeline:
    def __init__(self):
        self.candidate_columns_: list[str] = []
        self.near_constant_dropped_: list[str] = []
        self.missing_indicator_columns_: list[str] = []
        self.medians_: pd.Series | None = None
        self.correlation_dropped_: list[str] = []
        self.selected_features_: list[str] = []
        self.feature_importance_by_ecr_: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # _list_candidate_columns
    # ------------------------------------------------------------------
    # Que hace: arma la lista inicial de columnas candidatas a ser
    # feature: todas las numericas del dataset supervisado EXCEPTO
    # identificadores (ENTITY_ID, FECHA_*, etc.) y cualquier columna
    # derivada directamente del target (ver nota sobre DELTA_ORD__ abajo,
    # es el guardian anti-fuga mas importante de este modulo).
    # Recibe: df (dataset supervisado, tipicamente TRAIN).
    # Devuelve: lista de nombres de columna candidatas.
    # ------------------------------------------------------------------
    def _list_candidate_columns(self, df: pd.DataFrame) -> list[str]:
        # DELTA_ORD__<ECR> se excluye explicitamente: es TARGET_ORD - PREV_RATING_ORD,
        # es decir, una funcion casi directa del target. Incluirlo como feature
        # seria fuga de informacion (el modelo "vería" la respuesta). Se calcula
        # unicamente como variable auxiliar para construir el target de delta,
        # nunca como input del modelo.
        # PREV_RATING_ORD__<ECR> SI se mantiene como feature: es informacion
        # legitima disponible antes del corte objetivo (el ultimo rating publicado).
        exclude_prefixes = ("TARGET_ORD__", "TARGET_LABEL__", "DELTA_ORD__")
        exclude_exact = set(config.ID_LIKE_COLUMNS) | {
            "ENTITY_ID", "ENTITY_NAME", "ENTITY_TYPE", "CORTE",
            "FECHA_BASE", "FECHA_TARGET", "ENTIDAD", "TIPO_DE_ENTIDAD",
        }
        cols = []
        for c in df.columns:
            if c in exclude_exact:
                continue
            if any(c.startswith(p) for p in exclude_prefixes):
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            cols.append(c)
        return cols

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    # Que hace: ejecuta los 6 pasos documentados al inicio del archivo
    # (candidatas -> casi-constantes -> indicadores missing -> imputacion
    # -> correlacion -> importancia por ECR) y guarda todo lo aprendido
    # como atributos de la instancia, para que `transform` lo reutilice
    # sin volver a "ver" TRAIN.
    # Recibe: train_df (SOLO el split de entrenamiento -- nunca val/holdout
    # aqui, o se rompe la regla de oro del modulo), target_cols (lista de
    # columnas TARGET_ORD__<ECR>, usadas para calcular importancia).
    # Devuelve: self (permite encadenar pipeline.fit(...).transform(...)).
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, target_cols: list[str]) -> "FeaturePipeline":
        candidates = self._list_candidate_columns(train_df)
        self.candidate_columns_ = candidates
        log(f"Seleccion de variables: {len(candidates)} columnas candidatas iniciales (numericas, no-ID, no-target).")

        X = train_df[candidates].copy()

        # 2) Casi constantes (sobre TRAIN)
        near_constant = []
        for c in candidates:
            vc = X[c].value_counts(dropna=False, normalize=True)
            if len(vc) == 0 or vc.iloc[0] >= config.NEAR_CONSTANT_THRESHOLD:
                near_constant.append(c)
        self.near_constant_dropped_ = near_constant
        remaining = [c for c in candidates if c not in near_constant]
        log(f"  -> {len(near_constant)} columnas casi-constantes eliminadas (umbral={config.NEAR_CONSTANT_THRESHOLD}).")

        X = X[remaining]

        # 3) Indicadores de missing (para columnas que aun no tengan uno)
        missing_ind_cols = []
        for c in remaining:
            if c.endswith("_MISSING"):
                continue  # ya es un indicador
            if X[c].isna().any():
                missing_ind_cols.append(c)
        self.missing_indicator_columns_ = missing_ind_cols
        log(f"  -> {len(missing_ind_cols)} nuevos indicadores de missing creados.")

        # 4) Medianas de imputacion (sobre TRAIN, tras excluir casi-constantes)
        self.medians_ = X.median(numeric_only=True)

        X_imputed = X.fillna(self.medians_)
        for c in missing_ind_cols:
            X_imputed[f"{c}__WASNULL"] = X[c].isna().astype(int)

        # 5) Correlacion (sobre TRAIN imputado, solo variables numericas originales,
        # no sobre los indicadores binarios de missing)
        base_cols = remaining
        corr_dropped = []
        if len(base_cols) > 1:
            corr_matrix = X_imputed[base_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            for col in upper.columns:
                if col in corr_dropped:
                    continue
                correlated_with = upper.index[upper[col] > config.CORR_THRESHOLD].tolist()
                corr_dropped.extend([c for c in correlated_with if c not in corr_dropped and c != col])
        self.correlation_dropped_ = sorted(set(corr_dropped))
        log(f"  -> {len(self.correlation_dropped_)} columnas eliminadas por alta correlacion (> {config.CORR_THRESHOLD}).")

        remaining_after_corr = [c for c in base_cols if c not in self.correlation_dropped_]
        final_candidate_pool = remaining_after_corr + [f"{c}__WASNULL" for c in missing_ind_cols]

        X_final_pool = X_imputed[final_candidate_pool]

        # 6) Importancia por ECR (RandomForest ligero, solo en filas con target valido)
        importances_union = pd.Series(0.0, index=final_candidate_pool)
        for ecr in config.ECR_LIST:
            y = train_df[f"TARGET_ORD__{ecr}"]
            mask = y.notna()
            n_valid = mask.sum()
            if n_valid < 15:
                log(f"  -> ECR {ecr}: muy pocas observaciones validas en TRAIN ({n_valid}), "
                    f"se omite del ranking de importancia (no afecta su entrenamiento posterior).")
                continue
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=config.SEED,
                class_weight="balanced", n_jobs=-1,
            )
            rf.fit(X_final_pool.loc[mask], y.loc[mask].astype(int))
            imp = pd.Series(rf.feature_importances_, index=final_candidate_pool)
            self.feature_importance_by_ecr_[ecr] = imp.sort_values(ascending=False)
            importances_union = importances_union.add(imp, fill_value=0.0)

        importances_union = importances_union.sort_values(ascending=False)
        self.selected_features_ = importances_union.head(config.MAX_SELECTED_FEATURES).index.tolist()
        log(f"  -> {len(self.selected_features_)} variables finales seleccionadas "
            f"(limite configurado={config.MAX_SELECTED_FEATURES}).")

        return self

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    # Que hace: aplica EXACTAMENTE las transformaciones aprendidas en
    # fit() (mismas columnas, mismas medianas de imputacion, mismo
    # subconjunto final) sin volver a ajustar nada. Seguro de usar sobre
    # VAL, HOLDOUT o datos de prediccion futura.
    # Recibe: df (cualquier dataset con las mismas columnas base que
    # train_df usado en fit).
    # Devuelve: DataFrame con unicamente las columnas de
    # self.selected_features_, en el mismo orden que se uso para entrenar.
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica exactamente las transformaciones aprendidas en fit() (sin refit)."""
        base_needed = [c for c in self.candidate_columns_ if c not in self.near_constant_dropped_]
        X = df[base_needed].copy()
        X = X.fillna(self.medians_)
        for c in self.missing_indicator_columns_:
            X[f"{c}__WASNULL"] = df[c].isna().astype(int)
        # Devuelve solo las columnas finalmente seleccionadas (en el mismo orden)
        for c in self.selected_features_:
            if c not in X.columns:
                X[c] = 0  # columna de indicador que no existia en este subset (no deberia pasar)
        return X[self.selected_features_]
