# -*- coding: utf-8 -*-
"""
entity_homology.py
===================
Sistema de homologacion de entidades entre:
  (a) los nombres publicados por la SBS en los archivos de clasificaciones, y
  (b) los nombres usados en los 4 datasets financieros (bancos, caja
      municipal, financieras, CRAC).

Reglas (ver enunciado del proyecto):
  - Se normalizan nombres (mayusculas, sin acentos, espacios colapsados).
  - Se usa un diccionario EXPLICITO de excepciones (config.ENTITY_NAME_OVERRIDES)
    para los casos que no homologan por normalizacion directa.
  - NO se hace matching difuso silencioso: si un nombre normalizado (con o sin
    override) no aparece en el universo financiero, la entidad se registra en
    entidades_no_homologadas.csv y se excluye del entrenamiento.
  - Las entidades para las que se sabe de antemano que no existe dataset
    financiero (config.EXPECTED_ENTITIES_WITHOUT_FINANCIAL_DATA) tambien se
    excluyen, pero se marcan con un motivo distinto en la auditoria.
"""

from __future__ import annotations
import os
import pandas as pd

from .. import config
from ..utils import normalize_entity_name, log


class EntityRegistry:
    """
    Mantiene el universo canonico de entidades (ENTITY_ID, ENTITY_NAME,
    ENTITY_TYPE) construido a partir de los datasets FINANCIEROS (fuente de
    verdad para qué entidades tienen datos utilizables), y resuelve nombres
    provenientes de la SBS hacia ese universo.
    """

    def __init__(self):
        self.canonical_df: pd.DataFrame | None = None          # ENTITY_ID, ENTITY_NAME_NORM, ENTITY_TYPE
        self._norm_to_id: dict[str, str] = {}
        self.no_homologadas: list[dict] = []

    # ------------------------------------------------------------------
    # build_from_financial_sources
    # ------------------------------------------------------------------
    # Que hace: recorre los 4 datasets financieros y arma el universo
    # UNICO de entidades (una fila por nombre normalizado distinto),
    # asignandole un ENTITY_ID estable (ENT_0001, ENT_0002, ...).
    # Por que asi: los datasets financieros son la FUENTE DE VERDAD de que
    # entidades existen y de que tipo son (no la SBS) -- ver modulo docstring.
    # Recibe: sources (dict {nombre_fuente: DataFrame con columnas ENTIDAD,
    # TIPO_DE_ENTIDAD}, tipicamente el resultado ya renombrado por
    # financial_loader._apply_entity_continuity_renames).
    # Devuelve: self.canonical_df (ENTITY_ID, ENTITY_NAME, ENTITY_NAME_NORM,
    # ENTITY_TYPE), y lo guarda tambien como atributo de la instancia.
    # ------------------------------------------------------------------
    def build_from_financial_sources(self, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        sources: dict {nombre_fuente: dataframe con columnas ENTIDAD, TIPO_DE_ENTIDAD}
        Construye el universo canonico de entidades (una fila por entidad unica).
        """
        rows = []
        for source_name, df in sources.items():
            sub = df[["ENTIDAD", "TIPO_DE_ENTIDAD"]].drop_duplicates()
            for _, r in sub.iterrows():
                norm_name = normalize_entity_name(r["ENTIDAD"])
                entity_type = str(r["TIPO_DE_ENTIDAD"]).strip().upper()
                rows.append({
                    "ENTITY_NAME_NORM": norm_name,
                    "ENTITY_NAME": str(r["ENTIDAD"]).strip(),
                    "ENTITY_TYPE": entity_type,
                    "SOURCE": source_name,
                })
        raw = pd.DataFrame(rows).drop_duplicates(subset=["ENTITY_NAME_NORM"])
        raw = raw.sort_values(["ENTITY_TYPE", "ENTITY_NAME_NORM"]).reset_index(drop=True)
        raw["ENTITY_ID"] = [f"ENT_{i:04d}" for i in range(len(raw))]

        self.canonical_df = raw[["ENTITY_ID", "ENTITY_NAME", "ENTITY_NAME_NORM", "ENTITY_TYPE"]].copy()
        self._norm_to_id = dict(zip(self.canonical_df["ENTITY_NAME_NORM"], self.canonical_df["ENTITY_ID"]))
        return self.canonical_df

    # ------------------------------------------------------------------
    # resolve_sbs_name
    # ------------------------------------------------------------------
    # Que hace: intenta homologar un nombre de entidad tal como lo escribe
    # la SBS en un corte de clasificacion, hacia un ENTITY_ID del universo
    # canonico, probando en orden: (1) coincidencia directa normalizada,
    # (2) diccionario explicito de excepciones (config.ENTITY_NAME_OVERRIDES),
    # (3) exclusion esperada y documentada (entidad sabida sin dataset
    # financiero), (4) fallo real (se documenta para revision manual).
    # Por que asi: NO se hace matching difuso/automatico (por similitud de
    # texto) para evitar homologar por error dos entidades distintas; toda
    # homologacion no trivial queda explicita y auditable.
    # Recibe: raw_name (nombre tal como aparece en el archivo SBS),
    # raw_type (tipo tal como lo etiqueta la SBS ese corte), corte (para
    # la auditoria de fallos).
    # Devuelve: ENTITY_ID si se resuelve, o None (y registra el motivo).
    # ------------------------------------------------------------------
    def resolve_sbs_name(self, raw_name: str, raw_type: str, corte: str) -> str | None:
        """
        Intenta resolver un nombre SBS hacia un ENTITY_ID del universo canonico.
        Devuelve None y registra el fallo si no puede homologarse con confianza.
        """
        norm = normalize_entity_name(raw_name)

        # 1) Coincidencia directa tras normalizacion
        if norm in self._norm_to_id:
            return self._norm_to_id[norm]

        # 2) Diccionario explicito de excepciones
        if norm in config.ENTITY_NAME_OVERRIDES:
            target_norm = normalize_entity_name(config.ENTITY_NAME_OVERRIDES[norm])
            if target_norm in self._norm_to_id:
                return self._norm_to_id[target_norm]

        # 3) Exclusion esperada y documentada (sin dataset financiero)
        if norm in config.EXPECTED_ENTITIES_WITHOUT_FINANCIAL_DATA:
            self.no_homologadas.append({
                "CORTE": corte,
                "ENTITY_NAME_RAW": raw_name,
                "ENTITY_TYPE_RAW": raw_type,
                "MOTIVO": "Sin dataset financiero (exclusion esperada y documentada)",
            })
            return None

        # 4) No se pudo homologar con seguridad -> se registra como fallo real
        self.no_homologadas.append({
            "CORTE": corte,
            "ENTITY_NAME_RAW": raw_name,
            "ENTITY_TYPE_RAW": raw_type,
            "MOTIVO": "No homologable: no existe coincidencia directa ni en el diccionario de excepciones",
        })
        return None

    # ------------------------------------------------------------------
    def save_no_homologadas(self, path: str) -> pd.DataFrame:
        df = pd.DataFrame(self.no_homologadas).drop_duplicates()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df
