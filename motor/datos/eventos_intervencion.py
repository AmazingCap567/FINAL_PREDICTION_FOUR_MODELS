# -*- coding: utf-8 -*-
"""
eventos_intervencion.py
========================
Tabla CURADA MANUALMENTE de eventos reales de intervencion/disolucion de
entidades supervisadas por la SBS, verificados contra fuentes externas
(resoluciones SBS publicadas en El Peruano y cobertura de prensa
especializada). Esta NO es una lista teorica ni inventada: cada fila tiene
su resolucion SBS y fecha real.

Por que existe esta tabla en vez de derivar "fracaso" de la propia serie
financiera: se verifico que la senal "la entidad deja de aparecer en el
panel financiero" NO es confiable. De los 4 casos reales documentados
aqui, 2 (CMAC Sullana y Financiera Credinka) SIGUIERON reportando datos
financieros despues de ser intervenidas (el reporte contable continua
durante el proceso de traspaso de cartera a la entidad ganadora del
concurso), por lo que una heuristica de "discontinuidad de panel" los
habria pasado por alto. Ademas, esa misma heuristica genera FALSOS
POSITIVOS: Compartamos Financiera y Crediscotia Financiera tambien "dejan
de aparecer" con su nombre original, pero por conversion a banco y cambio
de razon social respectivamente -- no por fracaso (ver
config.FINANCIAL_ENTITY_CONTINUITY_RENAMES para el caso de Crediscotia).

Con solo 4 eventos en ~4 anios y ~49 entidades, esta tabla NO se usa para
entrenar un clasificador binario supervisado clasico (la muestra es
demasiado chica para que cualquier metrica de un modelo asi sea
estadisticamente significativa). Se usa para:
  (a) marcar la fecha a partir de la cual una entidad intervenida deja de
      ser elegible para generar predicciones futuras (ver
      dataset_builder.excluir_posteriores_a_intervencion), y
  (b) como conjunto de validacion CUALITATIVA del SCORE_DETERIORO (el
      proxy continuo de downgrades fuertes de rating, ver
      motor/features/riesgo.py): si el score no se eleva para estos 4
      casos en los cortes previos a la intervencion, es una senal de que
      el proxy no esta funcionando.
"""

from __future__ import annotations
import pandas as pd


# Clave: nombre normalizado de la entidad tal como queda en el universo
# financiero (ver motor.utils.normalize_entity_name), para poder cruzarlo
# directamente con ENTITY_NAME_NORM del registro de entidades.
EVENTOS_INTERVENCION = [
    {
        "ENTITY_NAME_NORM": "CRAC RAIZ",
        "TIPO_ENTIDAD": "CRAC",
        "FECHA_EVENTO": "2023-08-10",
        "TIPO_EVENTO": "INTERVENCION_Y_LIQUIDACION",
        "RESOLUCION_SBS": "Res. SBS N.° 2646-2023 (intervencion) / N.° 2672-2023 (disolucion)",
        "DESCRIPCION": "Intervenida por deterioro acelerado de solvencia; disuelta y en liquidacion.",
        "FUENTE": "El Peruano, 11-ago-2023; SBS FAQ CRAC Raiz.",
    },
    {
        "ENTITY_NAME_NORM": "AMERIKA FINANCIERA",
        "TIPO_ENTIDAD": "FINANCIERA",
        "FECHA_EVENTO": "2022-08-26",
        "TIPO_EVENTO": "DISOLUCION_VOLUNTARIA",
        "RESOLUCION_SBS": "Acuerdo de Junta General de Accionistas (sujeto a autorizacion SBS conforme Res. SBS N.° 455-99)",
        "DESCRIPCION": "Fallo el proyecto de fusion con Banco Pichincha; accionistas aprobaron disolucion y liquidacion.",
        "FUENTE": "Revista Gan@Más, ago-2022.",
    },
    {
        "ENTITY_NAME_NORM": "CMAC SULLANA",
        "TIPO_ENTIDAD": "CMAC",
        "FECHA_EVENTO": "2024-07-11",
        "TIPO_EVENTO": "INTERVENCION_Y_LIQUIDACION",
        "RESOLUCION_SBS": "Res. SBS N.° 2477-2024 (intervencion) / N.° 2497-2024 (disolucion)",
        "DESCRIPCION": "Intervenida por deterioro acelerado de solvencia (patrimonio); cartera transferida a CMAC Piura.",
        "FUENTE": "El Peruano, 15-jul-2024; Infobae, 14-jul-2024.",
    },
    {
        "ENTITY_NAME_NORM": "FINANCIERA CREDINKA",
        "TIPO_ENTIDAD": "FINANCIERA",
        "FECHA_EVENTO": "2024-09-19",
        "TIPO_EVENTO": "INTERVENCION",
        "RESOLUCION_SBS": "Res. SBS N.° 3341-2024",
        "DESCRIPCION": "Intervenida por acelerado deterioro de solvencia (reduccion de patrimonio de 59.53% en 12 meses); cartera transferida via concurso por invitacion.",
        "FUENTE": "Gestion, 19-sep-2024; El Comercio, 19-sep-2024.",
    },
]


# ---------------------------------------------------------------------------
# tabla_eventos
# ---------------------------------------------------------------------------
# Que hace: devuelve la tabla de eventos como DataFrame, con FECHA_EVENTO
# ya convertida a datetime.
# ---------------------------------------------------------------------------
def tabla_eventos() -> pd.DataFrame:
    df = pd.DataFrame(EVENTOS_INTERVENCION)
    df["FECHA_EVENTO"] = pd.to_datetime(df["FECHA_EVENTO"])
    return df


# ---------------------------------------------------------------------------
# fecha_limite_por_entidad
# ---------------------------------------------------------------------------
# Que hace: construye {ENTITY_NAME_NORM: fecha_evento} para las entidades
# con evento de intervencion conocido.
# Para que sirve: dataset_builder lo usa para NO generar filas de
# prediccion futura para una entidad despues de su fecha de intervencion,
# aunque el panel financiero siga teniendo datos posteriores (que
# corresponden al proceso de liquidacion/traspaso, no a la entidad
# operando con normalidad).
# ---------------------------------------------------------------------------
def fecha_limite_por_entidad() -> dict[str, pd.Timestamp]:
    df = tabla_eventos()
    return dict(zip(df["ENTITY_NAME_NORM"], df["FECHA_EVENTO"]))
