# -*- coding: utf-8 -*-
"""
config.py
=========
Configuracion central del proyecto de prediccion de clasificaciones ECR.

Todas las decisiones metodologicas relevantes (rutas, escalas ordinales,
diccionario de homologacion de entidades, hiperparametros, semillas) viven
aqui para que el resto del codigo no tenga "numeros magicos" dispersos.
"""

from __future__ import annotations
import os
import random
import numpy as np

# ---------------------------------------------------------------------------
# 1. SEMILLA / REPRODUCIBILIDAD
# ---------------------------------------------------------------------------
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fija la semilla en todas las librerias relevantes."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # algunas ops CPU no tienen version determinista
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 2. RUTAS
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLASIFICACIONES_DIR = os.path.join(DATA_DIR, "clasificaciones")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Separacion de salidas (punto 17 del proyecto):
#   - INTERFAZ_DIR: archivos limpios y estables, listos para que la futura
#     interfaz los consuma directamente (un Excel por tipo de entidad). La
#     interfaz NUNCA deberia necesitar leer nada de ANALISIS_DIR.
#   - ANALISIS_DIR: todo lo auxiliar para desarrollo/depuracion/auditoria
#     (metricas detalladas, artifacts de modelos, reporte HTML, etc.).
INTERFAZ_DIR = os.path.join(OUTPUT_DIR, "interfaz")
ANALISIS_DIR = os.path.join(OUTPUT_DIR, "analisis")
ARTIFACTS_DIR = os.path.join(ANALISIS_DIR, "artifacts")       # modelos, scalers, encoders
AUDIT_DIR = os.path.join(ANALISIS_DIR, "auditoria")           # csvs de auditoria
PRED_DIR = os.path.join(ANALISIS_DIR, "predicciones")
REPORT_DIR = os.path.join(ANALISIS_DIR, "reporte")
FIG_DIR = os.path.join(REPORT_DIR, "figuras")
DATASET_VARIABLES_DIR = os.path.join(PROJECT_ROOT, "dataset_variables")

FILE_BANCOS = os.path.join(DATA_DIR, "dataset_bancos.csv")
FILE_CAJA_MUNICIPAL = os.path.join(DATA_DIR, "dataset_caja_municipal.csv")
FILE_FINANCIERAS = os.path.join(DATA_DIR, "dataset_financieras.xlsx")
FILE_CRAC = os.path.join(DATA_DIR, "dataset_crac.xlsx")

for d in (OUTPUT_DIR, INTERFAZ_DIR, ANALISIS_DIR, ARTIFACTS_DIR, AUDIT_DIR, PRED_DIR, REPORT_DIR, FIG_DIR, DATASET_VARIABLES_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. CORTES DE CLASIFICACION (SBS)
# ---------------------------------------------------------------------------
# Nombre de archivo -> fecha de corte real que representa la publicacion SBS.
# Se documenta explicitamente la fecha asumida para cada corte (ultimo dia
# del mes correspondiente), ya que los archivos SBS solo traen el nombre
# del periodo (MARyyyy / SEPyyyy), no una fecha exacta dentro del propio
# archivo.
CORTES_CLASIFICACION = {
    "MAR2022": "2022-03-31",
    "SEP2022": "2022-09-30",
    "MAR2023": "2023-03-31",
    "SEP2023": "2023-09-30",
    "MAR2024": "2024-03-31",
    "SEP2024": "2024-09-30",
    "MAR2025": "2025-03-31",
    "SEP2025": "2025-09-30",
    "MAR2026": "2026-03-30",
}

# Numero maximo de meses hacia atras que se admite para buscar el ultimo
# dato financiero disponible ANTES del corte objetivo. Si no hay ningun dato
# financiero dentro de esta ventana, la observacion no se construye (no se
# inventa informacion).
MAX_LOOKBACK_MONTHS = 3

# Meses usados para features de variacion / tendencia (sin fuga de informacion:
# siempre construidos con datos <= fecha base).
LAG_MONTHS = (1, 3, 6)

# ---------------------------------------------------------------------------
# 4. ECR (AGENCIAS CLASIFICADORAS) OBJETIVO
# ---------------------------------------------------------------------------
ECR_LIST = ["APOYO", "CLASS", "JCR", "MICRORATE", "MOODYS", "PCR"]

# Nombres de columnas tal como aparecen en las tablas HTML de la SBS ->
# nombre canonico de ECR. Se usa coincidencia por substring (case-insensitive,
# sin acentos) para ser robusto a pequenas variaciones entre cortes.
ECR_COLUMN_HINTS = {
    "APOYO": ["apoyo"],
    "CLASS": ["class"],
    "JCR": ["jcr"],
    "MICRORATE": ["microrate"],
    "MOODYS": ["moodys", "moody's", "moody"],
    "PCR": ["pcr", "pacific credit rating"],
}

# ---------------------------------------------------------------------------
# 5. TIPOS DE ENTIDAD PERMITIDOS
# ---------------------------------------------------------------------------
ENTITY_TYPES = ["BANCO", "CMAC", "FINANCIERA", "CRAC"]

# Traduccion de "Tipo de Entidad" tal como aparece en el archivo SBS ->
# tipo canonico. Cualquier tipo SBS que no este en este diccionario
# (p.ej. Seguros, Fondo, Empresa de Creditos, Afianzadora) queda FUERA del
# universo de modelado porque no existe dataset financiero correspondiente.
SBS_ENTITY_TYPE_MAP = {
    "banco": "BANCO",
    "caja municipal de ahorro y credito": "CMAC",
    "caja municipal de ahorro y crédito": "CMAC",
    "caja rural de ahorro y credito": "CRAC",
    "caja rural de ahorro y crédito": "CRAC",
    "financiera": "FINANCIERA",
}

# ---------------------------------------------------------------------------
# 6. ESCALA ORDINAL DE RATINGS
# ---------------------------------------------------------------------------
# Escala maestra (de mejor a peor calidad crediticia). No todas las ECR usan
# todas las categorias; cada ECR usa el subconjunto que efectivamente
# publica, pero el ORDEN relativo es el mismo para todas (es la convencion
# estandar de escalas de letras en el mercado peruano). Esto se documenta
# explicitamente en vez de asumirse implicitamente.
RATING_MASTER_SCALE = [
    "A+", "A", "A-",
    "B+", "B", "B-",
    "C+", "C", "C-",
    "D+", "D", "D-",
    "E",
]
RATING_MASTER_ORDINAL = {label: i for i, label in enumerate(RATING_MASTER_SCALE)}

# Etiquetas que NO son un nivel de rating valido y deben tratarse como
# "sin clasificacion valida" (faltante), no como una clase mas:
#   - "RET" (retirado / rating withdrawn)
INVALID_RATING_TOKENS = {"RET", "NR", "SIN CLASIFICAR", "N/A", "NA", "-"}

# ---------------------------------------------------------------------------
# 7. DICCIONARIO EXPLICITO DE HOMOLOGACION DE ENTIDADES
# ---------------------------------------------------------------------------
# Clave: nombre normalizado (mayusculas, sin acentos, espacios simples) tal
# como aparece en los archivos de CLASIFICACIONES SBS.
# Valor: nombre normalizado tal como aparece en el dataset FINANCIERO
# correspondiente (bancos / caja municipal / financieras / crac).
#
# Este diccionario se construyo inspeccionando manualmente los 4 datasets
# financieros y las 8 tablas de clasificaciones (ver auditoria de
# entidades_no_homologadas.csv para lo que NO pudo mapearse).
ENTITY_NAME_OVERRIDES = {
    # --- BANCOS ---
    "ALFIN BANCO": "BANCO ALFIN",
    "BANCO DE CREDITO": "BANCO DE CREDITO",
    "BANCO BCI": "BCI PERU",
    "BBVA": "BANCO CONTINENTAL",          # BBVA Peru opera bajo la razon social "Banco Continental"
    "CITIBANK DEL PERU": "CITIBANK",
    "ICBC PERU BANK S.A.": "BANCO ICBC",
    "SANTANDER PERU": "BANCO SANTANDER",
    "SCOTIABANK PERU": "SCOTIABANK",
    "BN. SANTANDER CONS.": "SANTANDER CONSUMER BANK S.A",
    "BANCOM": "BANCO DE COMERCIO",
    "BANBIF": "BANCO INTERAMERICANO DE FINANZAS",
    # AGROBANCO, COFIDE, NACION: sin dataset financiero -> se excluyen
    # explicitamente (ver EXCLUDED_NO_FINANCIAL_DATA mas abajo).

    # --- CMAC ---
    # (la mayoria homologa 1:1 tras normalizar "CMAC <ciudad>")
    # CMCP LIMA: sin dataset financiero -> excluida.

    # --- CRAC ---
    "CRAC RAIZ EN LIQUIDA": "CRAC RAIZ",

    # --- FINANCIERAS ---
    "AMERIKA FINANC. EL": "AMERIKA FINANCIERA",
    "FINANC. CREDINKA": "FINANCIERA CREDINKA",
    "FINANC. PROEMPRESA": "FINANCIERA PROEMPRESA",
    # "Banco Efectiva" en las tablas SBS corresponde a la misma entidad que
    # aparece en el dataset financiero como "Financiera Efectiva" (la SBS
    # etiqueta el Tipo de Entidad de forma inconsistente entre cortes para
    # este caso puntual). Se homologa por nombre; el ENTITY_TYPE final que
    # prevalece es el del universo financiero (fuente de verdad), no el de
    # la etiqueta SBS de ese corte especifico.
    "BANCO EFECTIVA": "FINANCIERA EFECTIVA",
    # FINANCIERA SURGIR, INFINANCE XP S.A.: sin dataset financiero -> excluidas.
}

# Entidades SBS que se sabe de antemano que NO tienen dataset financiero
# (para no reportarlas como "error de homologacion" sino como exclusion
# esperada y documentada).
EXPECTED_ENTITIES_WITHOUT_FINANCIAL_DATA = {
    "AGROBANCO",
    "COFIDE",
    "NACION",
    "CMCP LIMA",
    "FINANCIERA SURGIR",
    "INFINANCE XP S.A.",
}

# ---------------------------------------------------------------------------
# 7bis. CONTINUIDAD DE ENTIDADES QUE CAMBIARON DE RAZON SOCIAL
# ---------------------------------------------------------------------------
# A diferencia de ENTITY_NAME_OVERRIDES (que homologa nombre SBS -> nombre
# dataset financiero), este diccionario resuelve el caso en que UNA MISMA
# entidad aparece con dos nombres distintos DENTRO del propio dataset
# financiero, en tramos de tiempo consecutivos y sin traslape, porque la SBS
# le aprobo un cambio de razon social. Sin este mapeo, entity_homology.py
# crearia dos ENTITY_ID distintos y la entidad "perderia" su historial al
# cambiar de nombre, aunque juridicamente sea la misma empresa.
#
# Clave: nombre nuevo (tal como aparece en el dataset financiero).
# Valor: nombre anterior, que se conserva como nombre canonico de la entidad.
#
# Verificado (13-mar-2025, Resolucion SBS N.º 01015-2025): "CrediScotia
# Financiera S.A." cambio su denominacion social a "Financiera Santander
# Consumer S.A." tras ser adquirida por Banco Santander (transferencia de
# acciones 28-feb-2025). Es la MISMA persona juridica (mismo Estatuto Social
# modificado, no una fusion ni una entidad nueva). En el panel no hay ningun
# mes de traslape entre ambos nombres (CrediScotia: ene-2022 .. feb-2025;
# Financiera Santander Consumer: mar-2025 .. may-2025), consistente con un
# simple cambio de nombre y no con dos entidades operando en paralelo.
FINANCIAL_ENTITY_CONTINUITY_RENAMES = {
    "FINANCIERA SANTANDER CONSUMER S.A.": "CREDISCOTIA FINANCIERA",
}

# ---------------------------------------------------------------------------
# 7ter. UMBRAL DE "DOWNGRADE FUERTE" (proxy continuo de riesgo/deterioro)
# ---------------------------------------------------------------------------
# Elegido a partir de la distribucion EMPIRICA de cambios de rating entre
# cortes consecutivos (no es un numero arbitrario): sobre 1009 pares
# entidad-ECR-corte con corte previo valido, el 91.4% no tiene cambio, las
# caidas de 1 escalon son razonablemente comunes (42 casos, ruido normal de
# mercado), y las caidas de 2 o mas escalones son raras (15 casos, ~1.5%) y
# coinciden en varios casos con eventos de intervencion SBS verificados
# (ver eventos_intervencion.py). DOWNGRADE_THRESHOLD_NOTCHES=2 separa esas
# dos poblaciones.
DOWNGRADE_THRESHOLD_NOTCHES = 2

# ---------------------------------------------------------------------------
# 8. COLUMNAS A EXCLUIR SIEMPRE DE LAS VARIABLES PREDICTORAS
# ---------------------------------------------------------------------------
ID_LIKE_COLUMNS = {
    "PERIODO", "ENTIDAD", "TIPO_DE_ENTIDAD", "FECHA", "ENTITY_ID",
    "ENTITY_NAME", "ENTITY_TYPE", "MES_SIN", "MES_COS", "ANIO",
}

# ---------------------------------------------------------------------------
# 9. HIPERPARAMETROS DEL MODELO
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "hidden_1": 64,
    "hidden_2": 32,
    "hidden_3": 16,
    "entity_type_embed_dim": 4,
    "dropout": 0.30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 20,
    "min_delta": 1e-4,
}

# Umbral de correlacion para eliminacion de variables redundantes.
CORR_THRESHOLD = 0.97
# Umbral de varianza minima (tras escalado no aplica; se usa sobre datos crudos
# como proporcion de valores no-missing con variacion).
NEAR_CONSTANT_THRESHOLD = 0.995
# Numero maximo de variables finales tras seleccion (controla dimensionalidad).
MAX_SELECTED_FEATURES = 60

# Metrica principal para reportes comparativos (Weighted Cohen's Kappa, ya que
# la accuracy no es apropiada para variables ordinales desbalanceadas).
PRIMARY_METRIC = "weighted_kappa"

# ---------------------------------------------------------------------------
# 10. MODELADO POR DELTA (cambio de rating) EN VEZ DE NIVEL ABSOLUTO
# ---------------------------------------------------------------------------
MAX_DELTA_CLIP = 3
DELTA_CLASSES = list(range(-MAX_DELTA_CLIP, MAX_DELTA_CLIP + 1))  # [-3..3] -> 7 clases
N_DELTA_CLASSES = len(DELTA_CLASSES)
DELTA_OFFSET = MAX_DELTA_CLIP  # indice de "sin cambio" (delta=0) dentro de las clases
