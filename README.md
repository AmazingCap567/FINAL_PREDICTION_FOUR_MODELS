# Motor de predicción de rating y riesgo de entidades financieras (SBS Perú)

Manual del proyecto. Pensado para que alguien sin contexto previo pueda
entender qué hace el motor, por qué está organizado así, y cómo correrlo.

---

## 1. ¿Qué hace este proyecto? (en lenguaje llano)

Perú tiene ~49 entidades financieras supervisadas por la SBS (bancos,
cajas municipales, cajas rurales y financieras). Cada semestre, hasta 6
clasificadoras de riesgo privadas (ECR: APOYO, CLASS, JCR, MICRORATE,
MOODYS, PCR) les asignan una **calificación crediticia** (parecida a un
rating de bonos: A+, A, A-, B+... hasta E). Esa calificación resume qué
tan sólida está la entidad.

Este motor hace dos cosas:

1. **Predice qué calificación le va a poner cada ECR a cada entidad en el
   próximo semestre**, usando sus estados financieros históricos (activos,
   cartera, morosidad, liquidez, solvencia, etc.). Es un problema de
   clasificación **ordinal** (las categorías están ordenadas: A+ es mejor
   que A, que es mejor que A-, etc.), no una clasificación cualquiera.
2. **Estima el riesgo de deterioro fuerte** de una entidad (una caída
   grande de calificación, el tipo de señal que suele preceder a una
   intervención de la SBS), como objetivo adicional al rating.

**Lo que este proyecto NO hace (todavía):** no tiene interfaz gráfica. Es
el motor de cálculo. Genera archivos Excel limpios (`outputs/interfaz/`)
pensados para que una interfaz futura los lea directamente, sin tener que
re-ejecutar el entrenamiento.

### ¿Por qué importa esto?

Una entidad financiera que se deteriora sin que nadie lo note a tiempo
puede terminar intervenida por la SBS (ver sección 6): sus depositantes y
la estabilidad del sistema financiero están en juego. Anticipar una
caída de calificación (o el riesgo de una intervención) con meses de
anticipación tiene valor real para reguladores, inversionistas y para la
propia gestión de riesgo de las entidades.

---

## 2. Requisitos

```bash
Python 3.10+
pip install -r requirements.txt
```

Incluye: pandas, numpy, scikit-learn, torch (CPU), **mord** (regresión
ordinal), **lightgbm**, openpyxl, lxml, beautifulsoup4, shap (opcional).

---

## 3. Mapa de carpetas

```
data/                          Datos fuente (no se modifican)
├── dataset_bancos.csv
├── dataset_caja_municipal.csv
├── dataset_financieras.xlsx
├── dataset_crac.xlsx
└── clasificaciones/           9 archivos .xls (en realidad HTML) de la SBS,
                                uno por corte semestral MAR2022..MAR2026

motor/                         El motor de predicción en sí
├── config.py                  Rutas, escalas, umbrales, diccionarios de
│                               homologación — la "configuración maestra"
├── utils.py                   Logging con formato de secciones
├── datos/                     Carga, homologación de entidades, integración
│   ├── sbs_loader.py             temporal (anti-fuga), eventos de riesgo
│   ├── financial_loader.py
│   ├── entity_homology.py
│   ├── ratings.py
│   ├── dataset_builder.py
│   └── eventos_intervencion.py   Tabla curada de 4 intervenciones SBS reales
├── features/                  Ingeniería y selección de variables
│   ├── feature_engineering.py    Lags, variaciones, tendencias
│   └── selection.py              Filtro + selección final (RandomForest)
├── modelos/                   Los 4 modelos comparables, cada uno en su
│   │                           propia carpeta con la misma interfaz
│   │                           (entrenar(X, y) -> modelo; predecir(modelo, X) -> preds)
│   ├── modelo_1_hist_gradient_boosting/
│   ├── modelo_2_mlp_ordinal/      (red neuronal PyTorch, multi-tarea)
│   ├── modelo_3_regresion_ordinal/ (mord, interpretable)
│   ├── modelo_4_lightgbm/
│   └── comparacion/
│       ├── baselines_piso.py      Mayoría / persistencia (NO son de los 4)
│       └── seleccion.py           Metodología reproducible del ganador
├── entrenamiento/              (reservado para orquestación futura)
└── evaluacion/                 Métricas, validación temporal, explicabilidad,
                                 reporte HTML

dataset_variables/              Qué variables usa REALMENTE el motor,
├── banco/                      por tipo de entidad (no una lista teórica)
├── cmac/
├── crac/
└── financiera/
    ├── variables_entrenamiento.xlsx
    ├── variables_descripcion.xlsx
    └── resumen_variables.txt

outputs/
├── interfaz/                   Salidas LIMPIAS para la futura interfaz
│                                (un Excel por tipo de entidad + consolidado)
└── analisis/                   Todo lo auxiliar: métricas detalladas,
    ├── auditoria/               artifacts del modelo, reporte HTML.
    ├── predicciones/            La interfaz NUNCA debería necesitar leer
    ├── reporte/                 nada de aquí.
    └── artifacts/

scripts/
├── run_evaluacion.py           Evalúa los 4 modelos contra el holdout
│                                conocido (MAR2026) y elige el ganador por ECR
├── run_prediccion.py           Genera la predicción real hacia el futuro
│                                (corte SEP2026) y escribe outputs/interfaz/
└── generar_dataset_variables.py  Genera dataset_variables/ por tipo de entidad
```

---

## 4. Cómo ejecutar

```bash
cd MASTER_PREDICTION_V2

# 1. Evaluar los 4 modelos contra el holdout conocido (MAR2026)
python scripts/run_evaluacion.py

# 2. Generar la documentación de variables por tipo de entidad
python scripts/generar_dataset_variables.py

# 3. Generar la predicción real hacia el futuro (para la interfaz)
python scripts/run_prediccion.py
```

Cada script imprime en consola un resumen legible por secciones. Los
detalles exhaustivos (tablas completas, matrices de confusión) van a
`outputs/analisis/`, no a la consola, para no saturarla.

---

## 5. Los 4 modelos comparables

| # | Modelo | Librería | Por qué se incluye |
|---|---|---|---|
| 1 | HistGradientBoosting | scikit-learn | Robusto, no lineal, referencia estándar |
| 2 | MLP ordinal multi-tarea | PyTorch | Comparte información entre las 6 ECR; predice el *delta* de rating, no el nivel absoluto |
| 3 | Regresión Logística Ordinal | mord | El único que asume explícitamente que las clases están ordenadas; interpretable |
| 4 | LightGBM | lightgbm | Gradient boosting con ponderación de clases nativa (útil para el desbalance) |

**Mayoría** y **Persistencia** (el modelo "no hace nada, asume que no
cambia") existen como piso de referencia en `comparacion/baselines_piso.py`,
pero no compiten como modelos de verdad — si un modelo no le gana a
Persistencia, no aporta valor.

El **ganador por ECR** se elige con un puntaje compuesto (Weighted Kappa +
MAE ordinal + F1 macro, normalizado y ponderado) definido en
`comparacion/seleccion.py`, no por una sola métrica.

---

## 6. Los 4 eventos de intervención SBS usados como referencia

Con solo ~49 entidades y 4 años de historia, no hay suficientes fracasos
reales como para entrenar un clasificador binario de "quiebra" con
confianza estadística. En su lugar, se usa una tabla **curada y verificada
externamente** (`motor/datos/eventos_intervencion.py`) de los 4 casos
reales documentados en el período:

| Entidad | Fecha | Qué pasó |
|---|---|---|
| CRAC Raíz | ago-2023 | Intervenida y liquidada (deterioro de solvencia) |
| Amérika Financiera | ago-2022 | Disolución voluntaria (falló su fusión con Banco Pichincha) |
| CMAC Sullana | jul-2024 | Intervenida y liquidada; cartera transferida a CMAC Piura |
| Financiera Credinka | sep-2024 | Intervenida (deterioro acelerado de solvencia) |

Cada fila tiene su resolución SBS y fuente. Esta tabla se usa para: (a)
excluir a esas entidades de predicciones futuras después de su fecha de
intervención (nunca se les "inventa" un futuro), y (b) validar
cualitativamente que el proxy de riesgo (deterioro de rating) efectivamente
se elevó antes de cada intervención real.

**Importante:** dos de estas 4 entidades (CMAC Sullana y Credinka)
siguieron reportando datos financieros después de ser intervenidas (el
reporte contable continúa durante el traspaso de cartera). Por eso la
tabla está curada manualmente y verificada contra fuentes externas, en
vez de inferirse solo de "la entidad deja de aparecer en los datos".

**Proxy de riesgo continuo (SCORE_DETERIORO / DOWNGRADE_FUERTE):** una
entidad "dispara" este target cuando su calificación cae 2 o más
escalones en un mismo corte, en cualquier ECR. El umbral de 2 se eligió
mirando la distribución real de cambios de rating (las caídas de 1
escalón son comunes y no distinguen mucho; las de 2+ son raras — ~1.5%
de los casos — y coinciden con varios de los eventos reales de la tabla
de arriba).

---

## 7. Metodología: anti-fuga y validación temporal

Regla de oro del proyecto: **para predecir en una fecha, el modelo solo
puede usar información que ya existía en esa fecha.**

- Cada fila del dataset usa el último dato financiero **estrictamente
  anterior** a la fecha de clasificación objetivo (ventana máxima
  configurable). El pipeline se detiene con un error si detecta alguna
  fila con `FECHA_BASE >= FECHA_TARGET`.
- Toda imputación, escalado y selección de variables se ajusta
  ÚNICAMENTE con el conjunto de entrenamiento; nunca con validación,
  holdout, o los datos de predicción futura.
- **Holdout de evaluación: MAR2026** (el corte más reciente con
  clasificación real publicada). Validación (early stopping del MLP):
  SEP2025. Entrenamiento: todos los cortes anteriores.
- Además del holdout único, hay validación **walk-forward** (se repite el
  entrenamiento avanzando el corte de prueba varias veces) para confirmar
  que el desempeño no depende de haber elegido "el mejor" corte de prueba
  por casualidad.

---

## 8. Glosario

- **ECR**: Empresa Clasificadora de Riesgo (agencia de rating): APOYO,
  CLASS, JCR, MICRORATE, MOODYS, PCR.
- **CMAC**: Caja Municipal de Ahorro y Crédito.
- **CRAC**: Caja Rural de Ahorro y Crédito.
- **Rating ordinal**: la calificación (A+, A, A-, ..., E) representada
  como un número (0 = mejor, más alto = peor) para poder hacer aritmética
  ordenada con ella (comparar, restar, calcular error promedio).
- **Notch / escalón**: un paso en la escala de rating (de B+ a B es 1
  notch).
- **Delta de rating**: diferencia entre el rating nuevo y el anterior, en
  notches. Positivo = empeoró (downgrade); negativo = mejoró (upgrade).
- **Holdout**: el corte de datos que el modelo nunca ve durante el
  entrenamiento, reservado para medir qué tan bien generaliza.
- **Walk-forward**: validación que repite el entrenamiento avanzando la
  fecha de corte de prueba varias veces, en vez de usar un único holdout.
- **Fuga de información (data leakage)**: cuando el modelo usa,
  accidentalmente, información que en la realidad no habría estado
  disponible en el momento de la predicción.
- **Weighted Kappa**: métrica de acuerdo ordinal que penaliza más los
  errores grandes (predecir E cuando era A+) que los pequeños (predecir A
  cuando era A-).

---

## 9. Limitaciones conocidas de esta etapa (resuelta)

- El modelo 2 (MLP) requiere PyTorch; en entornos sin GPU/con poco
  espacio en disco puede ser el más costoso de instalar.
- Con muestras tan chicas por tipo de entidad (6 a 19 entidades), toda
  métrica de desempeño debe leerse con cautela — no hay volumen para
  intervalos de confianza estrechos.
- El proxy de riesgo (`DOWNGRADE_FUERTE`) es una aproximación interna,
  no una clasificación oficial de riesgo de la SBS ni de ninguna ECR.
- No hay interfaz gráfica todavía; esa es la siguiente etapa del
  proyecto, y consumirá directamente los archivos de `outputs/interfaz/`.
