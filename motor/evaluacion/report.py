# -*- coding: utf-8 -*-
"""
report.py
=========
Genera el reporte HTML final, amigable y autocontenido (un solo archivo,
CSS inline), con las 17 secciones requeridas por el enunciado.
"""

from __future__ import annotations
import os
import html as html_lib
import pandas as pd

from .. import config
from ..utils import log

CSS = """
<style>
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0;
       background: #f4f6f9; color: #1c2530; }
.container { max-width: 1100px; margin: 0 auto; padding: 32px; }
h1 { color: #0b3d68; border-bottom: 3px solid #0b3d68; padding-bottom: 10px; }
h2 { color: #0b3d68; margin-top: 42px; border-left: 5px solid #2a8fbd; padding-left: 12px; }
h3 { color: #17506e; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 24px 0; font-size: 13.5px; background: white; }
th, td { border: 1px solid #d8dfe6; padding: 7px 10px; text-align: right; }
th { background: #0b3d68; color: white; text-align: center; }
td:first-child, th:first-child { text-align: left; }
.card { background: white; border-radius: 8px; padding: 18px 22px; margin: 14px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.badge-ok { color: #0a7d3c; font-weight: 600; }
.badge-warn { color: #b45300; font-weight: 600; }
.badge-bad { color: #b30021; font-weight: 600; }
.note { background: #eef6fb; border-left: 4px solid #2a8fbd; padding: 10px 16px; margin: 12px 0; font-size: 14px;}
.limitation { background: #fdf3e7; border-left: 4px solid #cf8b16; padding: 10px 16px; margin: 8px 0; font-size: 14px;}
.toc a { display: block; padding: 3px 0; color: #17506e; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
footer { text-align: center; color: #7a8794; font-size: 12px; margin: 40px 0 20px 0; }
</style>
"""


def df_to_html(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return "<p><i>Sin datos disponibles.</i></p>"
    shown = df.head(max_rows)
    extra = f"<p><i>... ({len(df) - max_rows} filas adicionales no mostradas)</i></p>" if len(df) > max_rows else ""
    return shown.to_html(border=0, na_rep="-") + extra


def build_report(context: dict, output_path: str) -> str:
    """
    context: diccionario con todos los dataframes/valores necesarios para
    poblar el reporte (ver main.py para las claves usadas).
    """
    c = context
    sections = []

    sections.append(f"""
    <h1>Reporte de Prediccion de Clasificaciones ECR &mdash; SBS Peru</h1>
    <p class="note">Proyecto predictivo: informacion financiera disponible en <b>t</b> se usa para
    predecir la clasificacion crediticia publicada por cada Empresa Clasificadora de Riesgo (ECR)
    en una fecha posterior <b>t'</b>. Se modelan 6 agencias de forma independiente
    (APOYO, CLASS, JCR, MICRORATE, MOODYS, PCR) mediante una red neuronal multi-task ordinal.</p>
    <div class="toc">
      <a href="#resumen">1. Resumen del dataset</a>
      <a href="#entidades">2. Entidades y tipos</a>
      <a href="#periodos">3. Periodos</a>
      <a href="#cobertura">4. Cobertura por ECR</a>
      <a href="#distribucion">5. Distribucion de ratings</a>
      <a href="#preparacion">6. Proceso de preparacion</a>
      <a href="#variables">7. Variables seleccionadas</a>
      <a href="#temporal">8. Estrategia temporal</a>
      <a href="#baselines">9. Comparacion de baselines</a>
      <a href="#mlp">10. Resultados del MLP</a>
      <a href="#metricas_ecr">11. Metricas por ECR</a>
      <a href="#metricas_tipo">12. Metricas por tipo de entidad</a>
      <a href="#matrices">13. Matrices de confusion</a>
      <a href="#importancia">14. Importancia de variables</a>
      <a href="#ejemplos">15. Predicciones de ejemplo</a>
      <a href="#limitaciones">16. Limitaciones</a>
      <a href="#conclusion">17. Conclusion</a>
    </div>
    """)

    # 1. Resumen del dataset
    sections.append(f"""
    <h2 id="resumen">1. Resumen del dataset</h2>
    <div class="card">
      <p>Filas del dataset supervisado (entidad x corte con al menos una ECR valida): <b>{c['n_supervised_rows']}</b></p>
      <p>Entidades unicas en el universo de modelado: <b>{c['n_entities']}</b></p>
      <p>Variables financieras candidatas (antes de seleccion): <b>{c['n_features_initial']}</b></p>
      <p>Variables finales seleccionadas para el modelo: <b>{c['n_features_selected']}</b></p>
      <p>Cortes de clasificacion utilizados: {", ".join(config.CORTES_CLASIFICACION.keys())}</p>
    </div>
    """)

    # 2. Entidades y tipos
    sections.append(f"""
    <h2 id="entidades">2. Entidades y tipos</h2>
    <div class="card">{df_to_html(c['entities_by_type'])}</div>
    <h3>Entidades no homologadas (excluidas)</h3>
    <div class="card">{df_to_html(c['no_homologadas'])}</div>
    """)

    # 3. Periodos
    sections.append(f"""
    <h2 id="periodos">3. Periodos</h2>
    <div class="card">{df_to_html(c['periodos_resumen'])}</div>
    """)

    # 4. Cobertura por ECR
    sections.append(f"""
    <h2 id="cobertura">4. Cobertura por ECR</h2>
    <div class="card">{df_to_html(c['cobertura_ecr'])}</div>
    """)

    # 5. Distribucion de ratings
    dist_html = "".join(f"<h3>{ecr}</h3><div class='card'>{df_to_html(df)}</div>"
                         for ecr, df in c['distribucion_ratings'].items())
    sections.append(f"""
    <h2 id="distribucion">5. Distribucion de ratings</h2>
    {dist_html}
    """)

    # 6. Proceso de preparacion
    sections.append(f"""
    <h2 id="preparacion">6. Proceso de preparacion</h2>
    <div class="card">
      <ol>
        <li>Parsing robusto de 8 archivos SBS (HTML disfrazado de .xls, encoding UTF-8 explicito).</li>
        <li>Homologacion de entidades: normalizacion + diccionario explicito de excepciones; matching
            fallido -> <code>entidades_no_homologadas.csv</code> (sin fuzzy matching silencioso).</li>
        <li>Limpieza de rating: eliminacion de modificadores de perspectiva (&uarr;/&darr;) y de
            tokens invalidos (retirado, etc.).</li>
        <li>Escala ordinal LOCAL por ECR (cada agencia se juzga contra su propio conjunto de
            categorias observadas).</li>
        <li>Integracion temporal: para cada (entidad, corte) se busca el ultimo dato financiero
            estrictamente anterior a la fecha de clasificacion, dentro de una ventana de
            {config.MAX_LOOKBACK_MONTHS} meses.</li>
        <li>Feature engineering: niveles + variaciones (1/3/6 meses) + tendencia de 6 meses, sobre
            columnas TOTAL (se excluyen los desgloses MN/ME del arbol de lags para controlar
            dimensionalidad).</li>
        <li>Seleccion de variables ajustada EXCLUSIVAMENTE en TRAIN: filtro de casi-constantes,
            imputacion por mediana, filtro de correlacion, e importancia de RandomForest.</li>
      </ol>
    </div>
    """)

    # 7. Variables seleccionadas
    sections.append(f"""
    <h2 id="variables">7. Variables seleccionadas</h2>
    <div class="card">{df_to_html(c['selected_features_df'], max_rows=60)}</div>
    """)

    # 8. Estrategia temporal
    sections.append(f"""
    <h2 id="temporal">8. Estrategia temporal</h2>
    <div class="card">
      <p>Holdout final (fuera de muestra, nunca visto durante entrenamiento/seleccion): <b>{c['holdout_corte']}</b></p>
      <p>Corte de validacion (early stopping del modelo final): <b>{c['final_val_corte']}</b></p>
      <p>Cortes de entrenamiento del modelo final: {", ".join(c['final_train_cortes'])}</p>
      <h3>Validacion walk-forward (metodologia)</h3>
      {df_to_html(c['walk_forward_summary'])}
    </div>
    """)

    # 9. Comparacion de baselines
    sections.append(f"""
    <h2 id="baselines">9. Comparacion de baselines (holdout {c['holdout_corte']})</h2>
    <div class="card">{df_to_html(c['baseline_comparison'])}</div>
    <div class="note">{c['persistence_vs_mlp_note']}</div>
    """)

    # 10. Resultados del MLP
    sections.append(f"""
    <h2 id="mlp">10. Resultados del MLP (holdout {c['holdout_corte']})</h2>
    <div class="card">{df_to_html(c['mlp_holdout_metrics'])}</div>
    """)

    # 11. Metricas por ECR (ya cubierto arriba, se repite version detallada walk-forward)
    sections.append(f"""
    <h2 id="metricas_ecr">11. Metricas por ECR (promedio walk-forward)</h2>
    <div class="card">{df_to_html(c['metrics_by_ecr_walkforward'])}</div>
    """)

    # 12. Metricas por tipo de entidad
    sections.append(f"""
    <h2 id="metricas_tipo">12. Metricas por tipo de entidad (holdout)</h2>
    <div class="card">{df_to_html(c['metrics_by_entity_type'])}</div>
    """)

    # 13. Matrices de confusion
    cm_html = "".join(f"<h3>{ecr}</h3><div class='card'>{df_to_html(df)}</div>"
                       for ecr, df in c['confusion_matrices'].items())
    sections.append(f"""
    <h2 id="matrices">13. Matrices de confusion (holdout, escala local por ECR)</h2>
    {cm_html}
    """)

    # 14. Importancia de variables
    imp_html = "".join(f"<h3>{ecr}</h3><div class='card'>{df_to_html(df.head(15))}</div>"
                        for ecr, df in c['importance_by_ecr'].items())
    sections.append(f"""
    <h2 id="importancia">14. Importancia de variables (permutation importance)</h2>
    <div class="note">La importancia reportada es <b>predictiva</b>, no causal.</div>
    {imp_html}
    """)

    # 15. Predicciones de ejemplo
    sections.append(f"""
    <h2 id="ejemplos">15. Predicciones de ejemplo (holdout {c['holdout_corte']})</h2>
    <div class="card">{df_to_html(c['predictions_sample'], max_rows=25)}</div>
    """)

    # 16. Limitaciones
    sections.append(f"""
    <h2 id="limitaciones">16. Limitaciones</h2>
    <div class="limitation">
      <ul>
        <li>El numero de cortes de clasificacion disponibles es reducido (8 cortes semestrales),
            por lo que la validacion walk-forward tiene pocos folds y las metricas de holdout
            estan sujetas a alta varianza muestral.</li>
        <li>Varias ECR (JCR, MICRORATE, CLASS) cubren pocas entidades, lo que limita la
            capacidad del modelo de aprender patrones robustos para esas agencias especificas.</li>
        <li>La fecha exacta de cada corte de clasificacion se aproximo al ultimo dia del mes
            correspondiente (MAR/SEP), ya que los archivos SBS no traen una fecha exacta
            embebida.</li>
        <li>La ventana de busqueda de dato financiero anterior al corte se limito a
            {config.MAX_LOOKBACK_MONTHS} meses; entidades sin dato financiero reciente quedan
            fuera de esa observacion (no se inventa informacion).</li>
        <li>La importancia de variables es asociativa/predictiva, no debe interpretarse como
            relacion causal.</li>
      </ul>
    </div>
    """)

    # 17. Conclusion
    sections.append(f"""
    <h2 id="conclusion">17. Conclusion</h2>
    <div class="card">
      <p>{c['conclusion_text']}</p>
    </div>
    <footer>Reporte generado automaticamente &mdash; Proyecto de prediccion de clasificaciones ECR (SBS Peru)</footer>
    """)

    full_html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>Reporte ECR SBS</title>{CSS}</head>
<body><div class="container">{''.join(sections)}</div></body></html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    log(f"Reporte HTML generado en: {output_path}")
    return output_path
