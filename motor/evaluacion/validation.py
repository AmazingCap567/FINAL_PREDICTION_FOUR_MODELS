# -*- coding: utf-8 -*-
"""
validation.py
=============
Estrategia de validacion TEMPORAL (nunca aleatoria) para evitar que
observaciones futuras de una misma entidad entrenen el modelo antes de la
fecha correspondiente.

- HOLDOUT_CORTE: el corte mas reciente (SEP2025) se reserva integramente
  para la evaluacion final fuera de muestra. Nunca se usa en ningun momento
  del entrenamiento, seleccion de variables, ni ajuste de hiperparametros.
- WALK-FORWARD FOLDS: sobre los cortes restantes (MAR2022..MAR2025) se arma
  una validacion walk-forward con ventana de entrenamiento creciente: se
  entrena con todos los cortes anteriores y se evalua sobre el siguiente.
- VAL_CORTE (para early stopping del modelo final): el ultimo corte antes
  del holdout (MAR2025) se usa como validacion de entrenamiento del modelo
  final; el modelo final se entrena con MAR2022..SEP2024.
"""

from __future__ import annotations
from .. import config

ALL_CORTES = list(config.CORTES_CLASIFICACION.keys())  # orden cronologico
HOLDOUT_CORTE = ALL_CORTES[-1]              # SEP2025
FINAL_VAL_CORTE = ALL_CORTES[-2]            # MAR2025
FINAL_TRAIN_CORTES = ALL_CORTES[:-2]        # MAR2022 .. SEP2024

# Walk-forward: minimo 3 cortes de historia antes de empezar a testear.
WALK_FORWARD_MIN_TRAIN = 3


def build_walk_forward_folds() -> list[dict]:
    """
    Devuelve una lista de folds {train_cortes: [...], test_corte: str}
    usando SOLO los cortes anteriores al holdout final (para no filtrar
    informacion del holdout durante la validacion de metodologia).
    """
    cortes_disponibles = [c for c in ALL_CORTES if c != HOLDOUT_CORTE]
    folds = []
    for i in range(WALK_FORWARD_MIN_TRAIN, len(cortes_disponibles)):
        train_cortes = cortes_disponibles[:i]
        test_corte = cortes_disponibles[i]
        folds.append({"train_cortes": train_cortes, "test_corte": test_corte})
    return folds
