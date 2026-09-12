# -*- coding: utf-8 -*-
"""
utils.py
========
Funciones auxiliares de proposito general: normalizacion de texto,
logging con formato consistente y utilidades de impresion de tablas.
"""

from __future__ import annotations
import re
import sys
import unicodedata
import datetime as dt


def log(msg: str, level: str = "INFO") -> None:
    """Imprime un mensaje con timestamp y nivel, y hace flush inmediato."""
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")
    sys.stdout.flush()


def section(title: str) -> None:
    """Imprime un encabezado de seccion visualmente distinguible en consola."""
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")
    sys.stdout.flush()


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")
    sys.stdout.flush()


def strip_accents(text: str) -> str:
    """Elimina acentos/diacriticos de una cadena, preservando el resto."""
    if text is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_entity_name(name: str) -> str:
    """
    Normaliza un nombre de entidad para homologacion:
    - Mayusculas
    - Sin acentos
    - Espacios colapsados
    - Sin puntuacion final tipica (S.A., S.A.A., etc. se preservan porque
      a veces son parte distintiva del nombre; solo se recortan espacios y
      puntos duplicados).
    """
    if name is None:
        return ""
    s = strip_accents(str(name)).upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Á", "A")  # salvaguarda adicional
    return s.strip()


def normalize_column_token(text: str) -> str:
    """Normaliza un texto para comparaciones robustas (sin acentos, minuscula, sin espacios extra)."""
    if text is None:
        return ""
    s = strip_accents(str(text)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_rating_modifiers(raw: str) -> str:
    """
    Elimina los modificadores de perspectiva/movimiento de un rating SBS.
    Ejemplos: 'A-↑' -> 'A-', 'C+↓' -> 'C+', 'C↓' -> 'C'.
    Tambien elimina espacios y normaliza a mayusculas.
    """
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    # Elimina flechas unicode de perspectiva
    s = s.replace("↑", "").replace("↓", "").replace("\u2191", "").replace("\u2193", "")
    # Elimina texto entre parentesis (notas ocasionales)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.strip()
    return s


def safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return 100.0 * numerator / denominator
