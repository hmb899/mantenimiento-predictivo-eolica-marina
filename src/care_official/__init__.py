# implementacion propia de la metrica CARE, la del benchmark oficial de Guck et al. (2024)
# viene de PruebasTFM/CAREBaseline/care_to_compare/src/care/, validada por 82 tests unitarios
# aqui solo se adaptaron los imports al paquete, la logica de la metrica es la misma
#
# este es el CARE oficial, aparte del propio del proyecto, src/models/evaluator.py
# (CAREEvaluator), para poder reportar los dos uno al lado del otro
"""metrica CARE oficial (Coverage, Accuracy, Reliability, Earliness)."""
from __future__ import annotations

from .care import (
    CAREResult,
    compute_care,
    weighted_average,
    prepare_dataset_inputs,
    prepare_normal_dataset_inputs,
)
from .labels import NORMAL_STATUS, compute_earliness_weights, EventWindow

__all__ = [
    "CAREResult",
    "compute_care",
    "weighted_average",
    "prepare_dataset_inputs",
    "prepare_normal_dataset_inputs",
    "NORMAL_STATUS",
    "compute_earliness_weights",
    "EventWindow",
]
