"""SheafProbe data layer: frozen schema, synthetic generator, and real-data loaders."""
from __future__ import annotations

from .schema import (
    Sample,
    Dataset,
    DataUnavailable,
    base_mask,
    one_hot,
)
from .synthetic import generate_dataset, confounds
from .loaders import load_openknot, load_ribonanza, load_openvaccine

__all__ = [
    "Sample",
    "Dataset",
    "DataUnavailable",
    "base_mask",
    "one_hot",
    "generate_dataset",
    "confounds",
    "load_openknot",
    "load_ribonanza",
    "load_openvaccine",
]
