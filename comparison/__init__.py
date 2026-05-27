"""
scorecard.comparison

Multi-model comparison framework for the PD_cust × f(deal) scorecard.
Fits arbitrary candidate models, evaluates discrimination, calibration,
and stability on development and OOT samples, and exports both PBI-ready
data and an interactive HTML prototype.
"""

from .model_comparison import (
    VariableConfig,
    ModelSpec,
    ModelResults,
    ModelComparison,
)

__all__ = [
    "VariableConfig",
    "ModelSpec",
    "ModelResults",
    "ModelComparison",
]