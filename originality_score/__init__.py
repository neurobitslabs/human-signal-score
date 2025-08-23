"""
Originality Score
=================

This package provides utilities for measuring the diversity of textual datasets and
computing a simple "Human Signal Score".  The intent is to offer an open and
transparent way to estimate the risk of model collapse due to training on
low–diversity or synthetic data.

See the README for details on how the metrics are calculated.
"""

from .score import (
    compute_dataset_entropy,
    compute_perplexity,
    compute_embedding_diversity,
    compute_human_signal_score,
)

__all__ = [
    "compute_dataset_entropy",
    "compute_perplexity",
    "compute_embedding_diversity",
    "compute_human_signal_score",
]