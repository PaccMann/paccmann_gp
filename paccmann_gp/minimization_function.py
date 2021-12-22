"""Minimization function module."""
from typing import Any

from .smiles_generator import SmilesGenerator


class MinimizationFunction:
    """Minimization function."""

    def evaluate(self, point: Any) -> float:
        """
        Evaluate a point.

        Args:
            point: point to evaluate.

        Returns:
            evaluation for the given point.
        """
        raise NotImplementedError("Please Implement this evaluation method")


class DecoderBasedMinimization(MinimizationFunction):
    """Decoder based minization function."""

    def __init__(self, smiles_decoder: SmilesGenerator) -> None:
        """
        Initialize a decoder based minimizer.

        Args:
            smiles_decoder: a decoder based generator.
        """
        self.generator = smiles_decoder
