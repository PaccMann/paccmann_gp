"""Minimization of a function callable with a SMILES string."""
from typing import Any, Callable, Dict, Iterable

import torch
from loguru import logger

from .minimization_function import DecoderBasedMinimization
from .smiles_generator import SmilesGenerator


class CallableMinimization(DecoderBasedMinimization):
    """Generic minimization function that works on any callable."""

    def __init__(
        self,
        smiles_decoder: SmilesGenerator,
        evaluator: Callable[[str, Iterable[Any]], float],
        batch_size: int,
        mode: str = "min",
        evaluator_kwargs: Dict = {},
    ):
        """
        Initialize a generic minization function for optimization.

        Args:
            smiles_decoder: a SMILES generator.
            evaluator (Callable[str]): Function sujbect to minimization, has
                to be callable with a str and will return a float.
            mode (str): Whether output of function should be minimized (default) or
                maximized.
            batch_size: size of the batch for evaluation.
            evaluator_kwargs (Dict): A dict of additional and fixed inputs
                passed to evaluation function. E.g., if evaluator is bimodal
                another modality such as a protein string could be passed here.
                NOTE: These values will be passed *after* the SMILES input.
        """
        super(CallableMinimization, self).__init__(smiles_decoder)
        self.evaluator = evaluator
        self.mode = mode
        if mode not in ["min", "max"]:
            raise ValueError(f"Unknown mode {mode}")
        self.batch = batch_size
        self.evaluator_kwargs = evaluator_kwargs

    def evaluate(self, point: Any) -> float:
        """
        Evaluate a point.

        Args:
            point: point to evaluate.

        Returns:
            evaluation for the given point.
        """
        latent_point = torch.tensor([[point]])
        batch_latent = latent_point.repeat(1, self.batch, 1)

        smiles = self.generator.generate_smiles(batch_latent)

        values = []
        for smile in smiles:
            try:
                values.append(self.evaluator(smile, **self.evaluator_kwargs))
            except Exception:
                values.append(0)
                logger.info("Calculation failed.")

        if self.mode == "max":
            return 1 - (sum(values) / len(values))
        elif self.mode == "min":
            return sum(values) / len(values)
