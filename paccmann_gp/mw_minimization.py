"""Distance from target molecular weight minimization module."""
from math import exp
from typing import Any

import torch
from rdkit import Chem
from loguru import logger
from rdkit.Chem.Descriptors import MolWt

from .minimization_function import DecoderBasedMinimization
from .smiles_generator import SmilesGenerator


class MWMinimization(DecoderBasedMinimization):
    """Minimization function for MW."""

    def __init__(
        self, smiles_decoder: SmilesGenerator, batch_size: int, target: float
    ) -> None:
        """
        Initialize a distance from target molecular weight function.

        Args:
            smiles_decoder: a SMILES generator.
            batch_size: size of the batch for evaluation.
            target: target molecular weight.
        """
        super(MWMinimization, self).__init__(smiles_decoder)
        self.batch = batch_size
        self.target = target

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

        mweights = []
        for smile in smiles:
            try:
                mweights.append(MolWt(Chem.MolFromSmiles(smile)))
            except Exception:
                logger.warning("MW calculation failed.")

        if len(mweights) > 0:
            return 1.0 - exp(-abs(self.target - (sum(mweights) / len(mweights))))
        else:
            return 1.0
