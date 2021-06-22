"""SA minimization Class module."""
import torch
from paccmann_generator.drug_evaluators.sas import SAS
from loguru import logger
from typing import Any
from .minimization_function import DecoderBasedMinimization
from .smiles_generator import SmilesGenerator


class SAMinimization(DecoderBasedMinimization):
    """ Minimization function for SA"""

    def __init__(self, smiles_decoder: SmilesGenerator, batch_size: int):
        """
        Initialize a minization function for SA minimization.

        Args:
            smiles_decoder: a SMILES generator.
            batch_size: size of the batch for evaluation.
        """
        super(SAMinimization, self).__init__(smiles_decoder)
        self.batch = batch_size
        self.sascore = SAS()

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

        sa_scores = []
        for smile in smiles:
            try:
                sa_scores.append(self.sascore(smile))
            except Exception:
                sa_scores.append(10)
                logger.warning("SA calculation failed.")

        return sum(sa_scores) / (len(sa_scores) * 10)  # /10 to get number between 0-1
