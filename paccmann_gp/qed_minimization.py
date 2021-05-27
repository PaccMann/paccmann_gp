"""QED minimization module."""
import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from loguru import logger
from typing import Any
from .minimization_function import DecoderBasedMinimization
from .smiles_generator import SmilesGenerator


class QEDMinimization(DecoderBasedMinimization):
    """Minimization function for QED."""

    def __init__(self, smiles_decoder: SmilesGenerator, batch_size: int):
        """
        Initialize a minization function for QED optimization.

        Args:
            smiles_decoder: a SMILES generator.
            batch_size: size of the batch for evaluation.
        """
        super(QEDMinimization, self).__init__(smiles_decoder)
        self.batch = batch_size

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

        qed_values = []
        for smile in smiles:
            try:
                qed_values.append(qed(Chem.MolFromSmiles(smile)))
            except Exception:
                qed_values.append(0)
                logger.warning("QED calculation failed.")

        return 1 - (sum(qed_values) / len(qed_values))
