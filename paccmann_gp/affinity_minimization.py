"""Target affinity minimization module."""
from typing import Any

import torch
from loguru import logger
from pytoda.transforms import LeftPadding, ToTensor

from .minimization_function import DecoderBasedMinimization
from .smiles_generator import SmilesGenerator


class AffinityMinimization(DecoderBasedMinimization):
    """Minimization function for target affinity."""

    def __init__(
        self,
        smiles_decoder: SmilesGenerator,
        batch_size: int,
        affinity_predictor: Any,
        protein: str,
    ) -> None:
        """
        Initialize an affinity minimization function.

        Args:
            smiles_decoder: a SMILES generator.
            batch_size: size of the batch for evaluation.
            affinity_predictor: an affinity predictor.
            protein: string descrition of a protein compatible with the generator and the predictor.
        """
        super(AffinityMinimization, self).__init__(smiles_decoder)

        self.batch = batch_size

        self.predictor = affinity_predictor
        self.to_tensor = ToTensor()

        self.protein = protein

        # protein to tensor
        self.pad_protein_predictor = LeftPadding(
            self.predictor.protein_padding_length,
            self.predictor.protein_language.padding_index,
        )

        self.protein_numeric = torch.unsqueeze(
            self.to_tensor(
                self.pad_protein_predictor(
                    self.predictor.protein_language.sequence_to_token_indexes(
                        self.protein
                    )
                )
            ),
            0,
        )

        self.pad_smiles_predictor = LeftPadding(
            self.predictor.smiles_padding_length,
            self.predictor.smiles_language.padding_index,
        )

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

        if len(smiles) > 1:
            # smiles to tensor for affinity prediction
            smiles_tensor = torch.cat(
                [
                    torch.unsqueeze(
                        self.to_tensor(
                            self.pad_smiles_predictor(
                                self.predictor.smiles_language.smiles_to_token_indexes(
                                    smile
                                )
                            )
                        ),
                        0,
                    )
                    for smile in smiles
                ],
                dim=0,
            )

            protein_tensor = self.protein_numeric.repeat(len(smiles), 1)

            # affinity predicition
            with torch.no_grad():
                try:
                    affinity_prediction, _ = self.predictor(
                        smiles_tensor, protein_tensor
                    )
                except Exception:
                    affinity_prediction = torch.unsqueeze(
                        torch.tensor([0.0] * len(smiles)), 1
                    )
                    logger.warning("Affinity calculation failed.")
            return 1.0 - (
                sum(torch.squeeze(affinity_prediction.cpu(), 1).numpy()) / len(smiles)
            )
        else:
            return 1.0
