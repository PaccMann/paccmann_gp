"""Target affinity minimization Class module."""

import torch
from rdkit import Chem
from minimization_function import DecoderBasedMinimization
from pytoda.transforms import ToTensor, LeftPadding


class AffinityMinimization(DecoderBasedMinimization):
    """ Minimization function for target affinity"""

    def __init__(self, smiles_decoder, batch_size, affinity_predictor, receptor):

        super(AffinityMinimization, self).__init__(smiles_decoder)

        self.generator = smiles_decoder
        self.batch = batch_size

        self.predictor = affinity_predictor
        # self.device = get_device()
        self.to_tensor = ToTensor()

        self.receptor = receptor

        # protein to tensor
        self.pad_protein_predictor = LeftPadding(
            self.predictor.protein_padding_length,
            self.predictor.protein_language.padding_index,
        )

        self.receptor_numeric = torch.unsqueeze(
            self.to_tensor(
                self.pad_protein_predictor(
                    self.predictor.protein_language.sequence_to_token_indexes(
                        [self.receptor]
                    )
                )
            ),
            0,
        )

        self.pad_smiles_predictor = LeftPadding(
            self.predictor.smiles_padding_length,
            self.predictor.smiles_language.padding_index,
        )

    def evaluate(self, point):
        """
        Evaluation of the target affinity minimization function.

        Arguments:
            point: The latent coordinate (list of size latent_dim)

        """

        latent_point = torch.tensor([[point]])

        batch_latent = latent_point.repeat(1, self.batch, 1)

        smiles = self.generator.generate_smiles(batch_latent)

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

        protein_tensor = self.receptor_numeric.repeat(len(smiles), 1)

        # Affinity predicition
        with torch.no_grad():
            affinity_prediction, pred_dict = self.predictor(
                smiles_tensor, protein_tensor
            )

        return 1 - (sum(torch.squeeze(affinity_prediction, 1).numpy()) / len(smiles))
