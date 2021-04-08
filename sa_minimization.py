"""SA minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from minimization_function import DecoderBasedMinimization
from paccmann_generator.drug_evaluators.sas import SAS


class SAMinimization(DecoderBasedMinimization):
    """ Minimization function for SA"""

    def __init__(self, smiles_decoder, batch_size):
        super(SAMinimization, self).__init__(smiles_decoder)
        self.generator = smiles_decoder
        self.batch = batch_size

    def evaluate(self, point):
        """
        Evaluation of the SA score minimization function.

        Arguments:
            point: The latent coordinate (list of size latent_dim)

        """

        latent_point = torch.tensor([[point]])
        batch_latent = latent_point.repeat(1, self.batch, 1)
        smiles = self.generator.generate_smiles(batch_latent)
        sa_score = [SAS(smile) for smile in smiles]

        return sum(sa_score) / len(sa_score)
