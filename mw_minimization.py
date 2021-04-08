"""MW minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from minimization_function import DecoderBasedMinimization


class MWMinimization(DecoderBasedMinimization):
    """ Minimization function for MW"""

    def __init__(self, smiles_decoder, batch_size, target):
        super(MWMinimization, self).__init__(smiles_decoder)

        self.generator = smiles_decoder
        self.batch = batch_size
        self.target = target

    def evaluate(self, point):
        """
        Evaluation of the MW minimization function.

        Arguments:
            point: The latent coordinate (list of size latent_dim)

        Returns:
            The difference between the target MW and actual MW at latentpoint
        """

        latent_point = torch.tensor([[point]])
        batch_latent = latent_point.repeat(1, self.batch, 1)

        smiles = self.generator.generate_smiles(batch_latent)

        mweight = [MolWt(Chem.MolFromSmiles(smile)) for smile in smiles]

        return abs(self.target - (sum(mweight) / len(mweight)))
