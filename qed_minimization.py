"""QED minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from smiles_generator import SmilesGenerator

class QEDMinization(SmilesGenerator):
    """ Minimization function for QED"""
    def __init__(self, model):
        """
        Initialization.

        Arguments:
            SVAEmodel: the loaded pretrained SVAE model with its parameters and smiles_language.
        """

        super().__init__(model)

    def evaluate(self, latentpoint):
        """
        Evaluation of the QED minimization function.

        Arguments:
            latentpoint: The latent coordinate (list of size latent_dim)

        """

        latent_point=torch.tensor([[latentpoint]])
        smiles=self.generate_smiles(latent_point)
        qed_value=[qed(Chem.MolFromSmiles(smile)) for smile in smiles]

        return 1-(sum(qed_value)/len(qed_value))
