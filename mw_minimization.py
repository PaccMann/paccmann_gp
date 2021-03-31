"""MW minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from smiles_generator import SmilesGenerator

class MWMinization(SmilesGenerator):
    """ Minimization function for MW"""
    def __init__(self, model):
        """
        Initialization.

        Arguments:
            SVAEmodel: the loaded pretrained SVAE model with its parameters and smiles_language.
        """

        super().__init__(model)

    def set_target(self,target):
        """
        Set the target MW.

        Arguments:
            target: The target MW (float)
        """

        self.target=target

    def evaluate(self, latentpoint):
        """
        Evaluation of the MW minimization function.

        Arguments:
            latentpoint: The latent coordinate (list of size latent_dim)

        Returns:
            The difference between the target MW and actual MW at latentpoint
        """

        latent_point=torch.tensor([[latentpoint]])
        smiles=self.generate_smiles(latent_point)
        mweight=[MolWt(Chem.MolFromSmiles(smile)) for smile in smiles]

        return abs(self.target-(sum(mweight)/len(mweight)))
