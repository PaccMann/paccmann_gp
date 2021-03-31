"""SA minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from smiles_generator import SmilesGenerator
from paccmann_generator.drug_evaluators.sas  import  SAS

class SAMinization(SmilesGenerator):
    """ Minimization function for SA"""
    def __init__(self, model):
        """
        Initialization.

        Arguments:
            SVAEmodel: the loaded pretrained SVAE model with its parameters and smiles_language.
        """

        super().__init__(model)
        self.sas = SAS()

    def evaluate(self, latentpoint):
        """
        Evaluation of the SA score minimization function.

        Arguments:
            latentpoint: The latent coordinate (list of size latent_dim)

        """

        latent_point=torch.tensor([[latentpoint]])
        smiles=self.generate_smiles(latent_point)
        sa_score=[self.sas(smile) for smile in smiles]

        return sum(sa_score)/len(sa_score)
