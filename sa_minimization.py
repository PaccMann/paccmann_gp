"""SA minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from minimization_function import DecoderBasedMinimization
from paccmann_generator.drug_evaluators.sas  import  SAS

class SAMinimization(DecoderBasedMinimization):
    """ Minimization function for SA"""
    def __init__(self):
        self.sas = SAS()

    def evaluate(self, point):
        """
        Evaluation of the SA score minimization function.

        Arguments:
            point: The latent coordinate (list of size latent_dim)

        """

        latent_point=torch.tensor([[point]])
        batch_latent=latent_point.repeat(1,25,1)
        smiles = None
        while smiles is None:
            try:
                smiles=self.generator.generate_smiles(batch_latent)
            except:
                pass
        sa_score=[self.sas(smile) for smile in smiles]

        return sum(sa_score)/len(sa_score)
