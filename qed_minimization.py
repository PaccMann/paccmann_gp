"""QED minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from minimization_function import DecoderBasedMinimization

class QEDMinimization(DecoderBasedMinimization):
    """ Minimization function for QED"""

    def evaluate(self, point):
        """
        Evaluation of the QED minimization function.

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
        qed_value=[qed(Chem.MolFromSmiles(smile)) for smile in smiles]

        return 1-(sum(qed_value)/len(qed_value))
