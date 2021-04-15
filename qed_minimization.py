"""QED minimization Class module."""

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from minimization_function import DecoderBasedMinimization


class QEDMinimization(DecoderBasedMinimization):
    """ Minimization function for QED"""

    def __init__(self, smiles_decoder, batch_size):
        super(QEDMinimization, self).__init__(smiles_decoder)
        self.generator = smiles_decoder
        self.batch = batch_size

    def evaluate(self, point):
        """
        Evaluation of the QED minimization function.

        Arguments:
            point: The latent coordinate (list of size latent_dim)

        """

        latent_point = torch.tensor([[point]])
        batch_latent = latent_point.repeat(1, self.batch, 1)

        smiles = self.generator.generate_smiles(batch_latent)

        qed_values = []
        for smile in smiles:
            try:
                qed_values.append(qed(Chem.MolFromSmiles(smile)))
            except:
                qed_values.append(0)
                print("QED calculation failed.")

        return 1 - (sum(qed_values) / len(qed_values))
