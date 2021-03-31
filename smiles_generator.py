"""Smiles Generator from Latent module."""

import torch
from paccmann_chemistry.utils.search import SamplingSearch
from rdkit import Chem


class SmilesGenerator:
    """ Smiles generator """
    def __init__(self, model):
        """
        Initialization.

        Arguments:
            SVAEmodel: the loaded pretrained SVAE model with its parameters and smiles_language.
        """

        self.model = model

    def generate_smiles(
        self,
        latent_point,
        tosmiles=True,
        search=SamplingSearch()):
        """
        Generate a smiles or selfies code from latent latent_point.

        Arguments:
            latent_point: the input latent point as tensor with shape '[1,1,latent_dim]'
            tomsiles: boolean to specify if output should be smiles (True) or   numerical sequence (False)

        Returns:
            Smiles or numerical representation (list)
        """

        mols_numerical=self.model.generate(latent_point,             prime_input=torch.LongTensor([self.model.smiles_language.start_index]), end_token=torch.LongTensor([self.model.smiles_language.stop_index]),search=search)

        # Convert numerical ooutput to smiles
        if tosmiles==True:
            smiles_num_tuple = [(self.model.smiles_language.token_indexes_to_smiles(num_mol.tolist()),torch.cat([num_mol.long(),torch.tensor(2 * [self.model.smiles_language.stop_index])])) for num_mol in iter(mols_numerical)]

            numericals = [sm[1] for sm in smiles_num_tuple]

            smiles = [self.model.smiles_language.selfies_to_smiles(sm[0]) for sm in smiles_num_tuple]

            imgs = [Chem.MolFromSmiles(s, sanitize=True) for s in smiles]

            valid_idxs = [ind for ind in range(len(imgs)) if imgs[ind] is not None]

            smiles = [smiles[ind] for ind in range(len(imgs)) if not ( imgs[ind] is None)]

            return self.generate_smiles(latent_point) if smiles==[] else smiles
        else:
            return mols_numerical
