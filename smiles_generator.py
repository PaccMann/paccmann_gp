"""Smiles Decoder from Latent module."""

import torch
from paccmann_chemistry.utils.search import SamplingSearch
from rdkit import Chem


class SmilesGenerator:
    """ Smiles Generator """
    def __init__(self, model,search=SamplingSearch()):
        """
        Initialization.

        Arguments:
            SVAEmodel: the loaded pretrained SVAE model with its parameters and smiles_language.
        """

        self.model = model
        self.search=search

    def generate_smiles(
        self,
        latent_point,
        tosmiles=True):
        """
        Generate a smiles or selfies code from latent latent_point.

        Arguments:
            latent_point: the input latent point as tensor with shape '[1,batch_size,latent_dim]'
            tomsiles: boolean to specify if output should be smiles (True) or   numerical sequence (False)

        Returns:
            Smiles or numerical representation (list)
        """

        mols_numerical=self.model.generate(latent_point,             prime_input=torch.LongTensor([self.model.smiles_language.start_index]), end_token=torch.LongTensor([self.model.smiles_language.stop_index]),search=self.search)

        # Convert numerical ooutput to smiles
        if tosmiles==True:
            smiles_num = [self.model.smiles_language.token_indexes_to_smiles(num_mol.tolist()) for num_mol in iter(mols_numerical)]

            smiles = [self.model.smiles_language.selfies_to_smiles(sm) for sm in smiles_num]
            imgs=[]
            for s in smiles:
                try: imgs.append(Chem.MolFromSmiles(s, sanitize=True))
                except: pass


            valid_idxs = [ind for ind in range(len(imgs)) if imgs[ind] is not None]

            smiles = [smiles[ind] for ind in range(len(imgs)) if not ( imgs[ind] is None)]

            return self.generate_smiles(latent_point) if smiles==[] else smiles
        else:
            return mols_numerical
