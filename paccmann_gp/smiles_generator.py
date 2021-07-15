"""SMILES decoding from latent space module."""
from typing import Any, List

import torch
from rdkit import Chem
from paccmann_chemistry.models.vae import TeacherVAE
from paccmann_chemistry.utils import get_device
from paccmann_chemistry.utils.search import SamplingSearch

device = get_device()


class SmilesGenerator:
    """ Smiles Generator """

    def __init__(self, model: TeacherVAE, search=SamplingSearch()):
        """
        Initialization.

        Args:
            model: loaded pretrained SVAE model with its parameters and smiles_language.
            search: search used in decoding SMILES.
        """
        self.model = model
        self.search = search

    def generate_smiles(self, latent_point: Any, to_smiles: bool = True) -> List[Any]:
        """
        Generate a smiles or selfies code from latent latent_point.

        Args:
            latent_point: the input latent point as tensor with shape `[1,batch_size,latent_dim]`.
            to_siles: boolean to specify if output should be SMILES (True) or numerical sequence (False).

        Returns:
            molecules represented as SMILES or tokens.
        """
        molecules_numerical = self.model.generate(
            latent_point,
            prime_input=torch.LongTensor([self.model.smiles_language.start_index]).to(
                device
            ),
            end_token=torch.LongTensor([self.model.smiles_language.stop_index]).to(
                device
            ),
            search=self.search,
        )

        # convert numerical output to smiles
        if to_smiles:
            smiles_num = [
                self.model.smiles_language.token_indexes_to_smiles(
                    molecule_numerical.tolist()
                )
                for molecule_numerical in iter(molecules_numerical)
            ]

            smiles = [
                self.model.smiles_language.selfies_to_smiles(sm) for sm in smiles_num
            ]
            molecules = []
            for a_smiles in smiles:
                try:
                    molecules.append(Chem.MolFromSmiles(a_smiles, sanitize=True))
                except Exception:
                    molecules.append(None)

            smiles = [
                smiles[index]
                for index in range(len(molecules))
                if not (molecules[index] is None)
            ]
            return smiles if smiles else self.generate_smiles(latent_point)
        else:
            return molecules_numerical
