"""SMILES decoding from latent space module."""
import sys
from itertools import count
from typing import Any, List

import torch
from rdkit import Chem
from paccmann_chemistry.models.vae import TeacherVAE
from paccmann_chemistry.utils import get_device
from paccmann_chemistry.utils.search import SamplingSearch

device = get_device()


def get_stack_size(size_hint: int = 2) -> int:
    """Stack size from caller's frame from: https://stackoverflow.com/a/47956089.

    Args:
        size_hint: hint for the stack size. Defaults to 2.

    Returns:
        size of the stack.
    """
    get_frame = sys._getframe
    frame = None
    try:
        while True:
            frame = get_frame(size_hint)
            size_hint *= 2
    except ValueError:
        if frame:
            size_hint //= 2
        else:
            while not frame:
                size_hint = max(2, size_hint // 2)
                try:
                    frame = get_frame(size_hint)
                except ValueError:
                    continue

    for size in count(size_hint):
        frame = frame.f_back
        if not frame:
            return size


class SmilesGenerator:
    """ Smiles Generator """

    def __init__(
        self,
        model: TeacherVAE,
        search=SamplingSearch(),
        generated_length: int = 100,
        max_stack_size: int = 50,
    ):
        """
        Initialization.

        Args:
            model: loaded pretrained SVAE model with its parameters and smiles_language.
            search: search used in decoding SMILES.
            generated_length: length of the generated SMILES string.
            max_stack_size: maximum stack size.
        """
        self.model = model
        self.search = search
        self.generated_length = generated_length
        self.max_stack_size = max_stack_size

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
            generate_len=self.generated_length,
        )

        generated_molecules = []
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
            if get_stack_size() < self.max_stack_size:
                generated_molecules = (
                    smiles if smiles else self.generate_smiles(latent_point)
                )
        else:
            generated_molecules = molecules_numerical
        return generated_molecules
