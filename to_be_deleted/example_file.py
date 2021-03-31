"""Generate smiles from latent code with BO for MW."""
import json
import logging
import os
import sys
import argparse
import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, qed
from skopt import gp_minimize
import csv

from paccmann_generator.drug_evaluators.sas  import  SAS
from paccmann_chemistry.models.vae import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE)
from paccmann_chemistry.utils import (
    collate_fn, get_device, disable_rdkit_logging)
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from GP_generator_MW import GP_generator_for_MW

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('svae')

#parser
parser = argparse.ArgumentParser(description='SVAE SMILE generation with BO.')
parser.add_argument(
    'model_path', type=str, help='Path to the trained model (SELFIES VAE)')



def main(parser_namespace):
    #Model loading

    model_path = parser_namespace.model_path
    weights_path = os.path.join(model_path, 'weights', 'best_rec.pt')

    params = dict()
    with open(os.path.join(model_path, 'model_params.json'), 'r') as f:
        params.update(json.load(f))

    smiles_language = SMILESLanguage.load(
        os.path.join(model_path, 'selfies_language.pkl'))

    latent_point=torch.randn(1,1, 256)
    # initialize encoder, decoder, testVAE, and GP_generator_MW
    gru_encoder = StackGRUEncoder(params)
    gru_decoder = StackGRUDecoder(params)
    gru_vae = TeacherVAE(gru_encoder, gru_decoder)
    gru_vae.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    gru_vae._associate_language(smiles_language)
    gru_vae.eval()
    GP_BO=GP_generator_for_MW(gru_vae)
    SAScore=SAS()

#This works
    smiles2=gru_vae.generate(latent_z=latent_point,prime_input=torch.tensor([gru_vae.smiles_language.start_index]), end_token=torch.tensor([gru_vae.smiles_language.stop_index]))

#This does not work
    smiles3=gru_vae.decoder.generate_from_latent(latent_z=latent_point,prime_input=torch.tensor([gru_vae.smiles_language.start_index]), end_token=torch.tensor([gru_vae.smiles_language.stop_index]))
    print(smiles3)

if __name__ == "__main__":
    main(parser.parse_args())
