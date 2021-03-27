"""Generate smiles from latent code with BO for MW."""
import json
import logging
import os
import sys
import argparse
import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from skopt import gp_minimize

from paccmann_chemistry.models.training import get_data_preparation
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
    disable_rdkit_logging()
    device = get_device()

    model_path = parser_namespace.model_path
    weights_path = os.path.join(model_path, 'weights', 'best_rec.pt')

    params = dict()
    with open(os.path.join(model_path, 'model_params.json'), 'r') as f:
        params.update(json.load(f))

    smiles_language = SMILESLanguage.load(
        os.path.join(model_path, 'selfies_language.pkl'))


    # initialize encoder, decoder, testVAE, and GP_generator_MW
    gru_encoder = StackGRUEncoder(params).to(device)
    gru_decoder = StackGRUDecoder(params).to(device)
    gru_vae = TeacherVAE(gru_encoder, gru_decoder).to(device)
    gru_vae.load_state_dict(torch.load(weights_path, map_location=device))
    params.update({
        'vocab_size': smiles_language.number_of_tokens,
        'pad_index': smiles_language.padding_index
    })

    gru_vae._associate_language(smiles_language)
    gru_vae.eval()
    gru_vae.to(device)
    GP_BO=GP_generator_for_MW(gru_vae)

    ## Test without optimisation and random latent point
    latent_point = torch.randn(1,1, 256) #random latent-point
    smiles=GP_BO.generate_smiles(latent_point)
    mweights=[MolWt(Chem.MolFromSmiles(smile)) for smile in smiles]
    print('MW without optimisation:', mweights[0])

    ## With Optimisation_process
    #select target Molecular weight
    target=300
    print('Target MW:',target)

    res1=GP_BO.Optimisation_process(target)

    Optimised_smiles=[]
    Optimised_mws=[]

    for i in range(10):
        smilesOPT=GP_BO.generate_smiles(torch.tensor([[res1.x]]))

        mweights=[MolWt(Chem.MolFromSmiles(smile)) for smile in smilesOPT]
        Optimised_mws.append(mweights[0])
        Optimised_smiles.append(smilesOPT[0])
    print('molecular weights',Optimised_mws)
    print('Average MW with optimisation:', sum(Optimised_mws)/len(Optimised_mws))

#END main function


if __name__ == "__main__":
    main(parser.parse_args())
