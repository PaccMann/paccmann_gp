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


    # initialize encoder and decoder
    gru_encoder = StackGRUEncoder(params).to(device)
    gru_decoder = StackGRUDecoder(params).to(device)
    gru_vae = TeacherVAE(gru_encoder, gru_decoder).to(device)
    gru_vae.load_state_dict(torch.load(weights_path, map_location=device))

    # Updating the vocab size will break the model
    params.update({
        'vocab_size': smiles_language.number_of_tokens,
        'pad_index': smiles_language.padding_index
    })


    gru_vae.eval()
    gru_vae.to(device)

    ## Test without optimisation and random latent point
    latent_point = torch.randn(1,1, 256) #random latent-point
    smiles=[]

    while smiles ==[]: #loop as some molecules are not valid smiles
        mols_numerical=gru_vae.generate(latent_point,     prime_input=torch.LongTensor([smiles_language.start_index]), end_token=torch.LongTensor([smiles_language.stop_index]))

        smiles=numerical_to_cleaned_smiles(mols_numerical,smiles_language)

    mweights=[MolWt(Chem.MolFromSmiles(smile)) for smile in smiles]

    ## With BO
    #select target Molecular weight
    target=300
    print('Target MW:',target)

    #Specify minimisation function for BO
    def MW_minimisation(ar,target=target,gru_vae=gru_vae, smiles_language=smiles_language):
        latent_point=torch.tensor([[ar]])
        smiles=[]
        while smiles ==[]: #loop as some molecules are not valid smiles
            mols_numerical=gru_vae.generate(latent_point,     prime_input=torch.LongTensor([smiles_language.start_index]), end_token=torch.LongTensor([smiles_language.stop_index]))

            smiles=numerical_to_cleaned_smiles(mols_numerical,smiles_language)

        mweights=[MolWt(Chem.MolFromSmiles(smile)) for smile in smiles]
        return abs(target-mweights[0]) # Different formula?

    #BO
    res = gp_minimize(MW_minimisation,# the function to minimize
                  [(-5.0, 5.0)]*256,  # Boundaries
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization p
                  random_state=1234)

#Questions: Not too sure about input and boundaries?

    #Generate 10 Smiles for optimised latent point
    Optimised_smiles=[]
    Optimised_mws=[]
    for i in range(10):
        smiles=[]
        while smiles ==[]:
            mols_numerical=gru_vae.generate(torch.tensor([[res.x]]),     prime_input=torch.LongTensor([smiles_language.start_index]), end_token=torch.LongTensor([smiles_language.stop_index]))
            smiles=numerical_to_cleaned_smiles(mols_numerical,smiles_language)


        mweights=[MolWt(Chem.MolFromSmiles(smile)) for smile in smiles]
        Optimised_mws.append(mweights[0])
        Optimised_smiles.append(smiles[0])

    print('Average MW with optimisation:', sum(Optimised_mws)/len(Optimised_mws))

#END main function



#function for converting numerical output of decoder to cleaned smiles
def numerical_to_cleaned_smiles(numeric_mol,smiles_language):
    smiles_num_tuple = [(
    smiles_language.token_indexes_to_smiles(num_mol.tolist()),
        torch.cat([num_mol.long(),
        torch.tensor(2 * [smiles_language.stop_index])]
        )) for num_mol in iter(numeric_mol)]

    numericals = [sm[1] for sm in smiles_num_tuple]

    smiles = [smiles_language.selfies_to_smiles(sm[0])
            for sm in smiles_num_tuple]
    imgs = [Chem.MolFromSmiles(s, sanitize=True) for s in smiles]

    valid_idxs = [ind for ind in range(len(imgs)) if imgs[ind] is not None]

    smiles = [
            smiles[ind] for ind in range(len(imgs))
            if not ( imgs[ind] is None)
        ]
    return smiles



if __name__ == "__main__":
    main(parser.parse_args())
