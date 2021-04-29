"""Generate smiles from latent code with BO for MW."""
import json
import logging
import os
import sys
import argparse
import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, qed
import csv
from pytoda.transforms import ToTensor, LeftPadding
from paccmann_generator.drug_evaluators.sas import SAS
from paccmann_chemistry.models.vae import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils import collate_fn, get_device, disable_rdkit_logging
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.proteins.protein_language import ProteinLanguage
from paccmann_predictor.models import MODEL_FACTORY
from gp_optimizer import GPOptimizer
from smiles_generator import SmilesGenerator
from mw_minimization import MWMinimization
from sa_minimization import SAMinimization
from qed_minimization import QEDMinimization
from affinity_minimization import AffinityMinimization
from combined_minimization import CombinedMinimization
from loguru import logger
import pickle

# parser
parser = argparse.ArgumentParser(description="SVAE SMILE generation with BO.")
parser.add_argument(
    "svae_path", type=str, help="Path to the trained model (SELFIES VAE)"
)
parser.add_argument(
    "affinity_path", type=str, help="Path to the trained model (SELFIES VAE)."
)
parser.add_argument(
    "optimisation_name", type=str, help="Name for optimisation."
)



def main(parser_namespace):
    # Model loading
    disable_rdkit_logging()
    affinity_path = parser_namespace.affinity_path
    svae_path = parser_namespace.svae_path
    svae_weights_path = os.path.join(svae_path, "weights", "best_rec.pt")
    results_file_name = parser_namespace.optimisation_name

    logger.add(results_file_name + ".log", rotation="10 MB")

    svae_params = dict()
    with open(os.path.join(svae_path, "model_params.json"), "r") as f:
        svae_params.update(json.load(f))

    smiles_language = SMILESLanguage.load(
        os.path.join(svae_path, "selfies_language.pkl")
    )

    # initialize encoder, decoder, testVAE, and GP_generator_MW
    gru_encoder = StackGRUEncoder(svae_params)
    gru_decoder = StackGRUDecoder(svae_params)
    gru_vae = TeacherVAE(gru_encoder, gru_decoder)
    gru_vae.load_state_dict(torch.load(svae_weights_path, map_location=get_device()))

    gru_vae._associate_language(smiles_language)
    gru_vae.eval()

    smiles_generator = SmilesGenerator(gru_vae)

    with open(os.path.join(affinity_path, "model_params.json")) as f:
        predictor_params = json.load(f)
    affinity_predictor = MODEL_FACTORY["bimodal_mca"](predictor_params)
    affinity_predictor.load(
        os.path.join(
            affinity_path,
            f"weights/best_{predictor_params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt",
        ),
        map_location=get_device(),
    )
    affinity_protein_language = ProteinLanguage.load(
        os.path.join(affinity_path, "protein_language.pkl")
    )
    affinity_smiles_language = SMILESLanguage.load(
        os.path.join(affinity_path, "smiles_language.pkl")
    )
    affinity_predictor._associate_language(affinity_smiles_language)
    affinity_predictor._associate_language(affinity_protein_language)
    affinity_predictor.eval()

    erg_protein = "MASTIKEALSVVSEDQSLFECAYGTPHLAKTEMTASSSSDYGQTSKMSPRVPQQDWLSQPPARVTIKMECNPSQVNGSRNSPDECSVAKGGKMVGSPDTVGMNYGSYMEEKHMPPPNMTTNERRVIVPADPTLWSTDHVRQWLEWAVKEYGLPDVNILLFQNIDGKELCKMTKDDFQRLTPSYNADILLSHLHYLRETPLPHLTSDDVDKALQNSPRLMHARNTGGAAFIFPNTSVYPEATQRITTRPDLPYEPPRRSAWTGHGHPTPQSKAAQPSPSTVPKTEDQRPQLDPYQILGPTSSRLANPGSGQIQLWQFLLELLSDSSNSSCITWEGTNGEFKMTDPDEVARRWGERKSKPNMNYDKLSRALRYYYDKNIMTKVHGKRYAYKFDFHGIAQALQPHPPESSLYKYPSDLPYMGSYHAHPQKMNFVAPHPPALPVTSSSFFAAPNPYWNSPTGGIYPNTRLPTSHMPSHLGTYY"

    target_minimization_function = AffinityMinimization(
        smiles_generator, 30, affinity_predictor, erg_protein
    )
    qed_function = QEDMinimization(smiles_generator, 30)
    sa_function = SAMinimization(smiles_generator, 30)
    combined_minimization = CombinedMinimization(
        [target_minimization_function, qed_function, sa_function], [0.75, 1, 0.5], 1
    )
    target_optimizer = GPOptimizer(combined_minimization.evaluate)

    params = dict(
        dimensions=[(-5.0, 5.0)] * 256,
        acq_func="EI",
        n_calls=20,
        n_initial_points=19,
        initial_point_generator="random",
        random_state=1234,
    )
    logger.info('Optimisation parameters: {params}',params=params)

    #Optimisation
    for j in range(5):
        res = target_optimizer.optimize(params)
        latent_point = torch.tensor([[res.x]])

        with open(results_file_name + '_LP' + str(j+1) + '.pkl', 'wb') as f:
            pickle.dump(latent_point, f)

        smile_set = set()

        while len(smile_set) < 20:
            smiles = smiles_generator.generate_smiles(latent_point.repeat(1, 30, 1))
            smile_set.update(set(smiles))
        smile_set = list(smile_set)

        pad_smiles_predictor = LeftPadding(
            affinity_predictor.smiles_padding_length,
            affinity_predictor.smiles_language.padding_index,
        )
        to_tensor = ToTensor(get_device())
        smiles_num = [
            torch.unsqueeze(
                to_tensor(
                    pad_smiles_predictor(
                        affinity_predictor.smiles_language.smiles_to_token_indexes(
                            smile
                        )
                    )
                ),
                0,
            )
            for smile in smile_set
        ]

        smiles_tensor = torch.cat(smiles_num, dim=0)

        pad_protein_predictor = LeftPadding(
            affinity_predictor.protein_padding_length,
            affinity_predictor.protein_language.padding_index,
        )

        protein_num = torch.unsqueeze(
            to_tensor(
                pad_protein_predictor(
                    affinity_predictor.protein_language.sequence_to_token_indexes(
                        [erg_protein]
                    )
                )
            ),
            0,
        )
        protein_num = protein_num.repeat(len(smile_set), 1)

        with torch.no_grad():
            pred, pred_dict = affinity_predictor(smiles_tensor, protein_num)
        affinities = torch.squeeze(pred, 1).numpy()

        sas = SAS()
        sa_scores = [sas(smile) for smile in smile_set]
        qed_scores = [qed(Chem.MolFromSmiles(smile)) for smile in smile_set]

        #save to file
        file = results_file_name + str(j + 1) + ".txt"
        logger.info('Creating {file}', file=file)

        with open(file, "w") as f:
            f.write(
                f'{"point":<10}{"Affinity":<10}{"QED":<10}{"SA":<10}{"smiles":<15}\n'
            )
            for i in range(20):
                dat = [i + 1, affinities[i], qed_scores[i], sa_scores[i], smile_set[i]]
                f.write(
                    f'{dat[0]:<10}{"%.3f"%dat[1]:<10}{"%.3f"%dat[2]:<10}{"%.3f"%dat[3]:<10}{dat[4]:<15}\n'
                )


if __name__ == "__main__":
    main(parser.parse_args())
