[![Python package](https://github.com/PaccMann/paccmann_gp/actions/workflows/python-package.yml/badge.svg)](https://github.com/PaccMann/paccmann_gp/actions/workflows/python-package.yml)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# paccmann_gp

Bayesian Optimisation with Gaussian Processes for molecular generative models.

## Installation

Create a conda environment:

```sh
conda env create -f conda.yml
```

Activate the environment:

```sh
conda activate paccmann_gp
```

## Example usage

In the `examples` directory is an example script [example.py](./examples/example.py) that makes use of `paccmann_gp` for a combined optimisation for QED, SAscore and affinity to the transcription factor ERG.

```console
(paccmann_gp) $ python examples/example.py -h
usage: example.py [-h]
                    svae_path affinity_path
                    optimisation_name


positional arguments:
  svae_path          Path to downloaded SVAE model.
  affinity_path      Path to the downloaded affinity prediction model.
  optimisation_name  Name for the optimisation.
```

The trained SVAE and affinity models can be downloaded from the SELFIESVAE and affinity folders located [here](https://ibm.ent.box.com/v/paccmann-sarscov2/folder/122603752964).

## Citation
If you use this repo in your projects, please temporarily cite the following:

```bib
@article{born2021active,
  title={Active site sequence representation of human kinases outperforms full sequence for affinity prediction and inhibitor generation: 3D effects in a 1D model},
  author={Born, Jannis and Huynh, Tien and Stroobants, Astrid and Cornell, Wendy and Manica, Matteo},
  year={2021}
}
```