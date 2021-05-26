# paccmann_gp

Bayesian Optimisation for generative models.

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
