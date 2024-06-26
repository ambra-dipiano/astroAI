# astroAI

You may clone the repository and move into its root directory to work.

Step1:
```bash
git clone git@github.com:ambra-dipiano/astroAI.git
```

Step2:
```bash
cd astroAI
```

## Environment

This software uses [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for managing environments. You are free to use a different solution though compatibility is not guaranteed. Once you have installed and activated `mamba` you can proceed with creating the environment.

Step1: 
```bash 
mamba create -n astroai
```
Step2: 
```bash
mamba activate astroai
```
Step3:
```bash
mamba install -f environment.yml
```

## Installation

Once the environment is created and activate, you may install the software.

```bash
pip install .
```

Alternatively you may install in editable mode.

```bash
pip install -e .
```

## Datasets

To run this code you are required to provide DL3 simulations in a compatible FITS format:
- simulation with gammapy
- simulation with ctools
- provided sample datasets [simulations_dataset_v1](https://zenodo.org/)

## Using this code

- You can preprocess data using the instructions in the dedicated [README](./astroai/tools/README.md).
- You can train models anew following the instructions in the dedicated [README](./astroai/models/README.md) or use the provided [cnn_models_v1](https://zenodo.org/).
- You can compare the resuts with the reference real-time analysis pipeline following the instructions in the dedicated [README](./astroai/pipes/README.md).

## Configurations

- A commented configuration template for preprocessing and model training is provided here: [template_cnn.yml](./astroai/conf/template_cnn.yml).
- A commented configuration template to execute the reference rtal-time analysis pipeline is provided here: [template_gp.yml](./astroai/conf/template_gp.yml).
