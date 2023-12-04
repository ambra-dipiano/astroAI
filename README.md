# astroAI

## Setup

You should clone the repository and move into its root directory to work.

Step1:
```bash
git clone git@github.com:ambra-dipiano/astroAI.git
```

Step2:
```bash
cd astroAI
```

## Environment

This software uses `venv` virtual environment. You are free to use a different solution but requirements compatibility is not guaranteed. 

Step1: 
```bash 
python -m venv astroai
```
Step2: 
```bash
source astroai/bin/activate
```
Step3:
```bash
python -m pip install -r requirements.txt
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

## Configuration file
How to configure the software.

## Preprocessing pipeline
A single pipeline to manage preprocessing for all different tools.

## Models
An ensamble of deep learning models, with different architctures and goals.

### CNN-based models
A collection of CNN-based models providing a number of different tools for high level analysis.

#### Binary classifier
This model aims to classifying counts maps containing only background signal (class 0) and counts maps containing source and background signal.

#### Image cleaner
This model aims to remove background noise from a counts map.

#### Hotspots regressor
This model aims to localise the coordinates of hotspots in a counts maps, whereas the hotspots have a given minimum Gaussian significance.

#### Background classifier
This model aims to classify various levels of background in a counts map, in order to associate the most appropriate Instrument Response Function.