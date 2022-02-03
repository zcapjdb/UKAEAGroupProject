# UKAEA Group Project
Repository for the UCL CDT in DIS group project with UKAEA.

<!-- omit in toc -->
- [UKAEA Group Project](#ukaeagroupproject) 
- [Data Preparation:](#data-preparation)
- [Missing Outputs:](#missing-outputs)
- [QLKNN Reproduction:](#qlknn-reproduction)
- [Autoencoder:](#autoencoder)

## Installation
TODO: add in requirements.txt
To install the package run: `python setup.py install`

## Data Preparation:
The models used in the project take the .h5 file provided and convert it to pickled dataframes. The data is also split into a training, validation and test set at this point. This is done in [QLKNNDataPreparation.ipynb](src/notebooks/QLKNNDataPreparation.ipynb).

Initial data exploration can also be found in [DataExploration.ipynb](src/notebooks/DataExploration.ipynb).

## Missing Outputs:
A major difficulty in applying a neural network surrogate model to the data is that QuaLiKiz, which is used to train the surrogate, does not always map an input to an output.

It is therefore of interest to try and understand why this is the case. To do so a classifier is trained on the model inputs to determine if a given set of inputs gives a corresponding output. TODO: add link to Thandi's notebook

## QLKNN Reproduction:
The QLKNN model is defined in [QLKNN.py](src/scripts/QLKNN.py).
Trained using [train.py](src/scripts/train.py)
Results evaluated in [RegressionOutputs.ipynb](src/notebooks/RegressionOutputs.ipynb).

## Autoencoder:
The autoencoder is defined in [Autoencoder.py](src/scripts/Autoencoder.py).
Trained using [train_ae.py](src/scripts/train_ae.py)
Results evaluated in [AutoencoderOutputs.ipynb](src/notebooks/AutoencoderOutputs.ipynb). TODO: This is not yet implemented.
