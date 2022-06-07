# Neural Networks with Sequentially Semiseparable Weight Matrices

This repository contains the code, which was used to run the experiments described in 

Kissel, Gottwald, Gjeroska, Paukner, Diepold: Backpropagation Through States: Training Neural Networks with Sequentially Semiseparable Weight Matrices

## File Structure

- ``benchmark.py``: Script used for running benchmarks. This script imports the other scripts defining the models, the data and the training procedure. Therefore, this script can be seen as a starting point to understand how the other scripts and models can be used. 
- ``datasets.py``: Script for loading the datasets (also providing the data splits used in the paper).
- ``model_training.py``: Functions used for model training (based on gradient descent for the torch-model parameters). It also provides functions for computing the number of model parameters as well as the accuracy score for given predictions.
- ``standardmodels.py``: Defining the standard neural networks used as benchmark models in the paper. Moreover, the base feature extractor ConvFeatureExtractorBase is defined in this python script, which is also used as feature extractor by the models based on structured weight matrices. 
- ``semiseparablemodels.py``: Model definitions for the neural networks using sequentially semiseparable weight matrices (for fully connected as well as convolutional neural networks).
- ``rank1models.py``: Model definitions for the neural networks using rank-1 weight matrices (for fully connected as well as convolutional neural networks). 

## Usage

We recomment using anaconda or miniconda to avoid troubles with required python packages. We used conda 4.10.1, but any higher version should be fine. Conda can be obtained from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). 

Create your conda environment using

``conda create --name statespace python=3.7``

Afterwards, you can activate the environment using

``conda activate statespace``

and install the required packages with

``python3 -m pip install -r requirements.txt``

Then, you can start benchmarks using

``python3 benchmark.py``

You can set the hyperparameters for the benchmark directly in the benchmark.py file. The benchmark run will create a "benchmark_result.p" pickle file in the folder of the script. This is a pickled python dict, which contains the results of the benchmark (you can see the structure in the benchmark.py file). 
