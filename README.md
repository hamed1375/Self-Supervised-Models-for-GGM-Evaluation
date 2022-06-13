
# Evaluating Graph Generative Models with Contrastively Learned Features

Most of the code is from: <https://github.com/uoguelph-mlrg/GGM-metrics>

# Table of Contents  
- [Requirements and installation](#requirements-and-installation)
- [Reproducing main results](#reproducing-main-results)
  

# Requirements and installation

Python version used is 3.8.10
Most of the requirements are in requirements.txt
```
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv\
 torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
```

Requirements are for testing on CPU. For testing on GPU, please install cuda version of the libraries.

You can download the data using [download_datasets.sh](./download_datasets.sh), (run `sh download_datasets.sh` in terminal). 
For mixing with GRAN generated graphs you should also download GRAN graphs through [download_gran_graphs.sh](./download_gran_graphs.sh), (run `sh download_gran_graphs.sh` in terminal).

# Reproducing main results
The arguments used for training is provided in [config.py](./config.py). 

A general expression of how you can run an experiment is as:
```
python main.py --permutation_type={permutation type} {feature_extractor}\
--dataset={dataset} {other args if you want to change default}
```
Permutation type can be one of the {mixing-random, rewiring-edges, mode-collapse, mode-dropping}.
Feature extractor can be graphcl, infograph, or random-gin. If the network is not already pretrained, code will train the model, save it in the saved_models, and run the experiments with the trained model.

For running with the hyperparameters used in the paper you can make --use_predefined_hyperparams equal to true. Making this argument true will ignore the num_layers, epochs, hidden_dim set in the arguments and replace them with the default one. For checking other changeable arguments check [config.py](./config.py)

A simple experiment running example is like this:
```
python main.py --permutation_type=rewiring-edges --feature_extractor=graphcl \
--dataset=grid --num_layers=2 --hidden_dim=16 
```

Rank correlations are automatically computed and printed at the end of each experiment, and results are stored in experiment_results/. 
You can also use [create_bash_script.py](./create_bash_script.py) for creating a set of experiments. After running it, bash commands will be saved in [all_commands.sh](./all_commands.sh). You can run it through the terminal with command `sh all_commands.sh`