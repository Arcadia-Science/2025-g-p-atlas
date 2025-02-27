# 2025-g-p-atlas

[![run with conda](https://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)

## Purpose

This repository contains scripts for running and analyzing the output of **G-P Atlas**, a method for creating holistic genotype-to-phenotype mappings that capture all phenotypic and genetic data in a single model. These models can be used for phenotype prediction from genetic or phenotypic data, linkage analysis and so on. The method is explained and demonstrated in 10.57844/arcadia-d316-721f

Key functionalities include:

- Running the **G-P Atlas** model to infer genetic contributions to phenotypes
- Computing model performance metrics such as **mean squared error (MSE), mean absolute percentage error (MAPE), and coefficient of determination (R²)**
- Visualizing and comparing the results of different **G-P Atlas runs**
- Generating **ROC curves** and evaluating variable importance

## Installation and Setup

To directly replicate the environment used to produce the pub use conda in the following way:
Note: this environemnt is linux or mac specific.

1. If you haven't already installed some form of conda, install miniconda:
   [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and [Mamba](https://mamba.readthedocs.io/en/latest/).

2. Create and activate the environment.
   From the main repository folder:
   ```bash
   mamba env create -n gp_atlas_env --file envs/exact_environment.yml
   conda activate gp_atlas_env
   ```
To use PIP to install the necessary files for application to other datasets:
Note: be sure to use an envionrment manager. This is only to install the necessary dependencies once you have a virtual environement.

1. From the main repository folder:
   ```bash
   pip3 install -c envs/requirements.txt
   ```

## Data

All data anlyzed in 10.57844/arcadia-d316-721f are contained in this repository. They are contained in `./data`. See the folder structure below. There are two dataset, one simulated dataset from 10.57844/arcadia-5953-995f and one emperical, F1 yeast hybrid population from 10.1038/nature11867.

## Overview

TODO: Add more detailed overview

### Description of the folder structure

```
2025-g-p-atlas
├── data
│   ├── g_p_atlas_format_simulated_data #contains test/train files for simulated data
│   └── yeast_data			#contains folders with original and reformatted yeast data
│       ├── g_p_atlas_format_data	#contains reformatted yeast data
│       └── orig_data			#contains original format yeast data and scripts to reformat that data
├── results				#contains the output of G-P Atlas and plotting scripts
│   ├── full_g_p_run_simulated_data	#contains output for runs on simulated data
│   └── yeast_data			#contains folders with output for rund on yeast data
│       ├── full_g_p_atlas_all_yeast_data	#output for G-P Atlas runs on the full yeast dataset and plots
│       ├── no_hidden_layer_genotype_all_yeast_data	#output for G-P Atlas(no genotype hidden layer) runs on the full yeast dataset and plots
│       ├── no_hidden_layer_phenotype_all_yeast_data	#output for G-P Atlas(no phenotype hidden layer) runs on the full yeast dataset and plots
│       └── 41586_2013_BFnature11867_MOESM88_ESM_yeat_linkage_data.csv	#Linage data from 10.1038/nature11867. It is used in several plots.
├──envs 			# contains environement files
│   ├── dev.yml                 # conda .yml for managing repository updates  
│   ├── exact_environment.yml   # conda .yml file for creating an environment to replicate the pub
│   └── requirements.txt        # requirements.txt file to install dependencies with PIP
├── LICENSE                     
├── Makefile
├── pyproject.toml
├── README.md			# this file
└── src				# contains all of the source code for G-P Atlas
    ├── g-p_atlas_analysis_scripts	# contains the main G-P Atlas script
    │   ├── g_p_atlas_1_layer_g_p.py	# G-P Atlas with no hidden layer in genotype encoder
    │   ├── g_p_atlas_1_layer_p_p.py	# G-P Atlas with no hidden layer in phenotype encoder
    │   └── g_p_atlas.py		# the full G-P Atlas script
    └── plotting			# contains scripts for plotting the output of G-P Atlas
        ├── helper_functions.py		# helper functions used in more than one plotting script
        ├── matplotlibrc		# default matplotlib format file used to create plots for the pub
        ├── plot_compare_coef_det_two_runs.py	# compares the R-squared for multiple G-P Atlas runs
        ├── plot_gp_simulated_linkage.py	# linkage analysis of simulated data
        ├── plot_gp_simulated_run.py		# performance visualization for runs on simulated data
        ├── plot_gp_yeast_linkage.py		# linkage analysis of yeast data
        ├── plot_gp_yeast_run.py		# performance visualization for runs on yeast data
        └── README.md				# readme discussing all of the plotting functions
```

### Methods

If you have set up your environment and downloaded this repository, to recapitulate the primary analyses from the pub 10.57844/arcadia-d316-721f:

1. De-compress all data:	`./gunzip -r ./data`
2. Run G-P Atlas on simulated data:	`./src/g-p_atlas_analysis_scripts/g_p_atlas.py --dataset_path ./data/g_p_atlas_format_simulated_data/ --n_epochs 1000 --n_epochs_gen 1000 --test_suffix test.pk --train_suffix train.pk --sd_noise 0.8 --gen_noise 0.8 --n_alleles 3`
3. Run G-P Atlas on yeast data:	`./src/g-p_atlas_analysis_scripts/g_p_atlas.py --dataset_path ./data/yeast_data/g_p_atlas_format_data/ --n_epochs 1000 --n_epochs_gen 1000 --n_loci_measured 11623 --latent_dim 32 --e_hidden_dim 64 --d_hidden_dim 64 --ge_hidden_dim 2048 --n_phens_to_analyze 46 --n_phens_to_predict 46 --sd_noise 0.8 --gen_noise 0.8`
4. Create run plots for simulated data: `./plotting/plot_gp_simulated_run.py ./results/full_g_p_run_simulated_data/`
5. Create linkage plots for simulated data: `./plotting/plot_gp_simulated_linage.py ./results/full_g_p_run_simulated_data/`
6. Create run plots for yeast data: `./plotting/plot_gp_simulated_run.py ./results/yeast_data/full_g_p_atlas_all_yeast_data/`
7. Create linkage plots for yeast data: `./plotting/plot_gp_simulated_linage.py ./results/yeast_data/full_g_p_atlas_all_yeast_data/`


### Compute Specifications

Exact specifications used for the pub:

- OS: Linux (Ubuntu 22.04.5 LTS)
- Processor: AMD Ryzen 9 5950x 16-core processor
- RAM: 128 GB
- GPU: NVIDIA GeForce RTX 3070

Note: It is likely this will run efficiently with much less ram and fewer cores 

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).

