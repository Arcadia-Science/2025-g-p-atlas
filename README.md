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
   pip3 install -c env/requirements.txt
   ```

## Data

TODO: Add details about the description of input / output data and links to Zenodo depositions, if applicable.

## Overview

TODO: Add more detailed overview

### Description of the folder structure
```bash
.
├──src # Contains all of the scripts and other code to produce the pub
+--plotting # Contains all of the scripts used to create the plots in the pub
   -plot_gp_simulated_linkage.py # Linkage analysis of simulated data
   -plot_gp_simulated_run.py     # Performance visualization for runs on simulated
   -plot_gp_yeast_linkage.py     # Linkage analysis of yeast data
   -plot_gp_yeast_run.py         # Performance visualization for runs on yeast data
   -plot_compare_coef_det_two_runs.py # Compares the R-squared for multiple G-P Atlas runs
   -helper_functions.py          # Contains helper functions used in more than one script
   -README.md                    # Readme discussing all of the plotting functions
   -matplotlibrc                 # Default matplotlib format file used to create plots for the pub
  -g-p_atlas_analysis_scripts # Contains the main G-P Atlas scripts
   -g_p_atlas.py                 # Main script for running G-P Atlas
   -g_p_atlas_1_layer_g_p.py     # Version of G-P Atlas with no hidden layer in the genetic encoder
   -g_p_atlas_1_layer_p_p.py     # Version of G-P Atlas with no hidden layer in the phenotype encoder
 -envs # Contains the conda .yml file and the requirements.txt for installing required software
  -exact_environment.yml         # conda .yml file for creating an environment to replicate the pub
  -requirements.txt              # Requirements.txt file to facilitate installing dependencies with PIP
 -README.md # This file
 -Makefile # Makefile to controll the repository updates
 -pyproject.toml # .toml file to define the python module in this repository
 -LICENSE   # File containing the software liscence
```

### Methods

TODO: Include a brief, step-wise overview of analyses performed.

> Example:
>
> 1.  Download scripts using `download.ipynb`.
> 2.  Preprocess using `./preprocessing.sh -a data/`
> 3.  Run Snakemake pipeline `snakemake --snakefile Snakefile`
> 4.  Generate figures using `pub/make_figures.ipynb`.

### Compute Specifications

Exact specifications used for the pub:

- OS: Linux (Ubuntu 22.04.5 LTS)
- Processor: AMD Ryzen 9 5950x 16-core processor
- RAM: 128 GB
- GPU: NVIDIA GeForce RTX 3070

Note: It is very likely this will run efficiently with much less ram and fewer cores 

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).

