# 2025-g-p-atlas

[![run with conda](https://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)

## Purpose

This repository contains scripts for running and analyzing the output of **G-P Atlas**, a method for creating holistic genotype-to-phenotype mappings that capture all phenotypic and genetic data in a single model. These models can be used for phenotype prediction from genetic or phenotypic data, linkage analysis and so on. The method is explained and demonstrated in [this publication](https://doi.org/10.57844/arcadia-d316-721f). All data, results, and code used for that publication are contained in this repository.

Key functionalities include:

- Fitting the **G-P Atlas** model that creates a genotype-to-phenotype map for many phenotypes and genetic locations.
- Computing model performance metrics such as **mean squared error (MSE), mean absolute percentage error (MAPE), and coefficient of determination (R²)**.
- Visualizing and comparing the results of different **G-P Atlas runs**.
- Generating **ROC curves** for classifying loci contributing to phenotypes.
- Calculating the importance (variable importance) of loci, alleles, and phenotypes in predicting other phenotypes.
- Predicting phenotypes from genotypes.
- Predicting phenotypes from other phenotypes.

## Installation and Setup

To directly replicate the environment used to produce the pub, first install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and [Mamba](https://mamba.readthedocs.io/en/latest/).

Then create and activate a new virtual environment. From the main repository folder:

```bash
mamba env create -n gp_atlas_env --file envs/exact_environment.yml
mamba activate gp_atlas_env
```

If you do not need to exactly replicate the environment used in the pub and would prefer to use `pip` with an environment manager of your choice, you can install dependencies in this way:

```bash
pip install -r envs/requirements.txt
```

Note: be sure to use an environment manager alongside `pip` to install the required dependencies once you have a virtual environment.

## Data

All data anlyzed in https://doi.org/10.57844/arcadia-d316-721f are contained in the `data` folder of this repository. See the folder structure below. There are two datasets: one simulated dataset from https://doi.org/10.57844/arcadia-5953-995f and one empirical, F1 yeast hybrid, population from https://doi.org/10.1038/nature11867.

### Description of the folder structure

```
2025-g-p-atlas
├── data
│   ├── g_p_atlas_format_simulated_data # test/train files for simulated data
│   └── yeast_data			# folders with original and reformatted yeast data
│       ├── g_p_atlas_format_data	# reformatted yeast data
│       └── orig_data			# original format yeast data, scripts to reformat that data and a README about script usage
├── results				# output of G-P Atlas and plotting scripts
│   ├── full_g_p_run_simulated_data	# output for runs on simulated data
│   └── yeast_data			# output for runs on yeast data
│       ├── full_g_p_atlas_all_yeast_data	# output for G-P Atlas runs on the full yeast dataset and plots
│       ├── no_hidden_layer_genotype_all_yeast_data	# output for G-P Atlas runs (with no genotype hidden layer) on the full yeast dataset and plots
│       ├── no_hidden_layer_phenotype_all_yeast_data	# output for G-P Atlas runs (with no phenotype hidden layer) on the full yeast dataset and plots
│       └── 41586_2013_BFnature11867_MOESM88_ESM_yeat_linkage_data.csv	# Linage data from 10.1038/nature11867. It is used in several plots.
├──envs 			# environement files
│   ├── dev.yml                 # conda .yml for managing repository updates  
│   ├── exact_environment.yml   # conda .yml file for creating an environment to replicate the pub
│   └── requirements.txt        # requirements.txt file to install dependencies with PIP
├── LICENSE                     
├── Makefile
├── pyproject.toml
├── README.md			# this file
└── src				# all of the source code for G-P Atlas
    ├── g-p_atlas_analysis_scripts	# the main G-P Atlas script
    │   ├── g_p_atlas_1_layer_g_p.py	# G-P Atlas with no hidden layer in genotype encoder
    │   ├── g_p_atlas_1_layer_p_p.py	# G-P Atlas with no hidden layer in phenotype encoder
    │   ├── g_p_atlas.py		# the full G-P Atlas script
    │   └── README.me         # readme covering the use, data formatting, and output of g_p_atlas.py
    └── plotting			# scripts for plotting the output of G-P Atlas
        ├── helper_functions.py		# helper functions used in more than one plotting script
        ├── matplotlibrc		# default matplotlib format file used to create plots for the pub
        ├── plot_compare_coef_det_two_runs.py	# compares the R-squared for multiple G-P Atlas runs
        ├── plot_gp_simulated_linkage.py	# linkage analysis of simulated data
        ├── plot_gp_simulated_run.py		# performance visualization for runs on simulated data
        ├── plot_gp_yeast_linkage.py		# linkage analysis of yeast data
        ├── plot_gp_yeast_run.py		# performance visualization for runs on yeast data
        └── README.md				# readme discussing all of the plotting functions
```

### Usage

For guidance on data formatting for, and usage of, the G-P Atlas models, see [src/g-p_atlas_analysis_scripts/README.md](src/g-p_atlas_analysis_scripts/README.md).

To recapitulate the primary analyses from [the pub](https://doi.org/10.57844/arcadia-d316-721f), follow these steps:

1. Clone this repository and set up an environment with the appropriate dependencies (see above).

2. De-compress all data:	
   ```
   gunzip -r ./
   ```

3. Run G-P Atlas on simulated data:	
   ```
   python3 src/g-p_atlas_analysis_scripts/g_p_atlas.py \
   --dataset_path data/g_p_atlas_format_simulated_data/ \
   --n_epochs 1000 \
   --n_epochs_gen 1000 \
   --sd_noise 0.8 \
   --gen_noise 0.8 \
   --n_alleles 3
   ```

4. Run G-P Atlas on yeast data:	
   ```sh
   python3 src/g-p_atlas_analysis_scripts/g_p_atlas.py \
      --dataset_path data/yeast_data/g_p_atlas_format_data/ \
      --n_epochs 1000 \
      --n_epochs_gen 1000 \
      --n_loci_measured 11623 \
      --latent_dim 32 \
      --e_hidden_dim 64 \
      --d_hidden_dim 64 \
      --ge_hidden_dim 2048 \
      --n_phens_to_analyze 46 \
      --n_phens_to_predict 46 \
      --sd_noise 0.8 \
      --gen_noise 0.8
   ```
5. Create run plots for simulated data: 
   ```sh
   python3 src/plotting/plot_gp_simulated_run.py \
      results/full_g_p_run_simulated_data/
   ```

6. Create linkage plots for simulated data: 
   ```sh
   python3 src/plotting/plot_gp_simulated_linkage.py \
      results/full_g_p_run_simulated_data/ \
      data/g_p_atlas_format_simulated_data/test.pk
   ```

7. Create run plots for yeast data: 
   ```sh
   python3 src/plotting/plot_gp_yeast_run.py \
      results/yeast_data/full_g_p_atlas_all_yeast_data/
   ```

8. Create linkage plots for yeast data: 
   ```sh
   python3 src/plotting/plot_gp_yeast_linkage.py \
      results/yeast_data/full_g_p_atlas_all_yeast_data/ \
      data/yeast_data/g_p_atlas_format_data/test.pk
   ```


### Compute Specifications

Exact specifications used for the pub are as follows:

- OS: Linux (Ubuntu 22.04.5 LTS)
- Processor: AMD Ryzen 9 5950x 16-core processor
- RAM: 128 GB
- GPU: NVIDIA GeForce RTX 3070

Note: It is likely the code in this repository will run efficiently with less memory and fewer cores, but a GPU is advised.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
