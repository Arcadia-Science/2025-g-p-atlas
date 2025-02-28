# G-P Atlas

## Overview
This script implements the **G-P Atlas** method for mapping genotype to phenotype as described in [this publication](https://doi.org/10.57844/arcadia-d316-721f). It employs neural networks to encode and decode genetic and phenotypic data and applies feature attribution analysis for interpretability.

## Features
- **Genotype-Phenotype Mapping**: Uses neural networks to predict phenotypic traits from genotypic data.
- **Phenotype Autoencoder**: Learns representations of phenotypic traits.
- **Feature Attribution Analysis**: Implements Captum's Feature Ablation for importance evaluation.
- **Data Handling**: Loads and processes genotype and phenotype datasets from pickled files.
- **Training and Evaluation**: Supports training and validation of models with performance metrics like MSE, R², and Pearson correlation.

## Requirements

The script relies on the following dependencies:
- Python 3.x
- PyTorch
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Captum

Install them using:
```sh
pip install torch numpy scipy scikit-learn matplotlib captum
```

## Usage
### Command-line Arguments
The script accepts various arguments to configure model parameters. Below is a complete list of arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_alleles` | Number of segregating causal alleles | `2` |
| `--n_locs` | Number of causal loci (genes) | `900` |
| `--n_env` | Number of environmental components | `3` |
| `--n_phens` | Number of phenotypes | `30` |
| `--gen_lw` | Genetic loss weight | `1.0` |
| `--eng_lw` | Environmental loss weight | `0.1` |
| `--n_epochs` | Number of training epochs | `100` |
| `--batch_size` | Batch size | `16` |
| `--lr_r` | Learning rate | `0.001` |
| `--b1` | Adam optimizer beta1 | `0.5` |
| `--b2` | Adam optimizer beta2 | `0.999` |
| `--n_cpu` | Number of CPU cores for data loading | `14` |
| `--e_hidden_dim` | Encoder hidden layer size | `32` |
| `--d_hidden_dim` | Decoder hidden layer size | `32` |
| `--ge_hidden_dim` | Genetic encoder hidden layer size | `32` |
| `--batchnorm_momentum` | Momentum for batch normalization | `0.8` |
| `--latent_dim` | Latent space dimension | `32` |
| `--n_phens_to_analyze` | Number of phenotypes to analyze | `30` |
| `--sd_noise` | Standard deviation of noise added to phenotypes | `0.1` |
| `--gen_noise` | Noise added to genetic data | `0.3` |
| `--n_phens_to_predict` | Number of phenotypes to predict | `30` |
| `--n_epochs_gen` | Number of epochs for genetic encoder training | `100` |
| `--n_loci_measured` | Number of genetic loci with available data | `3000` |
| `--l1_lambda` | L1 regularization weight | `0.8` |
| `--l2_lambda` | L2 regularization weight | `0.01` |
| `--dataset_path` | Path to dataset files | `None` |
| `--train_suffix` | Training dataset filename | `train.pk` |
| `--test_suffix` | Test dataset filename | `test.pk` |
| `--hot_start` | Load precomputed weights | `False` |
| `--hot_start_path_e` | Path to phenotype encoder weights | `None` |
| `--hot_start_path_d` | Path to phenotype decoder weights | `None` |
| `--hot_start_path_ge` | Path to genotype encoder weights | `None` |
| `--calculate_importance` | Flag whether to calculate variable importance. Calculating variable importance takes a lot of time and isn't necessary during hyperparameter tuning. Expects "True" or "False" | `False` |

To see all options, run:
```sh
python3 g_p_atlas.py --help
```

### Input Data
G-P Atlas expects input data to be a pickled python dictionary with `phenotype` and `genotype` entries.

#### details of phenotype input
Currenly, G-P Atlas only operats on continuously traits and expects inputs to be floating point numbers. It does not require rescaling the mean and variance of these values. The object indexed by `phenotype` is expected to be a list of lists where each sublist is the of phenotypes for an individual. It expects the main list to be in this format: [[phenotypes from individual 1], [phenotypes from individual 2]...]. It expects each sublist to be [phenotype 1, phenotype 2, ...].
e.g., [[2.0,3.2,4.5...],[2.5,3.1,4.0...]...]
#### details of Genotype input
G-P Atlas expects the 'genotypes' entry to be a nested list where the main list is: [[genotypes from individual 1],[genotypes from individual 2]...]. It expects each sublist to be a list of sublists formatted as [[allelic state locus 1],[allelic state locus 2]...]. It expects allelic states to be 1 hot encoded. i.e., either [1,0] or [0,1] for a locus with 2 allelic states. If some loci in the population have more than 2 allelic states then all loci have to be 1 hot encoded with the possibility of the maximum number of allelic states. For example, if some loci have 3 allelic states then all loci have to be encoded as: [1,0,0], [0,1,0], or [0,0,1].
e.g., [[[0,1],[0,1],[1,0]...],[[0,1],[1,0],[0,1]]...]

### Running the Script
To train the model with default settings:
```sh
python3 g_p_atlas.py --dataset_path /path/to/data/
```

To load precomputed weights and resume training:
```sh
python3 g_p_atlas.py --hot_start True --hot_start_path_e /path/to/encoder.pt --hot_start_path_d /path/to/decoder.pt --hot_start_path_ge /path/to/gen_encoder.pt
```

## Output Files
- **Model Checkpoints**: Saves trained models (`phen_encoder_state.pt`, `phen_decoder_state.pt`, `gen_encoder_state.pt`)
- **G-P Atlas Run Parameters**: `run_params.txt` (Contains the exact command line entered for that run)
- **Plots**: 
  - `reconstruction_loss.svg` (Autoencoder reconstruction loss)
  - `phen_real_pred_dng_attr.svg` (Phenotype real vs. predicted values)
  - `phen_real_pred_pearsonsr_dng_attr.svg` (Pearson correlation distribution)
  - `phen_real_pred_mse_dng_attr.svg` (Mean squared error distribution)
  - `phen_real_pred_mape_dng_attr.svg` (Mean absolute percentage error distribution)
  - `phen_real_pred_r2_dng_attr.svg` (R² score distribution)
  - `g_p_attr.svg` (Distribution of feature attribution for genotype-phenotype mapping)
  - `p_p_attr.svg` (Distribution of feature attribution for phenotype-phenotype mapping)
- **Feature Attributions**:
  - `g_p_attr.pk` (Attributions to individual alleles for pehnotype prediciton) 
  - `p_p_attr.pk` (Attributions to individual phenotypes for phenotype prediction)
- **Prediction Statistics**: `test_stats.pk`
- **Paired real and predicted phenotype files**:
  These are pickled list of lists organized like this: `[[phenotypes],[predicted phenotypes']`
 - `phens_phen_encodings_dng_attr.pk` (Real and predicted phenotypes from the genotype-phenotype map)
 - `phens_phen_encodings_dng_attr_p.pk` (real and predicted phenotypes from the phenotype-phenotype map)

