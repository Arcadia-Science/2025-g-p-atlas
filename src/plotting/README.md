
# G-P Atlas Analysis Scripts

## Purpose

This repository contains a set of scripts for analyzing genetic and phenotypic data using G-P Atlas. These scripts generate various figures and analyses based on both real yeast cross-data and simulated datasets.
When run on G-P Atlas anaslysis of simulated data from 10.57844/arcadia-5953-995f and yeast data from 10.1038/nature11867, these scripts create the substantive figures presented in  10.57844/arcadia-d316-721f.

## Files in This Repository

### Main Analysis Scripts

- **`plot_gp_yeast_run.py`**: Runs an initial analysis on the yeast cross dataset and generates multiple visualizations based on phenotypic and genotypic data.
- **`plot_gp_simulated_run.py`**: Similar to the yeast run script but applied to simulated data for comparative analysis.
- **`plot_gp_yeast_linkage.py`**: Analyzes variable importance measures derived from G-P Atlas for yeast data, highlighting significant genomic regions.
- **`plot_gp_simulated_linkage.py`**: Performs an equivalent analysis on simulated datasets, focusing on the importance of genetic features.

### Helper Scripts

- **`helper_functions.py`**: Contains utility functions used across multiple scripts:
  - `mean_absolute_percentage_error(y_true, y_pred)`: Computes the mean absolute percentage error.
  - `MSE(y_true, y_pred)`: Calculates the mean squared error of a set of predictions.
  - `calc_coef_of_det(y_true, y_pred)`: Computes the coefficient of determination.
  - `calculate_fpr_threshold(fpr, thresholds)`: Uses interpolation to find the threshold required to achieve a 1% false positive rate.

- **`matplotlibrc`**: A configuration file for setting matplotlib's global plotting style. To take advantage of this formatting (which was used in the pub) run the plotting functions in this directory.

## Requirements

See README.md in the repository root folder for environment and requirements installation instructions.

Ensure you have the following dependencies installed:

- `Python 3.x`
- `matplotlib`
- `numpy`
- `seaborn`
- `scipy`
- `scikit-learn`

## Usage

These scripts are intended to be run on a folder containing the output of G-P Atlas (see src folder in the repository).
They expect that you have a dataset that was formatted appropirately (see README in g-p_atlas_analysis_scripts) and
was then analyzed using `g_p_atlas.py`

### Example Usage
```bash
python3 plot_gp_yeast_run.py [PATH TO G-P ATLAS OUTPUT]
python3 plot_gp_simulated_run.py [PATH TO G-P ATLAS OUTPUT]
python3 plot_gp_yeast_linkage.py [PATH TO G-P ATLAS OUTPUT]
```

## Outputs

The scripts generate various output files, including:

- **Scatter plots** of latent representations (e.g., `latent_representation_phenotype.png`)
- **Error distributions** for phenotype/genotype comparisons (e.g., `mse_kde_p_p.png`)
- **ROC Curves** for evaluating classification performance (e.g., `ROC_gene_classification.png`)
- **Linkage Analysis** plots highlighting genomic regions of interest

## Configuration

Some scripts require specific dataset structures. Ensure the input data files match the expected format outlined in the script comments.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

