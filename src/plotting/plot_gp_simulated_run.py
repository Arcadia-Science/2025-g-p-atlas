import pickle as pk
import sys

import helper_functions as hf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

"""This script runs an initial analysis on the output a G-P Atlas of simulated data
 data from 10.57844/arcadia-5953-995f and presented in 10.57844/arcadia-d316-721f. It
 creates a series of figures presented in that pub. It is intended to be used as follows:
 python3 plot_gp_yeast_run.py [PATH TO G-P ATLAS OUTPUT]"""

target_folder = sys.argv[1]  # folder containing output of G-P Atlas when run on simulated data

# load real and predicted phenotypes. Predictions are based on phenotypes
phen_encodings = pk.load(open(target_folder + "phens_phen_encodings_dng_attr_p.pk", "rb"))

# load real and predicted phenotypes. predictions are based on genotypes
gen_encodings = pk.load(open(target_folder + "phens_phen_encodings_dng_attr.pk", "rb"))

# tsne plot of phenotype latent space
tsne_model = TSNE(n_components=2,  init="random", perplexity=30)
latent_representation_phenotype = np.array(phen_encodings[2]).T
embeddings = tsne_model.fit_transform(latent_representation_phenotype)
plt.scatter(embeddings.T[0], embeddings.T[1])
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.ylabel("Dimension 2")
plt.xlabel("Dimension 1")
plt.ylim(-25, 25)
plt.xlim(-25, 25)
plt.savefig(target_folder + "latent_representation_phenotype.svg")
plt.savefig(target_folder + "latent_representation_phenotype.png")
plt.close()

# plot phen-phen relationships
phenotypes = np.array(phen_encodings[0]).T
phenotype_encodings = np.array(phen_encodings[1]).T
plt.plot(phenotypes, phenotype_encodings, "o")
plt.plot([0, 90], [0, 90], ":k", linewidth=0.75)
plt.xticks(list(range(0, 100, 10)), fontfamily="monospace")
plt.yticks(list(range(0, 100, 10)), fontfamily="monospace")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.ylim(0, 90)
plt.xlim(0, 90)
plt.xlabel("Real phenotype")
plt.ylabel("Predicted phenotype")
plt.savefig(target_folder + "p_p_plot_raw.svg")
plt.savefig(target_folder + "p_p_plot_raw.png")
plt.close()

# plot MAPE density
errs = [
    hf.mean_absolute_percentage_error(phenotypes[n], phenotype_encodings[n])
    for n in range(len(phenotypes))
]
sns.kdeplot(errs, label="Within phenotype error")
print("Median MAPE between real and predicted phenotypes based on phenotypic data")
print("Median MAPE within phenotypes")
median = np.median(errs)
plt.axvline(median)
print(median)
errs2 = [
    hf.mean_absolute_percentage_error(phenotypes.T[n], phenotype_encodings.T[n])
    for n in range(len(phenotypes.T))
]
print("Median MAPE within individuals")
median = np.median(errs2)
plt.axvline(median)
print(median)
sns.kdeplot(errs2, label="Within individual error")
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.xlabel("Mean absolute percentage error")
plt.legend()
plt.xlim(0, 15)
plt.savefig(target_folder + "mape_kde_p_p.svg")
plt.savefig(target_folder + "mape_kde_p_p.png")
plt.close()

# plot gen-phen relationships
phenotypes = np.array(gen_encodings[0]).T
phenotype_encodings = np.array(gen_encodings[1]).T
plt.plot(phenotypes, phenotype_encodings, "o")
plt.plot([0, 90], [0, 90], ":k", linewidth=0.75)
plt.xticks(list(range(0, 100, 10)), fontfamily="monospace")
plt.yticks(list(range(0, 100, 10)), fontfamily="monospace")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.ylim(0, 90)
plt.xlim(0, 90)
plt.xlabel("Real phenotype")
plt.ylabel("Predicted phenotype")
plt.savefig(target_folder + "g_p_plot_raw.svg")
plt.savefig(target_folder + "g_p_plot_raw.png")
plt.close()

# plot MAPE density
errs = [
    hf.mean_absolute_percentage_error(phenotypes[n], phenotype_encodings[n])
    for n in range(len(phenotypes))
]
sns.kdeplot(errs, label="Within phenotype error")
print("Median MAPE between real and predicted phenotypes based on genetic data")
print("Median MAPE within phenotypes")
median = np.median(errs)
plt.axvline(median)
print(median)
errs2 = [
    hf.mean_absolute_percentage_error(phenotypes.T[n], phenotype_encodings.T[n])
    for n in range(len(phenotypes.T))
]
sns.kdeplot(errs2, label="Within individual error")
median = np.median(errs2)
plt.axvline(median)
print("Median MAPE within individuals")
print(median)
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.xlabel("Mean absolute percentage error")
plt.legend()
plt.xlim(0, 15)
plt.savefig(target_folder + "mape_kde_g_p.svg")
plt.savefig(target_folder + "mape_kde_g_p.png")
plt.close()
