import os as os
import pickle as pk
import sys as sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import seaborn as sns
from sklearn.manifold import TSNE

'''This script runs an initial analysis on the output G-P Atlas of the yeast cross
 data from DOI_FOR_YEAST_PAPER and presented in DOI_FOR_PUB. It creates a series of figures
 presented in that pub. It is intended to be used as follows:
 python3 plot_gp_yeast_run.py [PATH TO G-P ATLAS OUTPUT]'''

target_folder = sys.argv[1] #folder containing output of G-P Atlas when run on the Yeast cross data

#helper functions
def mean_absolute_percentage_error(y_true, y_pred):
    #calculates the mean absolute percentage error of a set of predictions
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MSE(y_true,y_pred):
    #calculates the mean squared error of a set of predictions
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true-y_pred)**2)

def calc_coef_of_det(y_true,y_pred):
    #calculates the coefficient of determination of a model
        y_true = np.array(y_true)
        np.array(y_pred)
        ssres = np.sum((y_true-y_pred)**2)
        sstot = np.sum((y_true-np.mean(y_true))**2)
        return 1-(ssres/sstot)

#load real and predicted phenotypes. Predictions are based on phenotypes
phen_encodings = pk.load(open(target_folder+'phens_phen_encodings_dng_attr_p.pk'  ,'rb'))

#load real and predicted phenotypes. predictions are based on genotypes
gen_encodings = pk.load(open(target_folder+'phens_phen_encodings_dng_attr.pk'  ,'rb'))

#define a minimum value to avoid numerical underflow and deal with divide by zero issues
EPS = 1e-15

#tsne plot of phenotype latent space
tsne_model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50)
latent_representation_phenotype = np.array(phen_encodings[2]).T
embeddings = tsne_model.fit_transform(latent_representation_phenotype)
plt.scatter(embeddings.T[0],embeddings.T[1])
plt.xticks(fontfamily='monospace')
plt.yticks(fontfamily='monospace')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.ylabel('Dimension 2')
plt.xlabel('Dimension 1')
plt.ylim(-25,25)
plt.xlim(-25,25)
plt.savefig(target_folder+'latent_representation_phenotype.svg')
plt.savefig(target_folder+'latent_representation_phenotype.png')
plt.close()

#plot phen-phen relationships
phenotypes = np.array(phen_encodings[0]).T
phenotype_encodings = np.array(phen_encodings[1]).T
plt.plot(phenotypes,phenotype_encodings,'o')
plt.plot([-25,35],[-25,35],':k', linewidth = 0.75)
plt.xticks(list(range(-20,40,10)),fontfamily='monospace')
plt.yticks(list(range(-20,40,10)),fontfamily='monospace')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.ylim(-25,35)
plt.xlim(-25,35)
plt.xlabel('Real phenotype')
plt.ylabel('Predicted phenotype')
plt.savefig(target_folder+'p_p_plot_raw.svg')
plt.savefig(target_folder+'p_p_plot_raw.png')
plt.close()

#plot MSE density
errs = [MSE(phenotypes[n], phenotype_encodings[n])for n in range(len(phenotypes))]
sns.kdeplot(errs,label="within phenotype error")
median = np.median(errs)
plt.axvline(median)
print('Median MSE between real and predicted phenotypes based on phenotypiic data')
print('Median MSE within phenotypes')
print(median)
plt.axvline(median)
errs2 = [MSE(phenotypes.T[n], phenotype_encodings.T[n])for n in range(len(phenotypes.T))]
sns.kdeplot(errs2,label="Within individual error")
median = np.median(errs2)
plt.axvline(median)
print('Median MSE within individuals')
print(median)
plt.axvline(median)
plt.xticks(fontfamily='monospace')
plt.yticks(fontfamily='monospace')
plt.xlabel('Mean squared error')
plt.xlim(0,7)
plt.legend()
plt.savefig(target_folder+'mse_kde_p_p.svg')
plt.savefig(target_folder+'mse_kde_p_p.png')
plt.close()

#plot gen-phen relationships
phenotypes = np.array(gen_encodings[0]).T
phenotype_encodings = np.array(gen_encodings[1]).T
plt.plot(phenotypes,phenotype_encodings,'o')
plt.plot([-25,35],[-25,35],':k', linewidth = 0.75)
plt.xticks(list(range(-20,40,10)),fontfamily='monospace')
plt.yticks(list(range(-20,40,10)),fontfamily='monospace')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.ylim(-25,35)
plt.xlim(-25,35)
plt.xlabel('Real phenotype')
plt.ylabel('Predicted phenotype')
plt.savefig(target_folder+'g_p_plot_raw.svg')
plt.savefig(target_folder+'g_p_plot_raw.png')
plt.close()

#plot MSE density
errs = [MSE(phenotypes[n], phenotype_encodings[n])for n in range(len(phenotypes))]
sns.kdeplot(errs,label="Within phenotype error")
median = np.median(errs)
print('Median MSE between real and predicted phenotypes based on genetic data')
print('Median MSE within phenotypes')
print(median)
plt.axvline(median)
plt.xlim(0,7)
errs2 = [MSE(phenotypes.T[n], phenotype_encodings.T[n])for n in range(len(phenotypes.T))]
sns.kdeplot(errs2,label="Within individual error")
median = np.median(errs2)
print('Median MSE within individual')
print(median)
plt.axvline(median)
plt.xticks(fontfamily='monospace')
plt.yticks(fontfamily='monospace')
plt.xlabel('Mean squared error')
plt.legend()
plt.savefig(target_folder+'mse_kde_g_p.svg')
plt.savefig(target_folder+'mse_kde_g_p.png')
plt.close()


#plot heritability (narrow sense) vs the coefficient of determination for g-p atlas
herit = [0.85,0.33,0.38,0.40,0.59,0.65,0.44,0.66,0.60,0.59,0.56,0.41,0.32,0.56,0.35,0.48,\
0.44,0.63,0.64,0.80,0.33,0.43,0.74,0.27,0.50,0.72,0.64,0.48,0.54,0.49,0.57,0.74,0.53,0.78,\
0.58,0.71,0.43,0.55,0.73,0.29,0.50,0.73,0.53,0.69,0.54,0.55]
coef_det = [calc_coef_of_det(gen_encodings[0][n],gen_encodings[1][n]) for n in \
range(len(gen_encodings[0]))]
plt.plot(herit,coef_det,'o')
print('gp vs. h')
print(sc.stats.wilcoxon(herit,coef_det))
plt.plot([0,1],[0,1],':k', linewidth = 0.75)
plt.xticks(fontfamily='monospace')
plt.yticks(fontfamily='monospace')
plt.xlabel('Narrow sense heritability (h)')
plt.ylabel('Variance explained by G-P atlas')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig(target_folder+'coef_det_vs_h2_g_p.svg')
plt.savefig(target_folder+'coef_det_vs_h2_g_p.png')
plt.close()

#plot the additive variance explained by genomic prediction vs the
#coefficient of determination for g-p atlas
add_var = [0.79,0.25,0.32,0.40,0.52,0.53,0.40,0.54,0.55,0.45,0.53,0.31,0.28,0.47,0.24,0.35,0.40,\
0.56,0.62,0.74,0.40,0.47,0.65,0.31,0.47,0.64,0.54,0.39,0.50,0.47,0.54,0.61,0.48,0.61,0.43,0.60,0.38,\
0.52,0.60,0.28,0.43,0.57,0.46,0.71,0.52,0.55]
print('gp vs. additive variation explained by genomic prediction')
plt.plot(add_var,coef_det,'o')
print(sc.stats.wilcoxon(add_var,coef_det))
plt.plot([0,1],[0,1],':k', linewidth = 0.75)
plt.xticks(fontfamily='monospace')
plt.yticks(fontfamily='monospace')
plt.xlabel('Additive variance explained by genoimc prediction')
plt.ylabel('Variance explained by G-P atlas')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig(target_folder+'coef_det_vs_add_var_g_p.svg')
plt.savefig(target_folder+'coef_det_vs_add_var_g_p.png')
plt.close()

#plot the broad sense heritability vs coefficient of determinaytion for g-p atlas
herit_b = [0.96,0.44,0.47,0.49,0.85,0.71,0.78,0.72,0.70,0.77,0.75,0.76,0.63,0.58,0.60,0.74,0.81,\
0.84,0.89,0.87,0.57,0.87,0.94,0.42,0.65,0.86,0.79,0.63,0.81,0.70,0.77,0.91,0.89,0.86,0.70,0.73,0.56,\
0.76,0.80,0.40,0.78,0.76,0.84,0.83,0.88,0.90]
plt.plot(herit_b,coef_det,'o')
print('gp vs. H')
print(sc.stats.wilcoxon(herit_b,coef_det))
plt.plot([0,1],[0,1],':k', linewidth = 0.75)
plt.xticks(fontfamily='monospace')
plt.yticks(fontfamily='monospace')
plt.xlabel('Broad sense heritability (H)')
plt.ylabel('Variance explained by G-P atlas')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig(target_folder+'coef_det_vs_H2_g_p.svg')
plt.savefig(target_folder+'coef_det_vs_H2_g_p.png')
plt.close()

