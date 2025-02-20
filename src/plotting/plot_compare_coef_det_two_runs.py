import pickle as pk
import sys

import helper_functions as hf
import matplotlib.pyplot as plt
import scipy as sc

"""This script compares the output of two different G-P Atlas runs and plots the
 coefficient of determination for both models"""

target_folder_1 = sys.argv[1]  # folder containing the first output of G-P Atlas

target_folder_2 = sys.argv[2]  # folder containing the second output of G-P Atlas

# load real and predicted phenotypes. predictions are based on genotypes
with open(target_folder_1 + "phens_phen_encodings_dng_attr.pk", "rb") as data:
    gen_encodings_1 = pk.load(data)

with open(target_folder_2 + "phens_phen_encodings_dng_attr.pk", "rb") as data:
    gen_encodings_2 = pk.load(data)

# calculate the coefficient of determination for both models
coef_det_1 = [
    hf.calc_coef_of_det(gen_encodings_1[0][n], gen_encodings_1[1][n])
    for n in range(len(gen_encodings_1[0]))
]

coef_det_2 = [
    hf.calc_coef_of_det(gen_encodings_2[0][n], gen_encodings_2[1][n])
    for n in range(len(gen_encodings_2[0]))
]

# plot the comparison of the two coefficients of determination
plt.plot(coef_det_2, coef_det_1, "o")
print("coefficient of determination 1 vs. coefficient of determination 2")
print(sc.stats.wilcoxon(coef_det_1, coef_det_2))
print(sc.stats.pearsonr(coef_det_1, coef_det_2))
print(sc.stats.linregress(coef_det_1, coef_det_2))

plt.plot([0, 1], [0, 1], ":k", linewidth=0.75)
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.ylabel("Variance explained by full G-P atlas")
plt.xlabel("Variance explained by G-P atlas with no genetic interactions")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.savefig(target_folder_1 + "full_vs_1_layer_g_p.svg")
plt.savefig(target_folder_1 + "full_vs_1_layer_g_p.png")
plt.show()
plt.close()
