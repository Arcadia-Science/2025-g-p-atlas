import pickle as pk
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import helper_functions as hf

"""This script runs an analysis of the variable importance measures provided by
 on the output a G-P Atlas of simulated data from 10.57844/arcadia-5953-995f and
 presented in 10.57844/arcadia-d316-721f. It creates a series of figures
 presented in that pub. It is intended to be used as follows:
 python3 plot_gp_simulated_linkage.py [PATH TO G-P ATLAS OUTPUT]"""

# folder containing output of g-p atlas when run on simulated data
target_folder = sys.argv[1]

# load the variable imporance measures for genes
with open(target_folder + "g_p_attr.pk", "rb") as data:
    gene_attributions = pk.load(data))

# load the test data dictionary for the analysis
with open(target_folder + "../test.pk", "rb") as data:
    test_data = pk.load(data)

# plot ROC curve for alleles determining the utility of allele attributions in
# identifying influential genes
max_weights = np.max(
    [np.array(x).ravel() for x in test_data["weights"]], axis=0
)
labels = np.where(max_weights != 0, 1, 0)
max_attr = np.mean(np.array(gene_attributions) ** 2, axis=0)
fpr, tpr, thresholds = metrics.roc_curve(labels, max_attr)
auc = metrics.roc_auc_score(labels, max_attr)
plt.axis("scaled")
plt.plot([0, 1], [0, 1], "--", linewidth=1)
plt.ylim(-0.0, 1.02)
plt.plot(fpr, tpr, linewidth=1)
plt.xlim(-0.0, 1.02)
plt.xlabel("False positive rate")
plt.ylabel("False negative rate")
plt.text(0.65, 0.25, "AUC = " + str(np.around(auc, 2)))
plt.yticks(fontfamily="monospace")
plt.xticks(fontfamily="monospace")
plt.savefig(target_folder + "ROC_gene_classification.svg")
plt.savefig(target_folder + "ROC_gene_classification.png")
plt.close()

# plot genetic attributions vs additive genetic contribution of all alleles
max_attr = max_attr[labels>0] 
add_val = max_weights[labels>0]
one_percent_fpr = hf.calculate_fpr_threshold(fpr, thresholds)
print(len([x for x in max_attr if x > one_percent_fpr]))
print(len(max_attr))
print(len([y for y in [max(x) for x in [max_attr[i:i+3] for i in range(0,len(max_attr),3)]] if y > one_percent_fpr]))
print(len(max_attr)/3)

plt.scatter(add_val, max_attr, marker="o")
plt.plot([-0.02, 10.02], [one_percent_fpr, one_percent_fpr], color="C1", linewidth=1)
plt.legend("1% FPR", fontsize=14)
plt.ylabel("Mean squared variable importance")
plt.xlabel("Additive genetic influence (au)")
plt.yticks(fontfamily="monospace")
plt.xticks(fontfamily="monospace")
plt.savefig(target_folder + "MSV_vs_additive_contribution.svg")
plt.savefig(target_folder + "MSV_vs_additive_contribution.png")
plt.close()


# interactors and pleiotropy
# replots the additive genetic contributions vs the allele atribution determined by G-P Atlas
# and colors them by pleiotropy
number_of_alleles = test_data["n_loci"]*test_data["n_as"]
ints = np.zeros()
for x in test_data["interact_matrix"]:
    for y in x:
        ints[y[0] * 3 + y[-1]] += 1
        ints[y[1] * 3 + y[-1]] += 1

influential_ints = [ints[n] for n in range(len(labels)) if labels[n] > 0]

pleio = np.zeros(9000)
for x in test_data["pleiotropy_matrix"]:
    for y in x:
        pleio[3 * y[-1]] += 1
        pleio[3 * y[-1] + 1] += 1
        pleio[3 * y[-1] + 2] += 1

influential_pleio = [pleio[n] for n in range(len(labels)) if labels[n] > 0]

plt.scatter(add_val, max_attr, c=influential_pleio, marker="o")
plt.plot([-0.02, 10.02], [one_percent_fpr, one_percent_fpr], color="C1", linewidth=3)
plt.legend("1% FPR", fontsize=14)
plt.ylabel("Mean squared variable importance")
plt.xlabel("Additive genetic influence (au)")
plt.yticks(fontfamily="monospace")
plt.xticks(fontfamily="monospace")
plt.savefig(target_folder + "pleiotropy_plot_attribution_vs_weight.svg")
plt.close()
