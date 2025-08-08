import pickle as pk
import sys

import helper_functions as hf
import vida as v
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

"""This script runs an analysis of the variable importance measures provided by
 on the output a G-P Atlas of simulated data from 10.57844/arcadia-5953-995f and
 presented in 10.57844/arcadia-d316-721f. It creates a series of figures
 presented in that pub. It is intended to be used as follows:
 python3 plot_gp_simulated_linkage.py [PATH TO G-P ATLAS OUTPUT]"""

# folder containing output of g-p atlas when run on simulated data
target_folder = sys.argv[1]
test_dataset_file = sys.argv[2]

# load the variable imporance measures for genes
with open(target_folder + "g_p_attr.pk", "rb") as data:
    gene_attributions = pk.load(data)

# load the test data dictionary for the analysis
with open(test_dataset_file, "rb") as data:
    test_data = pk.load(data)

# plot ROC curve for alleles determining the utility of allele attributions in
# identifying influential genes
max_weights = np.max([np.array(x).ravel() for x in test_data["weights"]], axis=0)
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
max_attr = max_attr[labels > 0]
add_val = max_weights[labels > 0]
one_percent_fpr = hf.calculate_fpr_threshold(fpr, thresholds)
print("number of alleles above the 1% false positive rate threshold")
print(len([x for x in max_attr if x > one_percent_fpr]))
print("total number of alleles influencing phenotypes")
print(len(max_attr))
print("number of loci influencing phenotypes above the 1% false positive rate")
print(
    len(
        [
            y
            for y in [max(x) for x in [max_attr[i : i + 3] for i in range(0, len(max_attr), 3)]]
            if y > one_percent_fpr
        ]
    )
)
print("total number of loci influencing phenotypes")
print(int(len(max_attr) / 3))

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

# create variables for allele specific indices for the followingplots
allele_interaction_indices = list(
    set(hf.known_interaction_indices(test_data["interact_matrix"]).flatten())
)  # indices of alleles involved in interactions
allele_direct_indices = [
    x * 3 for x in list(set(np.array(test_data["inds_of_loci_influencing_phen"]).flatten()))
]  # indices of alleles directly influencing phenotypes
allele_direct_only_indices = [
    x for x in allele_direct_indices if x not in allele_interaction_indices
]
all_loci_indices = np.concatenate([allele_interaction_indices, allele_direct_only_indices])

# create a Q-Q plot to evaluate kurtosis
q_q_plot = hf.plot_qq_with_bands(
    np.array(gene_attributions), allele_interaction_indices, allele_direct_only_indices
)
plt.savefig(target_folder + "variable_importance_qq_interacting_vs_non.svg")
plt.savefig(target_folder + "variable_importance_qq_interacting_vs_non.png")
plt.close()
boot_strap_kurtosis_results = hf.bootstrap_kurtosis_test(
    np.array(gene_attributions), allele_interaction_indices, allele_direct_only_indices
)
hf.plot_kurtosis_results(boot_strap_kurtosis_results)
plt.savefig(target_folder + "boot_kurtosis_int_vs_non_int.svg")
plt.savefig(target_folder + "boot_kurtosis_int_vs_non_int.png")
plt.close()

# run vida analysis and make plots
vida_metrics = v.calculate_optimized_hybrid_vida_scores(
    np.array(gene_attributions),
    all_loci_indices,
    z_threshold=2.0,
    clumpiness_percentile=5,
    clumpiness_method="concentration",
)

performance_stats = v.plot_vida_performance(
    vida_metrics,
    allele_interaction_indices,
    allele_direct_indices,
    metric_name="vida_score",
    title_suffix=" (Optimized Parameters)",
)

plt.savefig(target_folder + "vida_plots_int_vs_non_int.svg")
plt.savefig(target_folder + "vida_plots_int_vs_non_int.png")
plt.close()
