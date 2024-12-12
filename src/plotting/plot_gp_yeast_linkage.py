import os as os
import pickle as pk
import sys as sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import metrics

"""This script runs an analysis of the variable importance measures provided by
 on the output a G-P Atlas analysis of yeast data from DOI_FOR_YEAST_PAPER and
 presented in DOI_FOR_PUB. It creates a series of figures
 presented in that pub. It is intended to be used as follows:
 python3 plot_gp_yeast_linkage.py [PATH TO G-P ATLAS OUTPUT]"""

target_folder = sys.argv[1]  # folder containing the output of G-P Atlas when run on yeast data

# load phenotypes and phenotype predictions.  predictions based on genetic data
gen_encodings = pk.load(open(target_folder + "phens_phen_encodings_dng_attr.pk", "rb"))

# load the variable imporance measures for genes
gp_attr = pk.load(open(target_folder + "g_p_attr.pk", "rb"))

# load data for the published analysis of the yeast data
yeast_chr_dat = pk.load(open(target_folder + "../yeast_cross_test.pk", "rb"))

# make useful colormap
default_cycler = plt.rcParams["axes.prop_cycle"]
colors = [c["color"] for c in default_cycler]
cmap = ListedColormap(colors)

# load previously created yeast linkage data and create some useful transformations of those data
yeast_linkage_data = (
    open(
        target_folder
        + "../41586_2013_BFnature11867_MOESM88_ESM_yeat_\
linkage_data.csv"
    )
    .read()
    .split("\n")
)
yeast_linkage_data = [x.split(",") for x in yeast_linkage_data]
linked_genes = [x[2] + "_" + x[3] for x in yeast_linkage_data[1:-1]]
linked_marker_dat = [[y for y in yeast_chr_dat["loci"] if z in y][0] for z in linked_genes]
linked_marker_index = [yeast_chr_dat["loci"].index(x) for x in linked_marker_dat]
sorted_linkage_data = sorted(yeast_linkage_data[1:-1], key=lambda x: (int(x[2]), int(x[5])))
sorted_linked_genes = [x[2] + "_" + x[3] for x in sorted_linkage_data]
sorted_marker_dat = [[y for y in yeast_chr_dat["loci"] if z in y][0] for z in sorted_linked_genes]
sorted_linked_marker_index = [yeast_chr_dat["loci"].index(x) for x in sorted_marker_dat]
sorted_variance_explained = [float(x[1]) for x in sorted_linkage_data]
sorted_absolute_marker_position = [int(x.split("_")[0]) for x in yeast_chr_dat["loci"]]

# create a dictionary for the absolute positions of markers
marker_absolute_position_dict = {}
for x in yeast_chr_dat["loci"]:
    abs_pos, chromosome, chr_pos, _, _ = x.split("_")
    key = chromosome.strip("chr").lstrip("0") + "_" + chr_pos
    marker_absolute_position_dict[key] = int(abs_pos)

# cluster yeast loci influencing phenotypes. Join genetic regions based on
# overlapping confidence intervals
locus_cluster_ids = [1]
current_id = 1
interval_start = [int(sorted_linkage_data[0][5])]
interval_stop = []
current_chromosome = 1
chromosome_labels = [1]
chromosome_start = [0]
chromosome_stop = []
for n in range(len(sorted_linkage_data) - 1):
    if (
        int(sorted_linkage_data[n + 1][5]) < int(sorted_linkage_data[n][6])
        and int(sorted_linkage_data[n + 1][2]) == current_chromosome
    ):
        locus_cluster_ids.append(current_id)
        chromosome_labels.append(current_chromosome)
    elif (
        int(sorted_linkage_data[n + 1][5]) >= int(sorted_linkage_data[n][6])
        and int(sorted_linkage_data[n + 1][2]) == current_chromosome
    ):
        current_id += 1
        locus_cluster_ids.append(current_id)
        interval_stop.append(int(sorted_linkage_data[n][6]))
        interval_start.append(int(sorted_linkage_data[n + 1][5]))
        chromosome_labels.append(current_chromosome)
    elif int(sorted_linkage_data[n + 1][2]) != current_chromosome:
        current_id += 1
        locus_cluster_ids.append(current_id)
        interval_stop.append(int(sorted_linkage_data[n][6]))
        interval_start.append(int(sorted_linkage_data[n + 1][5]))
        current_chromosome += 1
        chromosome_labels.append(current_chromosome)
interval_stop.append(int(sorted_linkage_data[-1][6]))

clusters, chr_labels_by_cluster = zip(
    *sorted(list(set(zip(locus_cluster_ids, chromosome_labels, strict=False))), key=lambda x: x[0]),
    strict=False,
)

# start plotting the genetic linkage figure
plt.figure(figsize=(15, 5.5))

# plot positions of linked intervals
absolute_interval_start = [
    marker_absolute_position_dict[str(chr_labels_by_cluster[n]) + "_" + str(interval_start[n])]
    for n in range(len(interval_start))
]
absolute_interval_stop = [
    marker_absolute_position_dict[str(chr_labels_by_cluster[n]) + "_" + str(interval_stop[n])]
    for n in range(len(interval_stop))
]
for n in range(len(absolute_interval_start)):
    plt.axvspan(absolute_interval_start[n], absolute_interval_stop[n], alpha=0.25)

# plot bars indicating chromosome
chromosome_start = [0]
for n in range(len(yeast_chr_dat["loci"]) - 1):
    _, chromosome, _, _, _ = yeast_chr_dat["loci"][n].split("_")
    abs_pos, next_chromosome, _, _, _ = yeast_chr_dat["loci"][n + 1].split("_")
    if int(next_chromosome.strip("chr")) == int(chromosome.strip("chr")) + 1:
        chromosome_start.append(int(abs_pos))
chromosome_start.append(int(yeast_chr_dat["loci"][-1].split("_")[0]))

# format attributtion data for plotting
gp_attr = np.mean(np.array(gp_attr) ** 2, axis=0)  # make attr mean squared across individuals
gp_attr_loci = gp_attr.reshape(int(len(gp_attr) / 2), 2).T
abs_attr_loc1 = np.array(gp_attr_loci[0])
abs_attr_loc2 = np.array(gp_attr_loci[1])
max_attr = np.max([abs_attr_loc1, abs_attr_loc2])
abs_attr_loc1 = abs_attr_loc1 / max_attr
abs_attr_loc2 = abs_attr_loc2 / max_attr
max_attrs = [np.max([abs_attr_loc1[n], abs_attr_loc2[n]]) for n in range(len(abs_attr_loc1))]

# calculate the 95th percentile of the attributions
percentile = np.percentile(max_attrs, 95)

# plot attribution data
plt.plot(sorted_absolute_marker_position, max_attrs, "o", markersize=2)

# format the plot
plt.plot(
    [sorted_absolute_marker_position[n] for n in sorted_linked_marker_index],
    sorted_variance_explained,
    "o",
    markersize=2,
)
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.ylabel("Scaled mean squared variable importance")
plt.xlabel("Genomic position (nucleotides, numbers indicate chromosomes)")
plt.ylim(0, 0.45)
plt.xlim(0, chromosome_start[-1])
for n in range(0, len(chromosome_start) - 1, 2):
    plt.plot(
        [chromosome_start[n], chromosome_start[n + 1]],
        [0, 0],
        "k",
        linewidth=8,
        solid_capstyle="butt",
    )

for n in range(1, len(chromosome_start), 2):
    plt.plot(
        [chromosome_start[n], chromosome_start[n + 1]],
        [0, 0],
        "r",
        linewidth=8,
        solid_capstyle="butt",
    )
plt.axhline(percentile, linestyle="dotted", c="k")
plt.xticks(
    [
        chromosome_start[n] + (chromosome_start[n + 1] - chromosome_start[n]) / 2
        for n in range(len(chromosome_start) - 1)
    ],
    [str(x) for x in range(1, 17)],
)
plt.savefig(target_folder + "attribute_vs_linked_region_plot.svg")
plt.savefig(target_folder + "attribute_vs_linked_region_plot.png")
plt.show()
plt.close()

# classification of linked loci for ROC
# classification of clusters
true_labels = [0]
for _x in range(len(clusters)):
    true_labels += [1]
    true_labels += [0]

max_attr_per_cluster = [
    max(
        [
            max_attrs[n]
            for n in range(len(max_attrs))
            if sorted_absolute_marker_position[n] > absolute_interval_start[m]
            and sorted_absolute_marker_position[n] < absolute_interval_stop[m]
        ]
    )
    for m in range(len(absolute_interval_start))
]

max_attr_per_non_linked = [
    [
        max_attrs[n]
        for n in range(len(max_attrs))
        if sorted_absolute_marker_position[n] < absolute_interval_start[m]
        and sorted_absolute_marker_position[n] > absolute_interval_stop[m - 1]
    ]
    for m in range(1, len(absolute_interval_start))
]
max_attr_per_non_linked = (
    [[]]
    + max_attr_per_non_linked
    + [
        [
            max_attrs[n]
            for n in range(len(max_attrs))
            if sorted_absolute_marker_position[n] > absolute_interval_stop[-1]
        ]
    ]
)

max_attr_per_non_linked = [max(x) if len(x) > 0 else 0 for x in max_attr_per_non_linked]

scores = [
    item
    for sublist in zip(max_attr_per_non_linked, max_attr_per_cluster + [[]], strict=False)
    for item in sublist
][:-1]

# ROC
fpr, tpr, thresholds = metrics.roc_curve(true_labels, scores)
auc = metrics.roc_auc_score(true_labels, scores)
plt.plot(fpr, tpr)
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.xlabel("False positive rate")
plt.ylabel("False negative rate")
plt.plot([0, 1], [0, 1], "--k")
plt.axis("scaled")
plt.text(0.65, 0.25, "AUC = " + str(np.around(auc, 2)), fontsize=18)
plt.savefig(target_folder + "ROC_for_clustered_linked_markers.svg")
plt.savefig(target_folder + "ROC_for_clustered_linked_markers.png")
plt.close()

# classification for all linked markers
abs_marker_conf_start_stop = [
    [
        marker_absolute_position_dict[x[2] + "_" + x[5]],
        marker_absolute_position_dict[x[2] + "_" + x[6]],
    ]
    for x in sorted_linkage_data
]
attr_per_locus = [
    [
        max_attrs[n]
        for n in range(len(max_attrs))
        if sorted_absolute_marker_position[n] >= m[0] and sorted_absolute_marker_position[n] <= m[1]
    ]
    for m in abs_marker_conf_start_stop
]
max_attr_per_locus = [max(x) if len(x) > 0 else 0 for x in attr_per_locus]

# variance explained vs max variable importance per linked locus
sns.jointplot(
    x=np.array(sorted_variance_explained) / max(sorted_variance_explained),
    y=np.array(max_attr_per_locus) / max(max_attr_per_locus),
    hue=0,
)
print(
    sc.stats.pearsonr(
        np.array(sorted_variance_explained) / max(sorted_variance_explained),
        np.array(max_attr_per_locus) / max(max_attr_per_locus),
    )
)
plt.plot([0, 1], [0, 1], ":k")
plt.xlim(0, 1.01)
plt.ylim(0, 1.01)
plt.xlabel("Scaled variance explained per linked locus")
plt.ylabel("Scaled variabe importance per linked locus")
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.savefig(target_folder + "variance_exp_vs_var_imp.svg")
plt.savefig(target_folder + "variance_exp_vs_var_imp.png")
plt.close()


# cunulative fraction of linked loci identified based on variable importance
percentiles = [np.percentile(max_attrs, x) for x in range(1, 101)]
cumulative_fraction = [
    len([x for x in max_attr_per_locus if x < y]) / len(max_attr_per_locus) for y in percentiles
]
cumulative_fraction_reverse = list(reversed(cumulative_fraction))
plt.plot(list(reversed(list(np.array(range(0, 100)) / 100))), cumulative_fraction_reverse)
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.ylabel("Fraction of linked loci identified")
plt.xlabel("Variable importance threshold (percentile)")
plt.ylim(0, 1.01)
plt.xlim(0, 1.001)
plt.savefig(target_folder + "fraction_loc_identified_per_variable_importance.svg")
plt.savefig(target_folder + "fraction_loc_identified_per_variable_importance.png")
plt.close()

# feature importance vs variance explained
max_attr_per_linked_locus = [max_attrs[n] for n in sorted_linked_marker_index]
sns.jointplot(
    x=np.array(sorted_variance_explained) / max(sorted_variance_explained),
    y=np.array(max_attr_per_linked_locus) / max(max_attr_per_linked_locus),
    hue=0,
)
print(
    sc.stats.pearsonr(
        np.array(sorted_variance_explained) / max(sorted_variance_explained),
        np.array(max_attr_per_linked_locus) / max(max_attr_per_linked_locus),
    )
)
print(len(sorted_absolute_marker_position))
print(len(max_attrs))
print(len(sorted_linked_marker_index))
print(len(sorted_variance_explained))
plt.xlim(0, 1.01)
plt.ylim(0, 1.01)
plt.xlabel("Scaled variance explained per linked locus")
plt.ylabel("Scaled variabe importance per linked locus")
plt.xticks(fontfamily="monospace")
plt.yticks(fontfamily="monospace")
plt.savefig(target_folder + "variance_exp_vs_var_imp_per_marker.svg")
plt.savefig(target_folder + "variance_exp_vs_var_imp_per_marker.png")
plt.show()
