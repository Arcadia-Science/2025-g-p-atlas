import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from itertools import product


def calculate_optimized_hybrid_vida_scores(
    vi_matrix,
    all_loci_indices,
    z_threshold=2.0,
    clumpiness_percentile=10,
    clumpiness_method="concentration",
    extreme_percentile=95,
):
    """
    Optimized version of hybrid vida with configurable parameters

    Parameters:
    -----------
    vi_matrix : numpy.ndarray
        The original data matrix with shape (n_samples, n_loci)
    all_loci_indices : array-like
        Indices of all loci to analyze
    z_threshold : float
        Z-score threshold for extreme values
    clumpiness_percentile : float
        Percentile for clumpiness calculation (e.g., 10 = top 10%)
    clumpiness_method : str
        Method for calculating clumpiness ('concentration', 'entropy', 'gini', 'adaptive')
    extreme_percentile : float
        Percentile threshold for defining extreme values in partner identification

    Returns:
    --------
    metrics_df : pandas.DataFrame
        DataFrame with optimized hybrid vida metrics for all loci
    """

    n_samples, n_loci = vi_matrix.shape
    all_loci_indices = np.array(all_loci_indices)
    all_loci_indices = all_loci_indices[all_loci_indices < n_loci]

    # Step 1: Calculate z-scores for all loci
    z_matrix = np.zeros_like(vi_matrix)

    for idx in all_loci_indices:
        locus_data = vi_matrix[:, idx]
        non_zero_mask = locus_data != 0

        if np.sum(non_zero_mask) >= 4:
            non_zero_values = locus_data[non_zero_mask]
            mean_val = np.mean(non_zero_values)
            std_val = np.std(non_zero_values)

            if std_val > 0:
                z_matrix[non_zero_mask, idx] = (non_zero_values - mean_val) / std_val

    # Step 2: Enhanced partner identification with configurable extreme threshold
    locus_pair_scores = {}

    for i, locus1 in enumerate(all_loci_indices):
        for j, locus2 in enumerate(all_loci_indices):
            if j <= i:
                continue

            # Get z-scores for both loci
            z1 = z_matrix[:, locus1]
            z2 = z_matrix[:, locus2]

            # Find samples where both have values
            valid_samples = (z1 != 0) & (z2 != 0)

            if np.sum(valid_samples) < 4:
                continue

            # Calculate correlation
            corr = np.corrcoef(z1[valid_samples], z2[valid_samples])[0, 1]

            # Calculate co-occurrence of extreme values with configurable threshold
            extreme1 = np.abs(z1) > z_threshold
            extreme2 = np.abs(z2) > z_threshold

            co_extreme = np.sum(extreme1 & extreme2)
            expected_co_extreme = np.sum(extreme1) * np.sum(extreme2) / n_samples

            co_extreme_ratio = co_extreme / expected_co_extreme if expected_co_extreme > 0 else 0

            # Store similarity score
            similarity = (abs(corr) + co_extreme_ratio) / 2

            locus_pair_scores[(locus1, locus2)] = similarity

    # Step 3: Partner identification with adaptive threshold
    locus_partners = {}

    for locus in all_loci_indices:
        # Find pairs involving this locus
        pairs = [(l1, l2) for (l1, l2) in locus_pair_scores.keys() if l1 == locus or l2 == locus]

        if not pairs:
            continue

        # Sort by similarity
        sorted_pairs = sorted(pairs, key=lambda p: locus_pair_scores[p], reverse=True)

        # Take top 10% as partners
        n_partners = max(1, int(0.1 * len(sorted_pairs)))
        partner_pairs = sorted_pairs[:n_partners]

        # Extract partner loci
        partners = [l2 if l1 == locus else l1 for (l1, l2) in partner_pairs]

        locus_partners[locus] = partners

    # Step 4: Enhanced clumpiness calculation
    def calculate_clumpiness(values, method, percentile):
        """Calculate clumpiness using different methods"""
        if len(values) < 3:
            return 0

        sorted_vals = np.sort(np.abs(values))[::-1]  # Sort by absolute value, descending

        if method == "concentration":
            # Original method: concentration in top percentile
            top_pct = int(max(1, np.ceil(percentile / 100 * len(sorted_vals))))
            return np.sum(sorted_vals[:top_pct]) / np.sum(sorted_vals)

        elif method == "entropy":
            # Entropy-based clumpiness (lower entropy = more clumped)
            # Bin the values and calculate entropy
            hist, _ = np.histogram(sorted_vals, bins=min(10, len(sorted_vals) // 2))
            hist = hist / np.sum(hist)  # Normalize
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log2(hist))
            max_entropy = np.log2(len(hist)) if len(hist) > 1 else 1
            return 1 - (entropy / max_entropy)  # Higher value = more clumped

        elif method == "gini":
            # Gini coefficient (higher = more unequal = more clumped)
            n = len(sorted_vals)
            cumsum = np.cumsum(sorted_vals)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

        elif method == "adaptive":
            # Adaptive method: use different percentiles based on distribution
            if stats.kurtosis(values) > 1:  # High kurtosis
                top_pct = int(max(1, np.ceil(5 / 100 * len(sorted_vals))))  # Top 5%
            else:  # Low kurtosis
                top_pct = int(max(1, np.ceil(15 / 100 * len(sorted_vals))))  # Top 15%
            return np.sum(sorted_vals[:top_pct]) / np.sum(sorted_vals)

        else:
            raise ValueError(f"Unknown clumpiness method: {method}")

    # Step 5: Calculate sample-level metrics with enhanced clumpiness
    sample_metrics = []

    for i in range(n_samples):
        for locus in locus_partners:
            partners = locus_partners[locus]

            if len(partners) < 3:
                continue

            # Get non-partners
            non_partners = [l for l in all_loci_indices if l != locus and l not in partners]

            # Extract z-scores for this sample
            partner_z = z_matrix[i, partners]
            non_partner_z = z_matrix[i, non_partners]

            # Filter out zeros
            partner_z = partner_z[partner_z != 0]
            non_partner_z = non_partner_z[non_partner_z != 0]

            if len(partner_z) < 3 or len(non_partner_z) < 3:
                continue

            # Calculate metrics
            partner_extreme_count = np.sum(np.abs(partner_z) > z_threshold)
            non_partner_extreme_count = np.sum(np.abs(non_partner_z) > z_threshold)

            partner_extreme_ratio = partner_extreme_count / len(partner_z)
            non_partner_extreme_ratio = non_partner_extreme_count / len(non_partner_z)

            # Enhanced clumpiness calculation
            partner_clumpiness = calculate_clumpiness(
                partner_z, clumpiness_method, clumpiness_percentile
            )
            non_partner_clumpiness = calculate_clumpiness(
                non_partner_z, clumpiness_method, clumpiness_percentile
            )

            # Store metrics
            sample_metrics.append(
                {
                    "sample_idx": i,
                    "locus": locus,
                    "extreme_ratio_diff": partner_extreme_ratio - non_partner_extreme_ratio,
                    "clumpiness_diff": partner_clumpiness - non_partner_clumpiness,
                }
            )

    # Step 6: Project sample-level metrics back to locus level
    locus_projected_metrics = []

    for locus in all_loci_indices:
        # Find samples where this locus has extreme values
        locus_z = z_matrix[:, locus]
        extreme_samples = np.where(np.abs(locus_z) > z_threshold)[0]

        if len(extreme_samples) < 3:
            continue

        # Get metrics for these samples
        locus_samples = [
            m for m in sample_metrics if m["locus"] == locus and m["sample_idx"] in extreme_samples
        ]

        if len(locus_samples) < 3:
            continue

        # Calculate averages
        avg_extreme_ratio_diff = np.mean([s["extreme_ratio_diff"] for s in locus_samples])
        avg_clumpiness_diff = np.mean([s["clumpiness_diff"] for s in locus_samples])

        # Store metrics
        locus_projected_metrics.append(
            {
                "locus": locus,
                "n_extreme_samples": len(extreme_samples),
                "avg_extreme_ratio_diff": avg_extreme_ratio_diff,
                "avg_clumpiness_diff": avg_clumpiness_diff,
                "vida_score": avg_extreme_ratio_diff * avg_clumpiness_diff,
            }
        )

    # Convert to DataFrame
    metrics_df = pd.DataFrame(locus_projected_metrics)

    # Sort by avg_clumpiness_diff (your best metric)
    if len(metrics_df) > 0:
        metrics_df = metrics_df.sort_values("avg_clumpiness_diff", ascending=False)

    return metrics_df


def optimize_vida_parameters(
    vi_matrix,
    all_loci_indices,
    allele_interaction_indices,
    allele_direct_indices,
    z_thresholds=[1.5, 2.0, 2.5, 3.0],
    clumpiness_percentiles=[5, 10, 15, 20],
    clumpiness_methods=["concentration", "gini", "entropy", "adaptive"],
):
    """
    Systematically optimize vida parameters
    """

    print("=== vida PARAMETER OPTIMIZATION ===")
    print(
        f"Testing {len(z_thresholds)} z-thresholds × {len(clumpiness_percentiles)} percentiles × {len(clumpiness_methods)} methods"
    )
    print(
        f"Total combinations: {len(z_thresholds) * len(clumpiness_percentiles) * len(clumpiness_methods)}"
    )
    print("-" * 60)

    results = []

    # Test all parameter combinations
    for z_thresh, clump_pct, clump_method in product(
        z_thresholds, clumpiness_percentiles, clumpiness_methods
    ):
        try:
            print(
                f"Testing: z_threshold={z_thresh}, clumpiness_pct={clump_pct}, method={clump_method}"
            )

            # Calculate metrics with current parameters
            metrics_df = calculate_optimized_hybrid_vida_scores(
                vi_matrix,
                all_loci_indices,
                z_threshold=z_thresh,
                clumpiness_percentile=clump_pct,
                clumpiness_method=clump_method,
            )

            if len(metrics_df) == 0:
                print(f"  No results - skipping")
                continue

            # Add labels for evaluation
            labeled_df = metrics_df.copy()
            labeled_df["is_interaction"] = False

            for idx in allele_interaction_indices:
                mask = labeled_df["locus"] == idx
                if any(mask):
                    labeled_df.loc[mask, "is_interaction"] = True

            # Filter to known loci
            known_loci_mask = labeled_df["locus"].isin(
                np.concatenate([allele_interaction_indices, allele_direct_indices])
            )
            labeled_df = labeled_df[known_loci_mask]

            if len(labeled_df) < 10:
                print(f"  Too few labeled loci - skipping")
                continue

            # Evaluate performance for each metric
            y_true = labeled_df["is_interaction"].astype(int)

            for metric in ["avg_extreme_ratio_diff", "avg_clumpiness_diff", "vida_score"]:
                if metric in labeled_df.columns:
                    scores = labeled_df[metric].fillna(labeled_df[metric].median()).values

                    # Test both directions
                    auc_pos = roc_auc_score(y_true, scores)
                    auc_neg = roc_auc_score(y_true, -scores)

                    best_auc = max(auc_pos, auc_neg)
                    direction = "higher" if auc_pos >= auc_neg else "lower"

                    results.append(
                        {
                            "z_threshold": z_thresh,
                            "clumpiness_percentile": clump_pct,
                            "clumpiness_method": clump_method,
                            "metric": metric,
                            "auc": best_auc,
                            "direction": direction,
                            "n_loci": len(labeled_df),
                        }
                    )

            print(f"  Completed successfully")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Convert to DataFrame and analyze results
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No successful parameter combinations!")
        return None

    # Find best parameters for each metric
    print("\n=== OPTIMIZATION RESULTS ===")
    print("-" * 40)

    best_configs = {}

    for metric in ["avg_extreme_ratio_diff", "avg_clumpiness_diff", "vida_score"]:
        metric_results = results_df[results_df["metric"] == metric]
        if len(metric_results) > 0:
            best_row = metric_results.loc[metric_results["auc"].idxmax()]
            best_configs[metric] = best_row

            print(f"\nBest {metric}:")
            print(f"  AUC: {best_row['auc']:.4f}")
            print(f"  Z-threshold: {best_row['z_threshold']}")
            print(f"  Clumpiness %ile: {best_row['clumpiness_percentile']}")
            print(f"  Clumpiness method: {best_row['clumpiness_method']}")
            print(f"  Direction: {best_row['direction']} values indicate interaction")

    # Find overall best configuration
    overall_best = results_df.loc[results_df["auc"].idxmax()]
    print(f"\n*** OVERALL BEST CONFIGURATION ***")
    print(f"Metric: {overall_best['metric']}")
    print(f"AUC: {overall_best['auc']:.4f}")
    print(
        f"Parameters: z_threshold={overall_best['z_threshold']}, "
        f"clumpiness_percentile={overall_best['clumpiness_percentile']}, "
        f"method='{overall_best['clumpiness_method']}'"
    )

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # AUC by z-threshold
    z_thresh_perf = results_df.groupby(["z_threshold", "metric"])["auc"].max().reset_index()
    for metric in z_thresh_perf["metric"].unique():
        metric_data = z_thresh_perf[z_thresh_perf["metric"] == metric]
        axes[0, 0].plot(metric_data["z_threshold"], metric_data["auc"], "o-", label=metric)
    axes[0, 0].set_xlabel("Z-threshold")
    axes[0, 0].set_ylabel("Best AUC")
    axes[0, 0].set_title("Performance vs Z-threshold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AUC by clumpiness percentile
    clump_perf = results_df.groupby(["clumpiness_percentile", "metric"])["auc"].max().reset_index()
    for metric in clump_perf["metric"].unique():
        metric_data = clump_perf[clump_perf["metric"] == metric]
        axes[0, 1].plot(
            metric_data["clumpiness_percentile"], metric_data["auc"], "o-", label=metric
        )
    axes[0, 1].set_xlabel("Clumpiness Percentile")
    axes[0, 1].set_ylabel("Best AUC")
    axes[0, 1].set_title("Performance vs Clumpiness Percentile")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC by method
    method_perf = results_df.groupby(["clumpiness_method", "metric"])["auc"].max().reset_index()
    methods = method_perf["clumpiness_method"].unique()
    metrics = method_perf["metric"].unique()

    x_pos = np.arange(len(methods))
    width = 0.25

    for i, metric in enumerate(metrics):
        metric_data = method_perf[method_perf["metric"] == metric]
        aucs = [
            metric_data[metric_data["clumpiness_method"] == m]["auc"].iloc[0]
            if len(metric_data[metric_data["clumpiness_method"] == m]) > 0
            else 0
            for m in methods
        ]
        axes[1, 0].bar(x_pos + i * width, aucs, width, label=metric)

    axes[1, 0].set_xlabel("Clumpiness Method")
    axes[1, 0].set_ylabel("Best AUC")
    axes[1, 0].set_title("Performance vs Clumpiness Method")
    axes[1, 0].set_xticks(x_pos + width)
    axes[1, 0].set_xticklabels(methods, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Parameter interaction heatmap for best metric
    best_metric = overall_best["metric"]
    best_metric_data = results_df[results_df["metric"] == best_metric]

    # Create heatmap data
    heatmap_data = best_metric_data.pivot_table(
        values="auc", index="z_threshold", columns="clumpiness_percentile", aggfunc="max"
    )

    im = axes[1, 1].imshow(heatmap_data.values, cmap="viridis", aspect="auto")
    axes[1, 1].set_xticks(range(len(heatmap_data.columns)))
    axes[1, 1].set_xticklabels(heatmap_data.columns)
    axes[1, 1].set_yticks(range(len(heatmap_data.index)))
    axes[1, 1].set_yticklabels(heatmap_data.index)
    axes[1, 1].set_xlabel("Clumpiness Percentile")
    axes[1, 1].set_ylabel("Z-threshold")
    axes[1, 1].set_title(f"Parameter Interaction for {best_metric}")

    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1], label="AUC")

    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            if not np.isnan(heatmap_data.iloc[i, j]):
                axes[1, 1].text(
                    j, i, f"{heatmap_data.iloc[i, j]:.3f}", ha="center", va="center", color="white"
                )

    plt.tight_layout()
    plt.show()

    return results_df, best_configs


"""
# Run the optimization
print("Starting vida parameter optimization...")
optimization_results, best_configurations = optimize_vida_parameters(
    np.array(tst_dat),
    all_loci_indices,
    allele_interaction_indices, 
    allele_direct_indices
)"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd


def plot_vida_performance(
    metrics_df,
    allele_interaction_indices,
    allele_direct_indices,
    metric_name="vida_score",
    title_suffix="",
):
    """
    Create ROC curve and enrichment plots for vida method

    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame with vida metrics and locus column
    allele_interaction_indices : array-like
        Indices of known interacting loci
    allele_direct_indices : array-like
        Indices of known non-interacting loci
    metric_name : str
        Name of the metric column to plot ('vida_score', 'avg_clumpiness_diff', etc.)
    title_suffix : str
        Additional text for plot titles
    """

    # Prepare labeled data
    labeled_df = metrics_df.copy()
    labeled_df["is_interaction"] = False

    # Add interaction labels
    for idx in allele_interaction_indices:
        mask = labeled_df["locus"] == idx
        if any(mask):
            labeled_df.loc[mask, "is_interaction"] = True

    # Filter to known loci only
    known_loci_mask = labeled_df["locus"].isin(
        np.concatenate([allele_interaction_indices, allele_direct_indices])
    )
    labeled_df = labeled_df[known_loci_mask]

    if len(labeled_df) == 0:
        print("No labeled data found for plotting!")
        return

    # Get true labels and scores
    y_true = labeled_df["is_interaction"].astype(int)
    scores = labeled_df[metric_name].fillna(labeled_df[metric_name].median()).values

    # Test both directions to find optimal
    from sklearn.metrics import roc_auc_score

    auc_pos = roc_auc_score(y_true, scores)
    auc_neg = roc_auc_score(y_true, -scores)

    if auc_pos >= auc_neg:
        final_scores = scores
        direction = "higher"
        optimal_auc = auc_pos
    else:
        final_scores = -scores
        direction = "lower"
        optimal_auc = auc_neg

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, final_scores)
    roc_auc = auc(fpr, tpr)

    for label in ax1.get_xticklabels():
        label.set_fontfamily("monospace")

    for label in ax1.get_yticklabels():
        label.set_fontfamily("monospace")

    ax1.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], lw=2, linestyle="--", alpha=0.6, label="Random classifier")

    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # plt.yticks(fontfamily="monospace")
    # plt.xticks(fontfamily="monospace")

    # Mark optimal point
    ax1.plot(
        fpr[optimal_idx],
        tpr[optimal_idx],
        "ro",
        markersize=8,
        label=f"Optimal threshold = {optimal_threshold:.3f}",
    )

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    # ax1.set_title(f'ROC Curve - {metric_name.replace("_", " ").title()}{title_suffix}')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Add performance metrics text
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    ax1.text(
        0.6,
        0.2,
        f"Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Plot 2: Enrichment Plot (Percentage of Interacting Loci vs Percentile)
    # Sort by metric (in the direction that indicates interaction)
    if direction == "higher":
        sorted_df = labeled_df.sort_values(metric_name, ascending=False)
    else:
        sorted_df = labeled_df.sort_values(metric_name, ascending=True)

    # Calculate cumulative percentage of interacting loci
    n_total = len(sorted_df)
    n_interacting_total = np.sum(sorted_df["is_interaction"])

    percentiles = np.arange(1, 101)  # 1% to 100%
    enrichment_values = []
    interacting_loci_values = [
        sorted_df[metric_name][n] for n in range(n_total) if sorted_df["is_interaction"][n] == 1
    ]

    for percentile in percentiles:
        """percentile_threshold = np.percentile(sorted_df[metric_name], percentile)
        total_above_threshold = sum(interacting_loci_values > percentile_threshold)
        enrichment_values.append(total_above_threshold/n_interacting_total*100)"""
        # Number of loci in top percentile
        n_top = int(np.ceil(percentile / 100 * n_total))
        print(percentile)
        # Number of interacting loci in top percentile
        n_interacting_top = np.sum(sorted_df.head(n_top)["is_interaction"])

        # Percentage of interacting loci in this percentile
        pct_interacting = n_interacting_top / n_top * 100 if n_top > 0 else 0
        enrichment_values.append(pct_interacting)

    # Expected percentage (random baseline)
    expected_pct = n_interacting_total / n_total * 100

    ax2.plot(percentiles, enrichment_values, "-", label="Observed")

    ax2.axhline(y=expected_pct, linestyle="--", label=f"Expected (random) = {expected_pct:.1f}%")

    for label in ax2.get_xticklabels():
        label.set_fontfamily("monospace")

    for label in ax2.get_yticklabels():
        label.set_fontfamily("monospace")

    ax2.set_xlabel("Percentile Threshold (%)")
    ax2.set_ylabel("Percentage of Interacting Loci (%)")
    # ax2.set_title(f'Enrichment Plot - {metric_name.replace("_", " ").title()}{title_suffix}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 100])
    ax2.set_ylim([0, 100])

    # Add enrichment statistics
    top_10_pct = enrichment_values[9]  # 10th percentile (index 9)
    top_5_pct = enrichment_values[4]  # 5th percentile (index 4)

    enrichment_10 = top_10_pct / expected_pct if expected_pct > 0 else 0
    enrichment_5 = top_5_pct / expected_pct if expected_pct > 0 else 0

    ax2.text(
        60,
        80,
        f"Top 10%: {top_10_pct:.1f}% ({enrichment_10:.1f}x)\nTop 5%: {top_5_pct:.1f}% ({enrichment_5:.1f}x)",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    # Calculate VIDA stats for known interacting and known non-interacting loci
    vida_int = [
        vida_metrics["vida_score"][n]
        for n in range(len(vida_metrics["vida_score"]))
        if vida_metrics["locus"][n] in allele_interaction_indices
    ]
    vida_non_int = [
        vida_metrics["vida_score"][n]
        for n in range(len(vida_metrics["vida_score"]))
        if vida_metrics["locus"][n] in allele_direct_only_indices
    ]
    mwu_results = sc.stats.mannwhitneyu(vida_int, vida_non_int)
    median_int_vida = np.median(vida_int)
    median_non_int_vida = np.median(vida_non_int)

    # Print summary statistics
    print(f"\n=== PERFORMANCE SUMMARY for {metric_name} ===")
    print(f"AUC: {optimal_auc:.4f}")
    print(f"Direction: {direction} values indicate interaction")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Total loci evaluated: {len(labeled_df)}")
    print(f"Interacting loci: {n_interacting_total}")
    print(f"Non-interacting loci: {n_total - n_interacting_total}")
    print(f"\nEnrichment in top 10%: {enrichment_10:.2f}x")
    print(f"Enrichment in top 5%: {enrichment_5:.2f}x")

    return {
        "auc": optimal_auc,
        "optimal_threshold": optimal_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "enrichment_10pct": enrichment_10,
        "enrichment_5pct": enrichment_5,
        "direction": direction,
    }


import numpy as np
from scipy.stats import hypergeom
import pandas as pd
import matplotlib.pyplot as plt


def hypergeometric_enrichment_test(
    vida_results,
    allele_interaction_indices,
    allele_direct_indices,
    metric_name="VIDA_score",
    top_percentiles=[5, 10, 20],
):
    """
    Conduct hypergeometric enrichment test for VIDA results

    Parameters:
    -----------
    vida_results : pandas.DataFrame
        DataFrame with VIDA scores and locus column
    allele_interaction_indices : array-like
        Known interacting loci indices
    allele_direct_indices : array-like
        Known non-interacting loci indices
    metric_name : str
        Column name of the metric to test (default: 'VIDA_score')
    top_percentiles : list
        List of percentiles to test for enrichment (default: [5, 10, 20])

    Returns:
    --------
    enrichment_results : pandas.DataFrame
        Results of enrichment tests for each percentile
    """

    print("=== HYPERGEOMETRIC ENRICHMENT TEST ===")
    print(f"Testing enrichment for metric: {metric_name}")
    print("-" * 50)

    # Prepare data with labels
    labeled_df = vida_results.copy()
    labeled_df["is_interaction"] = False

    # Add interaction labels
    for idx in allele_interaction_indices:
        mask = labeled_df["locus"] == idx
        if any(mask):
            labeled_df.loc[mask, "is_interaction"] = True

    # Filter to known loci only
    known_loci_mask = labeled_df["locus"].isin(
        np.concatenate([allele_interaction_indices, allele_direct_indices])
    )
    labeled_df = labeled_df[known_loci_mask]

    if len(labeled_df) == 0:
        print("No labeled data found!")
        return None

    # Determine optimal direction (higher or lower values indicate interaction)
    from sklearn.metrics import roc_auc_score

    y_true = labeled_df["is_interaction"].astype(int)
    scores = labeled_df[metric_name].fillna(labeled_df[metric_name].median())

    auc_pos = roc_auc_score(y_true, scores)
    auc_neg = roc_auc_score(y_true, -scores)

    if auc_pos >= auc_neg:
        ascending = False  # Higher values indicate interaction
        direction = "higher"
    else:
        ascending = True  # Lower values indicate interaction
        direction = "lower"

    # Sort by metric in the appropriate direction
    sorted_df = labeled_df.sort_values(metric_name, ascending=ascending)

    # Population parameters
    N = len(sorted_df)  # Total population size
    K = np.sum(sorted_df["is_interaction"])  # Total number of interactions in population

    print(f"Population parameters:")
    print(f"  Total loci (N): {N}")
    print(f"  Total interactions (K): {K}")
    print(f"  Total non-interactions: {N - K}")
    print(f"  Direction: {direction} values indicate interaction")
    print()

    # Test enrichment at different percentiles
    enrichment_results = []

    for percentile in top_percentiles:
        # Sample parameters
        n = int(np.ceil(percentile / 100 * N))  # Sample size (top percentile)
        k = np.sum(sorted_df.head(n)["is_interaction"])  # Observed interactions in sample

        # Hypergeometric test
        # P(X >= k) where X ~ Hypergeometric(N, K, n)
        p_value = hypergeom.sf(k - 1, N, K, n)  # sf gives P(X >= k)

        # Expected number of interactions (if random)
        expected = n * K / N

        # Enrichment factor
        enrichment = k / expected if expected > 0 else 0

        # 95% confidence interval for enrichment
        # Using Wilson score interval for proportions
        p_obs = k / n
        p_exp = K / N
        z = 1.96  # 95% confidence

        # Wilson CI for observed proportion
        denom = 1 + z**2 / n
        center = (p_obs + z**2 / (2 * n)) / denom
        width = z * np.sqrt(p_obs * (1 - p_obs) / n + z**2 / (4 * n**2)) / denom

        ci_lower = max(0, center - width)
        ci_upper = min(1, center + width)

        # Convert to enrichment CI
        enrichment_ci_lower = ci_lower / p_exp if p_exp > 0 else 0
        enrichment_ci_upper = ci_upper / p_exp if p_exp > 0 else 0

        enrichment_results.append(
            {
                "percentile": percentile,
                "sample_size": n,
                "observed_interactions": k,
                "expected_interactions": expected,
                "enrichment_factor": enrichment,
                "enrichment_ci_lower": enrichment_ci_lower,
                "enrichment_ci_upper": enrichment_ci_upper,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        )

        # Print results
        significance = (
            "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        )
        print(f"Top {percentile}% (n={n}):")
        print(f"  Observed interactions: {k}")
        print(f"  Expected interactions: {expected:.1f}")
        print(
            f"  Enrichment factor: {enrichment:.2f}x (95% CI: {enrichment_ci_lower:.2f}-{enrichment_ci_upper:.2f})"
        )
        print(f"  P-value: {p_value:.2e} {significance}")
        print()

    # Convert to DataFrame
    enrichment_df = pd.DataFrame(enrichment_results)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Enrichment factors with confidence intervals
    ax1.errorbar(
        enrichment_df["percentile"],
        enrichment_df["enrichment_factor"],
        yerr=[
            enrichment_df["enrichment_factor"] - enrichment_df["enrichment_ci_lower"],
            enrichment_df["enrichment_ci_upper"] - enrichment_df["enrichment_factor"],
        ],
        marker="o",
        capsize=5,
        capthick=2,
        linewidth=2,
    )

    ax1.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="No enrichment")
    ax1.set_xlabel("Top Percentile (%)")
    ax1.set_ylabel("Enrichment Factor")
    ax1.set_title("Interaction Enrichment by Percentile")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Highlight significant points
    sig_points = enrichment_df[enrichment_df["significant"]]
    if len(sig_points) > 0:
        ax1.scatter(
            sig_points["percentile"],
            sig_points["enrichment_factor"],
            color="red",
            s=100,
            alpha=0.7,
            zorder=5,
        )

    # Plot 2: P-values
    ax2.bar(
        enrichment_df["percentile"],
        -np.log10(enrichment_df["p_value"]),
        color=["red" if sig else "blue" for sig in enrichment_df["significant"]],
    )
    ax2.axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="p = 0.05")
    ax2.axhline(y=-np.log10(0.01), color="orange", linestyle="--", alpha=0.7, label="p = 0.01")
    ax2.set_xlabel("Top Percentile (%)")
    ax2.set_ylabel("-log10(P-value)")
    ax2.set_title("Statistical Significance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("=== ENRICHMENT SUMMARY ===")
    significant_tests = enrichment_df[enrichment_df["significant"]]
    if len(significant_tests) > 0:
        best_enrichment = significant_tests.loc[significant_tests["enrichment_factor"].idxmax()]
        print(f"Best significant enrichment:")
        print(
            f"  Top {best_enrichment['percentile']}%: {best_enrichment['enrichment_factor']:.2f}x enrichment"
        )
        print(f"  P-value: {best_enrichment['p_value']:.2e}")
        print(
            f"  95% CI: {best_enrichment['enrichment_ci_lower']:.2f}-{best_enrichment['enrichment_ci_upper']:.2f}"
        )
    else:
        print("No significant enrichment found at tested percentiles")

    return enrichment_df


# Example usage with your VIDA results
def example_enrichment_test():
    """
    Example of how to run the enrichment test with your VIDA results
    """

    # Assuming you have your VIDA results
    # vida_results = your_vida_dataframe
    # allele_interaction_indices = your_interaction_indices
    # allele_direct_indices = your_direct_indices

    print("Example usage:")
    print("enrichment_results = hypergeometric_enrichment_test(")
    print("    vida_results=final_metrics,")
    print("    allele_interaction_indices=allele_interaction_indices,")
    print("    allele_direct_indices=allele_direct_indices,")
    print("    metric_name='VIDA_score',")
    print("    top_percentiles=[5, 10, 15, 20, 25]")
    print(")")

    # You can also test individual components
    print("\n# Test individual components:")
    print(
        "clumpiness_enrichment = hypergeometric_enrichment_test(..., metric_name='avg_clumpiness_diff')"
    )
    print(
        "extreme_ratio_enrichment = hypergeometric_enrichment_test(..., metric_name='avg_extreme_ratio_diff')"
    )


# Advanced analysis function
def multiple_metric_enrichment_test(
    vida_results,
    allele_interaction_indices,
    allele_direct_indices,
    metrics=["VIDA_score", "avg_clumpiness_diff", "avg_extreme_ratio_diff"],
):
    """
    Test enrichment for multiple metrics simultaneously
    """

    print("=== MULTIPLE METRIC ENRICHMENT TEST ===")
    print("-" * 50)

    all_results = {}

    for metric in metrics:
        if metric in vida_results.columns:
            print(f"\nTesting {metric}:")
            print("=" * 30)

            results = hypergeometric_enrichment_test(
                vida_results,
                allele_interaction_indices,
                allele_direct_indices,
                metric_name=metric,
                top_percentiles=[5, 10, 20],
            )

            all_results[metric] = results

    # Compare best enrichments
    print("\n=== COMPARISON SUMMARY ===")
    for metric, results in all_results.items():
        if results is not None:
            best = results.loc[results["enrichment_factor"].idxmax()]
            print(
                f"{metric}: {best['enrichment_factor']:.2f}x at top {best['percentile']}% (p={best['p_value']:.2e})"
            )

    return all_results
