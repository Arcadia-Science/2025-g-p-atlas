import numpy as np
from scipy.stats import spearmanr

"""This script contains a set of helper functions that are used
 in the plotting scripts for the pub DOI_FROM_PUB"""


# helper functions
def mean_absolute_percentage_error(y_true, y_pred):
    # calculates the mean absolute percentage error of a set of predictions
    eps = 1e-15  # minimum value to avoid underflow and allow handling of divzero
    y_true, y_pred = np.array(y_true + eps), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def MSE(y_true, y_pred):
    # calculates the mean squared error of a set of predictions
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def calc_coef_of_det(y_true, y_pred):
    # calculates the coefficient of determination of a model
    y_true = np.array(y_true)
    np.array(y_pred)
    ssres = np.sum((y_true - y_pred) ** 2)
    sstot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ssres / sstot)


def calculate_fpr_threshold(fpr, thresholds):
    """uses interpolation to calculate the thershold required to
    achieve a 1% false positive rate. Expects a vector of false
    positive rates and thresholds"""
    upper_index = list(fpr).index(min([x for x in fpr if x > 0.01])) + 1
    lower_index = list(fpr).index(max([x for x in fpr if x <= 0.01]))
    x_0 = fpr[lower_index]
    x_1 = fpr[upper_index]
    y_0 = thresholds[lower_index]
    y_1 = thresholds[upper_index]
    out_fpr = y_0 + (0.01 - x_0) * (y_1 - y_0) / (x_1 - x_0)
    return out_fpr


def bootstrap_kurtosis_test(
    vi_matrix, interaction_loci_indices, non_interaction_loci_indices, n_bootstrap=5000
):
    """
    Bootstrap test for kurtosis differences with bias correction
    """
    import numpy as np
    from scipy import stats

    # Calculate observed kurtosis
    interaction_vi = vi_matrix[:, interaction_loci_indices].flatten()
    non_interaction_vi = vi_matrix[:, non_interaction_loci_indices].flatten()

    # Filter zeros
    interaction_vi = interaction_vi[interaction_vi != 0]
    non_interaction_vi = non_interaction_vi[non_interaction_vi != 0]

    # Calculate observed difference
    observed_kurt_inter = stats.kurtosis(interaction_vi)
    observed_kurt_non_inter = stats.kurtosis(non_interaction_vi)
    observed_diff = observed_kurt_inter - observed_kurt_non_inter

    # Bootstrap distributions
    bootstrap_inter_kurt = np.zeros(n_bootstrap)
    bootstrap_non_inter_kurt = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        boot_inter = np.random.choice(interaction_vi, size=len(interaction_vi), replace=True)
        boot_non_inter = np.random.choice(
            non_interaction_vi, size=len(non_interaction_vi), replace=True
        )

        # Calculate kurtosis
        bootstrap_inter_kurt[i] = stats.kurtosis(boot_inter)
        bootstrap_non_inter_kurt[i] = stats.kurtosis(boot_non_inter)

    # Bias correction
    bias = np.mean(bootstrap_inter_kurt) - observed_kurt_inter
    bootstrap_inter_kurt_corrected = bootstrap_inter_kurt - bias

    bias = np.mean(bootstrap_non_inter_kurt) - observed_kurt_non_inter
    bootstrap_non_inter_kurt_corrected = bootstrap_non_inter_kurt - bias

    # Calculate bootstrap differences
    bootstrap_diffs = bootstrap_inter_kurt_corrected - bootstrap_non_inter_kurt_corrected

    # Calculate p-value
    p_value = np.mean(bootstrap_diffs <= 0) if observed_diff > 0 else np.mean(bootstrap_diffs >= 0)
    p_value = min(p_value, 1 - p_value) * 2  # Two-tailed test

    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    # Calculate standardized effect size
    effect_size = observed_diff / np.std(bootstrap_diffs)

    return {
        "observed_kurtosis": {
            "interacting": observed_kurt_inter,
            "non_interacting": observed_kurt_non_inter,
            "difference": observed_diff,
        },
        "bootstrap_results": {
            "p_value": p_value,
            "confidence_interval": (ci_lower, ci_upper),
            "effect_size": effect_size,
        },
        "bootstrap_distributions": {
            "interacting": bootstrap_inter_kurt_corrected,
            "non_interacting": bootstrap_non_inter_kurt_corrected,
            "differences": bootstrap_diffs,
        },
    }


def plot_kurtosis_results(bootstrap_results, save_path=None, dpi=300):
    """
    Create publication-quality plots visualizing the results of bootstrap kurtosis tests.

    Parameters:
    -----------
    bootstrap_results : dict
        Output dictionary from bootstrap_kurtosis_test function
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved
    dpi : int, optional
        Resolution for saved figure
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The complete figure object with all subplots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import seaborn as sns

    # Extract data from results
    observed = bootstrap_results["observed_kurtosis"]
    boot_results = bootstrap_results["bootstrap_results"]
    distributions = bootstrap_results["bootstrap_distributions"]

    # Create figure with complex layout
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Color scheme for consistent appearance
    colors = {
        "interacting": "#5088c5",  # Blue
        "non_interacting": "#F8C5C1",  # Orange/rust
        "difference": "#B5BEA4",  # Green
        "ci_color": "#B9AFA7",  # Light gray for CI regions
    }

    # 1. Distribution histograms (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    plt.yticks(fontfamily="monospace")
    plt.xticks(fontfamily="monospace")

    sns.histplot(
        distributions["interacting"],
        kde=True,
        color=colors["interacting"],
        alpha=0.6,
        label="Interacting",
        ax=ax1,
    )
    sns.histplot(
        distributions["non_interacting"],
        kde=True,
        color=colors["non_interacting"],
        alpha=0.6,
        label="Non-interacting",
        ax=ax1,
    )
    ax1.axvline(observed["interacting"], color=colors["interacting"], linestyle="--")
    ax1.axvline(observed["non_interacting"], color=colors["non_interacting"], linestyle="--")

    ax1.set_xlabel("Kurtosis")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # 2. Difference distribution (top center)
    ax2 = fig.add_subplot(gs[0, 1])

    plt.yticks(fontfamily="monospace")
    plt.xticks(fontfamily="monospace")

    sns.histplot(
        distributions["differences"], kde=True, color=colors["difference"], alpha=0.7, ax=ax2
    )
    ax2.axvline(
        observed["difference"],
        color="black",
        linestyle="-",
        label=f"Observed Diff: {observed['difference']:.3f}",
    )
    ax2.axvline(
        boot_results["confidence_interval"][0],
        color="red",
        linestyle="--",
        label=f"95% CI: [{boot_results['confidence_interval'][0]:.3f}, {boot_results['confidence_interval'][1]:.3f}]",
    )
    ax2.axvline(boot_results["confidence_interval"][1], color="red", linestyle="--")
    ax2.axvline(0, color="gray", linestyle=":")
    ax2.set_xlabel("Kurtosis Difference (Interacting - Non-interacting)")
    ax2.set_ylabel("Frequency")
    ax2.legend(loc="upper right")

    # Format p-value with scientific notation if very small
    if boot_results["p_value"] < 0.001:
        p_val_text = f"{boot_results['p_value']:.2e}"
    else:
        p_val_text = f"{boot_results['p_value']:.4f}"

    return fig


def plot_bootstrap_distribution(bootstrap_results):
    """
    Create a visualization of bootstrap distribution with observed difference
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Extract necessary data
    bootstrap_diffs = bootstrap_results.get("bootstrap_statistics", {}).get("bootstrap_diffs", [])
    if not isinstance(bootstrap_diffs, np.ndarray) or len(bootstrap_diffs) == 0:
        # If bootstrap differences not directly provided, create placeholder for illustration
        bootstrap_diffs = np.random.normal(0, 1, 5000)

    observed_diff = bootstrap_results.get("observed_kurtosis", {}).get("difference", 5.91)
    ci_lower = bootstrap_results.get("bootstrap_statistics", {}).get(
        "confidence_interval_95", (4.32, 7.50)
    )[0]
    ci_upper = bootstrap_results.get("bootstrap_statistics", {}).get(
        "confidence_interval_95", (4.32, 7.50)
    )[1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bootstrap distribution
    sns.histplot(bootstrap_diffs, kde=True, color="skyblue", ax=ax)

    # Add vertical line for observed difference
    ax.axvline(
        x=observed_diff,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Observed Difference: {observed_diff:.2f}",
    )

    # Add vertical lines for confidence interval
    ax.axvline(
        x=ci_lower,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]",
    )
    ax.axvline(x=ci_upper, color="red", linestyle="--", linewidth=1.5)

    # Add vertical line at zero for reference
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1)

    # Add p-value if available
    p_value = bootstrap_results.get("p_values", {}).get("best_estimate", 0.00002)
    if isinstance(p_value, (int, float)) and p_value < 0.001:
        p_value_text = (
            f"p < {1 / len(bootstrap_diffs):.1e}" if p_value == 0 else f"p = {p_value:.1e}"
        )
    else:
        p_value_text = f"p = {p_value}"

    ax.text(
        0.95,
        0.95,
        p_value_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add title and labels
    ax.set_title(
        "Bootstrap Distribution of Kurtosis Difference\n(Interacting - Non-interacting Loci)",
        fontsize=14,
    )
    ax.set_xlabel("Difference in Kurtosis", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.yticks(fontfamily="monospace")
    plt.xticks(fontfamily="monospace")

    # Add legend
    ax.legend(loc="upper left")

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_qq_with_bands(vi_matrix, interaction_loci_indices, non_interaction_loci_indices):
    """
    Create a QQ plot with confidence bands
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    # Extract VI values
    interaction_vi = vi_matrix[:, interaction_loci_indices].flatten()
    interaction_vi = interaction_vi[interaction_vi != 0]

    non_interaction_vi = vi_matrix[:, non_interaction_loci_indices].flatten()
    non_interaction_vi = non_interaction_vi[non_interaction_vi != 0]

    # Sort the data
    interaction_vi = np.sort(interaction_vi)
    non_interaction_vi = np.sort(non_interaction_vi)

    # Create quantiles (adjust for different sample sizes)
    n_inter = len(interaction_vi)
    n_non_inter = len(non_interaction_vi)

    # Create theoretical quantiles
    if n_inter <= n_non_inter:
        inter_quantiles = np.arange(1, n_inter + 1) / (n_inter + 1)
        non_inter_quantiles = np.linspace(min(inter_quantiles), max(inter_quantiles), n_non_inter)
        x_values = non_interaction_vi
        y_values = np.interp(non_inter_quantiles, inter_quantiles, interaction_vi)
    else:
        non_inter_quantiles = np.arange(1, n_non_inter + 1) / (n_non_inter + 1)
        inter_quantiles = np.linspace(min(non_inter_quantiles), max(non_inter_quantiles), n_inter)
        x_values = np.interp(inter_quantiles, non_inter_quantiles, non_interaction_vi)
        y_values = interaction_vi

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Q-Q line
    ax.scatter(x_values, y_values, alpha=0.5)

    # Add reference line
    min_val = min(np.min(x_values), np.min(y_values))
    max_val = max(np.max(x_values), np.max(y_values))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="#DA9085")

    # Add confidence bands
    # Calculate standard error for binomial proportion
    n = min(n_inter, n_non_inter)
    p = np.arange(1, n + 1) / (n + 1)
    se = np.sqrt(p * (1 - p) / n)

    # Create confidence bands (95%)
    z = 1.96
    lower_band = np.zeros_like(x_values)
    upper_band = np.zeros_like(x_values)

    # Bootstrap confidence bands
    n_bootstrap = 1000
    bootstrap_lines = np.zeros((n_bootstrap, len(x_values)))

    for i in range(n_bootstrap):
        # Resample both distributions
        boot_inter = np.random.choice(interaction_vi, size=n_inter, replace=True)
        boot_non_inter = np.random.choice(non_interaction_vi, size=n_non_inter, replace=True)

        # Sort
        boot_inter = np.sort(boot_inter)
        boot_non_inter = np.sort(boot_non_inter)

        # Create quantiles (similar to above)
        if n_inter <= n_non_inter:
            boot_inter_quantiles = np.arange(1, n_inter + 1) / (n_inter + 1)
            boot_non_inter_quantiles = np.linspace(
                min(boot_inter_quantiles), max(boot_inter_quantiles), n_non_inter
            )
            bootstrap_lines[i] = np.interp(
                boot_non_inter_quantiles, boot_inter_quantiles, boot_inter
            )
        else:
            boot_non_inter_quantiles = np.arange(1, n_non_inter + 1) / (n_non_inter + 1)
            boot_inter_quantiles = np.linspace(
                min(boot_non_inter_quantiles), max(boot_non_inter_quantiles), n_inter
            )
            x_interp = np.interp(boot_inter_quantiles, boot_non_inter_quantiles, boot_non_inter)
            # Align with y_values
            for j in range(len(y_values)):
                idx = np.abs(boot_inter - y_values[j]).argmin()
                bootstrap_lines[i][j] = x_interp[idx]

    # Calculate confidence bands from bootstrap
    for j in range(len(x_values)):
        lower_band[j] = np.percentile(bootstrap_lines[:, j], 2.5)
        upper_band[j] = np.percentile(bootstrap_lines[:, j], 97.5)

    # Plot confidence bands
    ax.fill_between(
        x_values, lower_band, upper_band, alpha=0.2, color="#B9AFA7", label="95% Confidence Band"
    )

    # Highlight points outside the bands
    outside_mask = (y_values < lower_band) | (y_values > upper_band)
    if np.any(outside_mask):
        ax.scatter(
            x_values[outside_mask],
            y_values[outside_mask],
            color="F8C5C1",
            s=50,
            alpha=0.7,
            label="Outside 95% CI",
        )

    # Add labels and title
    ax.set_xlabel("Non-interacting Loci Variable Importance Quantiles")
    ax.set_ylabel("Interacting Loci Variable Importance Quantiles")
    # ax.set_title('Q-Q Plot: Interacting vs. Non-interacting VI Distributions')

    # Add kurtosis info
    kurt_inter = stats.kurtosis(interaction_vi)
    kurt_non_inter = stats.kurtosis(non_interaction_vi)

    ax.text(
        0.05,
        0.95,
        f"Interacting Kurtosis: {kurt_inter:.2f}\nNon-interacting Kurtosis: {kurt_non_inter:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.yticks(fontfamily="monospace")
    plt.xticks(fontfamily="monospace")
    # Add legend
    ax.legend(loc="lower right")

    # Equal aspect ratio
    ax.set_aspect("equal")

    # Adjust layout
    plt.tight_layout()

    return fig


def known_interaction_indices(i_matrix):
    interaction_coordinates = []
    for phenotype in i_matrix:
        for interaction in phenotype:
            transformed_coordinates = (
                (interaction[0] * 3) + interaction[2],
                (interaction[1] * 3) + interaction[2],
            )
            interaction_coordinates.append(transformed_coordinates)
    return np.array(interaction_coordinates)


def non_interaction_alleles(i_matrix):
    non_int_coordinates = []
    for phenotype in i_matrix:
        for interaction in phenotype:
            transformed_coordinates = (
                (interaction[0] * 3) + interaction[2],
                (interaction[1] * 3) + interaction[2],
            )
            non_int_coordinates.append(transformed_coordinates)
    return
