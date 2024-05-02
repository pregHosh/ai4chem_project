import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import pairwise_distances


# Model analysis----------------------------------------------------------
def corr_plot(true_values, predicted_values, property_name="E", filename=None):
    r2 = r2_score(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    density = gaussian_kde(predicted_values)(predicted_values)

    # plot------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc("font", size=18)

    ax.scatter(true_values, predicted_values, c=density, cmap="magma", alpha=1, s=50)
    cbar = plt.colorbar(
        ax.scatter(
            true_values, predicted_values, c=density, cmap="magma", alpha=1, s=15
        )
    )
    cbar.set_label("Probability Density")

    ax.set_xlabel(f"Computed {property_name}", fontsize=18)
    ax.set_ylabel(f"Predicted {property_name}", fontsize=18)
    ax.tick_params(axis="both", which="both")
    ax.set_title(f"R$^2$: {r2:.2f}\n MAE: {mae:.2f}", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Add a diagonal reference line
    ax.plot(
        [min(true_values), max(true_values)],
        [min(true_values), max(true_values)],
        color="black",
        linestyle="--",
        linewidth=2,
    )
    if filename:
        fig.savefig(filename, dpi=400)
    plt.show()


def dist_combo_plot(
    true_values, predicted_values, property_name="E ", filename=None
):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams["figure.autolayout"] = True
    sns.set_style("ticks")
    sns.set_context("talk")

    sns.kdeplot(true_values, label="True Values", color="blue", fill=True, linewidth=3)
    sns.kdeplot(
        predicted_values, label="Predicted Values", color="red", fill=True, linewidth=3
    )

    ax.set_xlabel(property_name, fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    leg = ax.legend(fontsize=18, loc="upper left")

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().tick_params(axis="x", direction="inout", length=8)
    plt.gca().tick_params(axis="y", direction="inout", length=8)

    sns.despine()
    if filename:
        plt.savefig(filename, dpi=400, bbox_inches="tight")
    plt.show()


def error_dist_plot(y_true, y_pred, output_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("font", size=18)

    errors = y_pred - y_true

    # Define bin edges to center the histogram at even numbers
    bin_edges = np.arange(-10, 9, 0.5)

    ax.hist(
        errors,
        bins=bin_edges,
        color="#1266A4",
        edgecolor="#1266A4",
        alpha=0.8,
        align="mid",
        rwidth=0.8,
    )
    plt.xticks(np.arange(-10, 9, 2))
    plt.xlim(-10, 9)
    plt.xlabel("Deviation")
    plt.ylabel("Frequency")
    plt.show()
    fig.savefig(output_name, dpi=400)


# Feature analysis--------------------------------------------------------


def dissim_plot(reprs, y_true, truncate=False, subset_size=0.1, job_name=None):
    all_indices_train = np.arange(len(y_true))
    size_train = int(np.floor(len(y_true) * subset_size))
    perform = np.random.choice(all_indices_train, size=size_train, replace=False)
    if truncate:
        reprs = reprs[perform]
        y_true = y_true[perform]

    dmat_bond_rep = pairwise_distances(reprs)
    dmat_target = np.subtract.outer(y_true, y_true)

    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=18)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    hist, xedges, yedges = np.histogram2d(
        1
        / dmat_bond_rep.mean()
        * dmat_bond_rep[np.triu_indices_from(dmat_bond_rep, k=1)],
        dmat_target[np.triu_indices_from(dmat_target, k=1)],
        bins=100,
        range=[[0, 1.5], [-20, 20]],
    )
    ax.imshow(
        hist.T,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        extent=[0, 5, -20, 20],
        cmap="ocean_r",
    )
    ax.set_xlabel("Euclidean distance between \n representations in the training set")
    ax.set_ylabel("Difference between forward \n")

    plt.tight_layout(pad=2)

    cax = ax.imshow(
        hist.T,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        extent=[0, 100, -25, 25],
        cmap="ocean_r",
    )
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.16, 0.02, 0.78])
    cbar = fig.colorbar(cax, cax=cbar_ax)
    cbar.set_ticks([])
    cbar.set_label("Probability density")
    fig.savefig(f"diss_plot_{job_name}.png", dpi=400)


def tSNE_dim_reduct(
    reprs,
    target_property=None,
    perplexity=30,
    truncate=False,
    subset_size=0.1,
    job_name=None,
):
    all_indices_train = np.arange(reprs.shape[0])
    size_train = int(np.floor(reprs.shape[0] * subset_size))
    perform = np.random.choice(all_indices_train, size=size_train, replace=False)
    if truncate:
        reprs = reprs[perform]
        if target_property is not None:
            target_property = target_property[perform]

    tsne = TSNE(n_components=2, perplexity=perplexity, n_jobs=-1, init="pca")
    embedding = tsne.fit_transform(reprs)

    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=18)
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(1, 1, 1)
    if target_property is None:
        plot = ax.scatter(embedding[:, 0], embedding[:, 1], c="black", s=10)
    else:
        plot = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=target_property,
            cmap="jet",
            vmin=np.round(target_property.min()),
            vmax=np.round(target_property.max()),
        )
        cbar = fig.colorbar(plot, ax=ax)
        cbar.set_label("BDE (kcal/mol) ")
    plt.savefig(f"tSNE_BB_{job_name}.png", dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Analyse the performance of the model by plotting the correlation plot, distribution plot, and error distribution plot.
        """
    )

    parser.add_argument(
        "-d",
        "--d",
        "--dir",
        dest="working_dir",
        type=str,
        default="predictive",
        help="Directory containing the model output",
    )
    parser.add_argument(
        "-tn",
        "--target_names",
        dest="target_names",
        nargs="+",
        type=str,
        default=["FEPA", "FEHA"],
        help="List of target names (default=[FEPA, FEHA])",
    )

    args = parser.parse_args()
    wdir = args.working_dir
    target_names = args.target_names
    y_true = np.load(f"{wdir}/y_trues.npy")
    y_pred = np.load(f"{wdir}/y_preds.npy")

    nan_idxs = ~np.isnan(y_true).any(axis=1)
    y_true = y_true[nan_idxs]
    y_pred = y_pred[nan_idxs]

    assert (
        len(target_names) == y_true.shape[1]
    ), "Number of specified target names does not match the number of targets in the model output"
    assert (
        y_true.shape[1] == y_pred.shape[1]
    ), "Number of targets in the model output does not match the number of targets in the true values"
    assert (
        y_true.shape[0] == y_pred.shape[0]
    ), "Number of samples in the model output does not match the number of samples in the true values"

    for i, target_name in enumerate(target_names):
        print("Plotting for ", target_name)

        corr_plot(
            y_true[:, i], y_pred[:, i], target_name, f"corr_plot_{target_name}.png"
        )
        dist_combo_plot(
            y_true[:, i],
            y_pred[:, i],
            target_name,
            f"dist_combo_plot_{target_name}.png",
        )
        error_dist_plot(
            y_true[:, i], y_pred[:, i], f"error_dist_plot_{target_name}.png"
        )

