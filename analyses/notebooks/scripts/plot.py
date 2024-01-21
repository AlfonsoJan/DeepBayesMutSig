"""
Module to create plots.
Author: H.C. Donker
"""
from matplotlib import pyplot as plt
import numpy as np

from .dataset import COSMIC_WEIGHTS


def trinucleotide(x):
    return x[0] + x[2] + x[-1]


def sortable_by_substitution(x):
    return x[1:-1] + x[0] + x[-1]


def plot_cosmic_2nd_order_signatures_95ci(weights, ax=None, tau_t=3):
    """Plot the distribution over COSMIC v3.4 mutational signatures."""
    if ax is None:
        ax = plt.gca()
    n_signatures = COSMIC_WEIGHTS.shape[0]
    cosmic_names = COSMIC_WEIGHTS.index

    uniform_threshold = tau_t / n_signatures
    w_avg = np.mean(weights, axis=0)
    w_lower = np.quantile(weights, 0.025, axis=0)
    w_upper = np.quantile(weights, 0.975, axis=0)
    w_err = np.stack([w_avg - w_lower, w_upper - w_avg], axis=0)
    is_in_sign = w_avg > uniform_threshold

    # Plot weights and colour the bars exceeding the uniform threshold red.
    bar_color = map(lambda x: "tab:red" if x else "tab:blue", is_in_sign)
    ax.bar(cosmic_names, w_avg, color=tuple(bar_color), yerr=w_err)
    # Rotate the tick labels and set their alignment.
    ax.tick_params(axis="x", labelrotation=90)
    # Make the x-axis ticklabels of weights that exceed the uniform threshold bold.
    for tick, color in zip(ax.get_xticklabels(), is_in_sign):
        tick.set_fontweight("bold" if color else "normal")
    ylim = min([max(w_upper) * 1.1, 1.0])
    ax.set_xlim([-0.5, n_signatures - 0.5])
    ax.set_ylim([0.0, ylim])
    ax.set_xlabel("Mutational Signature from COSMIC v3.4")
    ax.set_ylabel("Probability")
    return ax


def spectrum_plot_with_95ci(weights, ax=None, label=None, alpha=1.0):
    w_avg = np.mean(weights, axis=0)
    w_lower = np.quantile(weights, 0.025, axis=0)
    w_upper = np.quantile(weights, 0.975, axis=0)
    w_err = np.stack([w_avg - w_lower, w_upper - w_avg], axis=0)

    if ax is None:
        ax = plt.gca()
    ylim = min([max(w_upper) * 1.1, 1.0])
    ax.set_xlim([-0.5, 95.5])
    ax.set_ylim([0.0, ylim])

    # Determine order of tri-nucleotide substitutions names, so that it is the same
    # letter change per block.
    x_sortable = list(map(sortable_by_substitution, COSMIC_WEIGHTS.columns))
    sortable_name_idx = sorted(
        zip(x_sortable, COSMIC_WEIGHTS.columns, range(len(x_sortable)))
    )
    x_sorted_trinuc = list(map(lambda x: x[1], sortable_name_idx))
    x_sorted_idx = np.fromiter(map(lambda x: x[2], sortable_name_idx), dtype=int)

    ax.tick_params(axis="x", labelrotation=90)
    colours = {
        "C→A": "tab:blue",
        "C→G": "k",
        "C→T": "tab:red",
        "T→A": "tab:orange",
        "T→C": "tab:green",
        "T→G": "tab:pink",
    }
    for i, (name, c) in enumerate(colours.items()):
        x_centre = 16 * i - 0.5 + 8
        ax.text(x=x_centre, y=ylim * 0.9, s=name, size=12, horizontalalignment="center")
        x_block = [16 * i - 0.5, 16 * (i + 1) - 0.5]
        ax.fill_between(x_block, 1.0, 0.0, facecolor=c, linewidth=0.0, alpha=0.2)
    ax.bar(
        x_sorted_trinuc,
        height=w_avg[x_sorted_idx],
        yerr=w_err[:, x_sorted_idx],
        label=label,
        alpha=alpha,
    )
    ax.set_xticklabels(list(map(trinucleotide, x_sorted_trinuc)), size=8)
    ax.set_xlabel("Trinucleotide context")
    ax.set_ylabel("Probability")
    return ax
