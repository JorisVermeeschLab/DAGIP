from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import KernelPCA

from dagip.nipt.binning import ChromosomeBounds


MARKERS = ['x', 'o', 's', 'P', 'D', 'h', '*', 'p', '8']


def plot_ot_plan_degrees(ax, gamma: np.ndarray, eps: float = 1e-7):
    unique, counts = np.unique(np.sum(gamma > eps, axis=0), return_counts=True)
    ax.bar(
        unique - 0.15, counts, color='darkgoldenrod', width=0.3,
        label='Source domain'
    )
    unique2, counts = np.unique(np.sum(gamma > eps, axis=1), return_counts=True)
    ax.bar(
        unique2 + 0.15, counts, color='darkcyan', width=0.3,
        label='Target domain'
    )
    ax.legend()
    ax.set_xticks(range(1, max(max(unique), max(unique2)) + 1), range(1, max(max(unique), max(unique2)) + 1))
    ax.set_xlabel(r'#Similar points in source domain')
    ax.set_ylabel(r'#Points in target domain')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def scatter_plot(X: np.ndarray, y: np.ndarray, d: np.ndarray):
    y = y.astype(float)
    coords = KernelPCA().fit_transform(X)
    for domain in np.unique(d):
        mask = (d == domain)
        marker = MARKERS[domain % len(MARKERS)]
        plt.scatter(coords[mask, 0], coords[mask, 1], alpha=0.5, c=y[mask], marker=marker)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')


def plot_gc_bias(X1_adapted: np.ndarray, X1: np.ndarray, gc_content: np.ndarray):
    x1_adapted = np.mean(X1_adapted, axis=0)
    x1 = np.mean(X1, axis=0)
    chr_bounds = ChromosomeBounds.get_10kb()
    ys = [np.mean(X1_adapted[:, i:i+1]) - np.mean(X1[:, i:i+1]) for i in range(22)]
    cmap = plt.get_cmap('bwr')
    mask = (np.random.rand(len(gc_content)) < 0.1)
    mask = np.logical_and(mask, np.logical_and(gc_content > 0.3, gc_content < 0.7))
    fig, ax = plt.subplots(figsize=(6.05, 5.5))
    mappable = ax.scatter(x1[mask], x1_adapted[mask], c=gc_content[mask], s=2, cmap=cmap)
    ax.plot([0, 1.2], [0, 1.2], color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Average normalised read counts')
    ax.set_ylabel('Average corrected read counts')
    fig.colorbar(mappable)
    plt.tight_layout()


"""
fig, ax = plt.subplots(2, 1, figsize=(4, 8))
coords = TSNE().fit_transform(ChromosomeBounds.bin_from_10kb_to_1mb(X))
ax[0].scatter(coords[idx1, 0], coords[idx1, 1], edgecolors='mediumseagreen', marker='s', c='None')
ax[0].scatter(coords[idx2, 0], coords[idx2, 1], edgecolors='royalblue', marker='o', c='None')
idx3 = np.logical_and(d == 0, y == 1)
ax[0].scatter(coords[idx3, 0], coords[idx3, 1], color='orchid', marker='x')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_xlabel('First component')
ax[0].set_ylabel('Second component')
coords = TSNE().fit_transform(ChromosomeBounds.bin_from_10kb_to_1mb(X_adapted))
ax[1].scatter(coords[idx1, 0], coords[idx1, 1], label='Controls (TruSeq ChIP)', edgecolors='mediumseagreen', marker='s', c='None')
ax[1].scatter(coords[idx2, 0], coords[idx2, 1], label='Controls (KAPA HyperPrep)', edgecolors='royalblue', marker='o', c='None')
idx3 = np.logical_and(d == 0, y == 1)
ax[1].scatter(coords[idx3, 0], coords[idx3, 1], label='Haematological cancer', color='orchid', marker='x')
ax[1].legend(prop={'size': 7})
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].set_xlabel('First component')
ax[1].set_ylabel('Second component')
plt.tight_layout()
plt.savefig('tsne.png', dpi=200)
"""