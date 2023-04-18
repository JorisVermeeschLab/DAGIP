# -*- coding: utf-8 -*-
#
#  core.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import os.path
import random
from typing import Tuple, List, Union

import numpy as np
import ot
import scipy.stats
import torch
import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA

from dagip.nipt.binning import ChromosomeBounds
from dagip.nn.gc_correction import diff_gc_correction
from dagip.utils import log_


def wasserstein_distance(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2,
        return_plan: bool = False
) -> Union[np.ndarray, Tuple[float, np.ndarray]]:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    distances = cdist(X1, X2, metric='cityblock')
    gamma = ot.emd(a, b, distances ** p)
    distance = np.sum(gamma * distances) ** (1. / p)
    if return_plan:
        return distance, gamma
    else:
        return distance


def pairwise_distances(
        X1: np.ndarray,
        X2: np.ndarray,
) -> np.ndarray:
    distances = cdist(X1, X2, metric='cityblock')
    distances /= np.outer(np.mean(distances, axis=1), np.mean(distances, axis=0))
    return distances


def transport_plan(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    assert not np.any(np.isnan(X1))
    assert not np.any(np.isnan(X2))
    distances = pairwise_distances(X1, X2)
    assert not np.any(np.isnan(distances ** p))
    assert not np.any(np.isinf(distances ** p))

    return ot.emd(a, b, distances ** p)


def piecewise_transport_plans(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    bounds = ChromosomeBounds.get_bounds(X1.shape[1])
    gammas = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        distances = pairwise_distances(X1[:, start:end], X2[:, start:end])
        gamma = ot.emd(a, b, distances ** p)
        gammas.append(gamma)
    return np.asarray(gammas)


def wtest(X1, X2, n_runs=100) -> float:

    n, m = len(X1), len(X2)

    W = wasserstein_distance(X1, X2)
    values = list()
    X = np.concatenate([X1, X2], axis=0)
    for k in range(n_runs):
        np.random.shuffle(X)
        values.append(wasserstein_distance(X[:n], X[n:]))
    pvalue = np.mean(W < np.asarray(values))
    return float(pvalue)


def ot_mapping(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    gamma = transport_plan(X1, X2, p=p)
    gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
    return np.dot(gamma, X2)


def piecewise_ot_mapping(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    out = np.zeros(X1.shape)
    bounds = ChromosomeBounds.get_bounds(X1.shape[1])
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        distances = pairwise_distances(X1[:, start:end], X2[:, start:end])
        gamma = ot.emd(a, b, distances ** p)
        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
        out[:, start:end] = np.dot(gamma, X2[:, start:end])
    return out


def follow_same_distribution(X1: np.ndarray, X2: np.ndarray) -> bool:
    p_values = compute_p_values(X1, X2)
    return np.median(p_values) >= 0.5  # TODO


def compute_p_values(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    p_values = []
    for k in range(X1.shape[1]):
        # p_values.append(scipy.stats.anderson_ksamp([X1[:, k], X2[:, k]]).significance_level)
        try:
            p_values.append(scipy.stats.ks_2samp(X1[:, k], X2[:, k]).pvalue)
            # p_values.append(2 * scipy.stats.mannwhitneyu(X1[:, k], X2[:, k]).pvalue)
        except ValueError:
            pass
    return np.asarray(p_values)


def standardization(X: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return scale * (X - mu)


def ot_da(
        folder: str,
        X1: np.ndarray,
        X2: np.ndarray,
        side_info: np.ndarray,
        max_n_iter: int = 1000,  # 1000
        convergence_threshold: float = 0.5,
) -> np.ndarray:

    reg_rate = 1000

    scale = torch.FloatTensor(1. / np.maximum(np.std(X2, axis=0), 1e-3)[np.newaxis, :])

    gc_content = torch.FloatTensor((np.round(side_info[:, 0] * 1000).astype(int) // 10).astype(float))

    with torch.no_grad():
        X1_corrected = diff_gc_correction(torch.FloatTensor(X1), gc_content)
        target_diffs_1 = standardization(
            X1_corrected, torch.median(X1_corrected, dim=0).values.unsqueeze(0), scale
        )
        target_diffs_2 = standardization(
            torch.FloatTensor(X2), torch.median(torch.FloatTensor(X2), dim=0).values.unsqueeze(0), scale
        )

    X1_adapted = torch.nn.Parameter(torch.FloatTensor(np.copy(X1)))
    # X2_prime = torch.FloatTensor(ot_mapping((X1_corrected).cpu().data.numpy(), X2))
    X2_prime = torch.FloatTensor(piecewise_ot_mapping(X1_corrected.cpu().data.numpy(), X2))  # TODO

    optimizer = torch.optim.Adam([X1_adapted], lr=1e-3)  # 1e-3

    # Determine convergence criterion
    p_values = compute_p_values(X1, X2)
    median_p_value = np.median(p_values)
    log_(f'Median p-value: {median_p_value}')

    speed = 1.
    last = np.inf

    losses = []
    median_p_values = []
    for iteration in tqdm.tqdm(range(max_n_iter), desc='OT'):  # TODO

        optimizer.zero_grad()

        # GC correction
        X1_corrected = diff_gc_correction(X1_adapted, gc_content)

        # Define targets
        if iteration % 10 == 0:
            # X2_prime = torch.FloatTensor(ot_mapping(X1_corrected.cpu().data.numpy(), X2))
            X2_prime = torch.FloatTensor(piecewise_ot_mapping(X1_corrected.cpu().data.numpy(), X2))  # TODO

        # Compensate the reduction of variance caused by the OT mapping
        factor = torch.mean(torch.std(torch.FloatTensor(X2), dim=0)) / torch.mean(torch.std(X2_prime, dim=0))
        mu = torch.median(X2_prime, dim=0).values.unsqueeze(0)
        X2_prime = factor * (X2_prime - mu) + mu

        # Wasserstein distance
        loss = torch.mean((scale * (X1_corrected - X2_prime)) ** 2)

        # Regularization function
        mu = torch.median(torch.cat((X1_corrected, torch.FloatTensor(X2)), dim=0), dim=0).values.unsqueeze(0)
        diffs = standardization(X1_corrected, mu, scale)
        reg = 0.5 * torch.mean((diffs - target_diffs_1) ** 2)
        diffs = standardization(torch.FloatTensor(X2), mu, scale)
        reg = reg + 0.5 * torch.mean((diffs - target_diffs_2) ** 2)
        print(loss.item(), reg.item())
        total_loss = (speed * ((1 / (reg_rate + 1)) * loss + (reg_rate / (reg_rate + 1)) * reg))
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            X1_adapted.data = torch.clamp(X1_adapted.data, 0)
            X1_adapted.data = X1_adapted.data / torch.median(X1_adapted.data, dim=1).values.unsqueeze(1)

        # Update speed
        if total_loss.item() < last:
            speed *= 1.02
        else:
            speed *= 0.9
        speed = float(np.clip(speed, 0.1, 10))
        last = (loss + reg_rate * reg).item()

        losses.append([loss.item(), reg.item()])
        # print(last, losses[-1])
        # print((loss + reg).item(), loss.item(), reg.item(), len(X1), len(X2))

        # Check convergence
        #p_values = compute_p_values(X1_corrected.cpu().data.numpy(), X2)
        #median_p_values.append(np.median(p_values))
        median_p_values.append(wtest(X1_corrected.cpu().data.numpy(), X2))
        print('threshold', median_p_values[-1], convergence_threshold)
        if median_p_values[-1] >= convergence_threshold:
            break

        if (iteration > 20) and (iteration % 20 == 0):

            reg_rate *= 0.5

    losses = np.asarray(losses)

    X1_corrected = diff_gc_correction(X1_adapted, gc_content)
    X1_corrected = X1_corrected.cpu().data.numpy()

    _, gamma = wasserstein_distance(X1_corrected, X2, return_plan=True)

    # Save figures
    if not os.path.isdir(folder):
        os.makedirs(folder)
    p_values = compute_p_values(X1_corrected, X2)
    log_(f'Median p-value after correction: {np.median(p_values)}')
    plt.subplot(2, 1, 1)
    plt.violinplot(p_values, vert=False, showmeans=True, showextrema=True)
    plt.xlabel('Wilcoxon rank-sum p-value')
    plt.subplot(2, 1, 2)
    plt.violinplot(np.log(p_values), vert=False, showmeans=True, showextrema=True)
    plt.xlabel('Wilcoxon rank-sum log(p-value)')
    plt.savefig(os.path.join(folder, 'p-values-final.png'), dpi=300)
    plt.clf()
    plt.plot(p_values)
    plt.xlabel('Bin')
    plt.ylabel('Wilcoxon rank-sum p-value')
    plt.savefig(os.path.join(folder, 'p-values.png'), dpi=300)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(cdist(X1, X1))
    plt.subplot(1, 2, 2)
    plt.imshow(cdist(X1_corrected, X1_corrected))
    plt.savefig(os.path.join(folder, 'D11.png'), dpi=400)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(cdist(X1, X2))
    plt.subplot(1, 2, 2)
    plt.imshow(cdist(X1_corrected, X2))
    plt.savefig(os.path.join(folder, 'D12.png'), dpi=400)
    plt.clf()
    plt.plot(median_p_values)
    plt.xlabel('Iterations')
    plt.ylabel('Median p-value')
    plt.savefig(os.path.join(folder, 'median-p-values.png'), dpi=300)
    plt.clf()
    plt.plot(losses[:, 0], label='Loss')
    plt.plot(losses[:, 1], label='Reg')
    plt.plot(losses[:, 0] + losses[:, 1], label='Total')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss function')
    plt.savefig(os.path.join(folder, 'loss.png'), dpi=300)
    plt.clf()
    plt.imshow(gamma)
    plt.savefig(os.path.join(folder, 'transport-plan.png'), dpi=300)
    plt.clf()
    plt.subplot(1, 2, 1)
    pca = KernelPCA()
    pca.fit(X1)
    coords = pca.transform(X1)
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4)
    coords = pca.transform(X2)
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4)
    plt.subplot(1, 2, 2)
    pca = KernelPCA()
    pca.fit(X1_corrected)
    coords = pca.transform(X1_corrected)
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4)
    coords = pca.transform(X2)
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4)
    plt.savefig(os.path.join(folder, 'kpca.png'), dpi=300)
    plt.clf()

    return X1_corrected
