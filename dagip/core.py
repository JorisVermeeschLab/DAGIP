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
from typing import Tuple, List, Union

import numpy as np
import ot
import scipy.stats
import torch
import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

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
    distances = cdist(X1, X2)
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
    distances = cdist(X1, X2)
    distances /= np.outer(np.mean(distances, axis=1), np.mean(distances, axis=0))
    return distances


def ot_mapping(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    # distances = cdist(X1, X2)
    assert not np.any(np.isnan(X1))
    assert not np.any(np.isnan(X2))
    distances = pairwise_distances(X1, X2)
    assert not np.any(np.isnan(distances ** p))
    assert not np.any(np.isinf(distances ** p))

    gamma = ot.emd(a, b, distances ** p)
    gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
    return np.dot(gamma, X2)


def quantile_transform(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    inv_idx1 = np.argsort(np.argsort(x1))
    idx2 = np.argsort(x2)
    return x2[idx2[inv_idx1]]


def define_target_(
        x1: np.ndarray,
        x2_prime: np.ndarray
) -> np.ndarray:
    out = np.zeros(len(x2_prime))
    bounds = ChromosomeBounds.get_bounds(len(x1))
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        out[start:end] = quantile_transform(
            x1[start:end],
            x2_prime[start:end]
        )
    return out


def define_target(
        X1: np.ndarray,
        X2_prime: np.ndarray
) -> np.ndarray:
    out = np.zeros(X2_prime.shape)
    for i in range(len(X2_prime)):
        out[i, :] = define_target_(X1[i, :], X2_prime[i, :])
    return out


def follow_same_distribution(X1: np.ndarray, X2: np.ndarray) -> bool:
    p_values = compute_p_values(X1, X2)
    print(np.median(p_values))
    return np.median(p_values) >= 0.5  # TODO


def compute_p_values(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return np.asarray([scipy.stats.ranksums(X1[:, k], X2[:, k]).pvalue for k in range(X1.shape[1])])


def standardization(X: torch.Tensor, mu: torch.Tensor, ddof: int = 0, eps: float = 1e-5) -> torch.Tensor:
    n = len(X)
    mu = mu.unsqueeze(0)
    var = torch.sum(torch.square(X - mu), dim=0) / (n - ddof)
    var = torch.clamp(var.unsqueeze(0), eps)
    std = torch.sqrt(var)
    return (X - mu) / std


def ot_da(
        folder: str,
        X1: np.ndarray,
        X2: np.ndarray,
        side_info: np.ndarray,
        max_n_iter: int = 200,  # 1000
        convergence_threshold: float = 0.5,
        reg_rate: float = 1.0
) -> np.ndarray:

    gc_content = torch.FloatTensor((np.round(side_info[:, 0] * 1000).astype(int) // 10).astype(float))

    with torch.no_grad():
        X1_corrected = diff_gc_correction(torch.FloatTensor(X1), gc_content)
        target_diffs = standardization(X1_corrected, torch.median(X1_corrected, dim=0).values)

    check_every = 1

    X1_adapted = torch.nn.Parameter(torch.FloatTensor(np.copy(X1)))

    X1 = torch.FloatTensor(X1)
    optimizer = torch.optim.Adam([X1_adapted], lr=1e-3)  # 1e-3

    # Determine convergence criterion
    p_values = compute_p_values(X1.cpu().data.numpy(), X2)
    median_p_value = np.median(p_values)
    log_(f'Median p-value: {median_p_value}')
    if median_p_value >= convergence_threshold:
        convergence_threshold = 0.5 + 0.5 * median_p_value

    speed = 1.
    last = np.inf

    losses = []
    median_p_values = []
    for iteration in tqdm.tqdm(range(max_n_iter), desc='OT'):  # TODO

        optimizer.zero_grad()

        X1_corrected = diff_gc_correction(X1_adapted, gc_content)

        X2_prime = torch.FloatTensor(ot_mapping(X1_corrected.cpu().data.numpy(), X2))

        diffs = standardization(
            X1_corrected,
            torch.median(torch.cat((X1_corrected, torch.FloatTensor(X2)), dim=0), dim=0).values
        )

        loss = torch.mean((X1_corrected - X2_prime) ** 2)
        reg = reg_rate * torch.mean((diffs - target_diffs) ** 2)
        print(loss.item(), reg.item())
        (speed * (loss + reg)).backward()
        optimizer.step()
        with torch.no_grad():
            X1_adapted.data = torch.clamp(X1_adapted.data, 0)
            X1_adapted.data = X1_adapted.data / torch.median(X1_adapted.data, dim=1).values.unsqueeze(1)

        # Update speed
        if (loss + reg).item() < last:
            speed *= 1.1
        else:
            speed *= 0.9
        speed = float(np.clip(speed, 0.1, 10))
        last = (loss + reg).item()

        losses.append([loss.item(), reg.item()])
        # print(last, losses[-1])
        # print((loss + reg).item(), loss.item(), reg.item(), len(X1), len(X2))

        # Check convergence
        if (iteration % check_every == 0):
            p_values = compute_p_values(X1_corrected.cpu().data.numpy(), X2)
            median_p_values.append(np.median(p_values))
            # print('threshold', np.median(p_values), convergence_threshold)
            if np.median(p_values) >= convergence_threshold:
                break

    losses = np.asarray(losses)

    X1_corrected = diff_gc_correction(X1_adapted, gc_content)
    X1_corrected = X1_corrected.cpu().data.numpy()

    _, gamma = wasserstein_distance(X1_corrected, X2, return_plan=True)

    # Save figures
    if not os.path.isdir(folder):
        os.makedirs(folder)
    p_values = compute_p_values(X1_corrected, X2)
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

    return X1_corrected
