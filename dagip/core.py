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
from typing import Tuple, Union, Callable

import numpy as np
import ot
import scipy.stats
import torch
import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA

from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction import Identity
from dagip.retraction.base import Retraction
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


def default_u_test(x1: np.ndarray, x2: np.ndarray) -> float:
    return float(scipy.stats.ks_2samp(x1, x2).pvalue)


def compute_p_values(X1: np.ndarray, X2: np.ndarray, u_test: Callable) -> np.ndarray:
    p_values = []
    for k in range(X1.shape[1]):
        p_values.append(u_test(X1[:, k], X2[:, k]))
    return np.asarray(p_values)


def standardization(X: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return scale * (X - mu)


def ot_da(
        folder: str,
        X1: np.ndarray,
        X2: np.ndarray,
        ret: Retraction = Identity(),
        u_test: Callable = default_u_test,
        reg_rate: float = 1000.0,
        max_n_iter: int = 1000,  # 1000
        chromosome_wise: bool = False,
        convergence_threshold: float = 0.5,
        lr: float = 0.0005,
        verbose: bool = True
) -> np.ndarray:

    scale = torch.FloatTensor(1. / np.maximum(np.std(X2, axis=0), 1e-3)[np.newaxis, :])

    with torch.no_grad():
        X1_corrected = ret(X1)
        target_diffs_1 = standardization(
            X1_corrected, torch.median(X1_corrected, dim=0).values.unsqueeze(0), scale
        )
        target_diffs_2 = standardization(
            torch.FloatTensor(X2), torch.median(torch.FloatTensor(X2), dim=0).values.unsqueeze(0), scale
        )

    X1_adapted = torch.nn.Parameter(torch.FloatTensor(np.copy(X1)))
    if chromosome_wise:
        X2_prime = torch.FloatTensor(piecewise_ot_mapping(X1_corrected.cpu().data.numpy(), X2))
    else:
        X2_prime = torch.FloatTensor(ot_mapping((X1_corrected).cpu().data.numpy(), X2))

    optimizer = torch.optim.Adam([X1_adapted], lr=lr)

    # Determine convergence criterion
    p_values = compute_p_values(X1, X2, u_test)
    median_p_value = np.median(p_values)
    log_(f'Median p-value: {median_p_value}', verbose=verbose)

    speed = 1.
    last = np.inf

    losses = []
    median_p_values = []
    pbar = tqdm.tqdm(range(max_n_iter), desc='OT')
    for iteration in pbar:

        optimizer.zero_grad()

        # Retracting on the manifold
        X1_corrected = ret.f2(X1_adapted)

        # Define targets
        if iteration % 10 == 0:
            if chromosome_wise:
                X2_prime = torch.FloatTensor(piecewise_ot_mapping(X1_corrected.cpu().data.numpy(), X2))
            else:
                X2_prime = torch.FloatTensor(ot_mapping(X1_corrected.cpu().data.numpy(), X2))

        # Compensate the reduction of variance caused by the OT mapping
        factor = torch.mean(torch.std(torch.FloatTensor(X2), dim=0)) / torch.mean(torch.std(X2_prime, dim=0))
        mu = torch.mean(X2_prime, dim=0).unsqueeze(0)
        X2_prime = factor * (X2_prime - mu) + mu

        # Wasserstein distance
        loss = torch.mean((scale * (X1_corrected - X2_prime)) ** 2)

        # Regularization function
        mu = torch.median(torch.cat((X1_corrected, torch.FloatTensor(X2)), dim=0), dim=0).values.unsqueeze(0)
        diffs = standardization(X1_corrected, mu, scale)
        reg = 0.5 * torch.mean((diffs - target_diffs_1) ** 2)
        diffs = standardization(torch.FloatTensor(X2), mu, scale)
        reg = reg + 0.5 * torch.mean((diffs - target_diffs_2) ** 2)
        total_loss = (speed * ((1 / (reg_rate + 1)) * loss + (reg_rate / (reg_rate + 1)) * reg))
        total_loss.backward()
        optimizer.step()

        # Retraction mapping in the original space
        with torch.no_grad():
            X1_adapted.data = ret.f1(X1_adapted.data)

        # Update speed
        if total_loss.item() < last:
            speed *= 1.02
        else:
            speed *= 0.9
        speed = float(np.clip(speed, 0.1, 10))
        last = (loss + reg_rate * reg).item()

        losses.append([float(loss.item()), float(reg.item())])
        # print(last, losses[-1])
        # print((loss + reg).item(), loss.item(), reg.item(), len(X1), len(X2))

        # Check convergence
        p_values = compute_p_values(X1_corrected.cpu().data.numpy(), X2, u_test)
        median_p_values.append(float(np.median(p_values)))
        pbar.set_description(f'OT {median_p_values[-1]:.3f}')
        if median_p_values[-1] >= convergence_threshold:
            break
        if (iteration > 20) and (iteration % 20 == 0):
            reg_rate = max(1, reg_rate * 0.9)

    losses = np.asarray(losses, dtype=float)

    X1_corrected = X1_corrected.cpu().data.numpy()

    _, gamma = wasserstein_distance(X1_corrected, X2, return_plan=True)

    # Save figures
    try:
        if not os.path.isdir(folder):
            os.makedirs(folder)
        p_values = compute_p_values(X1_corrected, X2, u_test)
        log_(f'Median p-value after correction: {np.median(p_values)}', verbose=verbose)
        plt.subplot(2, 1, 1)
        plt.violinplot(p_values, vert=False, showmeans=True, showextrema=True)
        plt.xlabel('p-value')
        plt.subplot(2, 1, 2)
        plt.violinplot(np.log(p_values), vert=False, showmeans=True, showextrema=True)
        plt.xlabel('log(p-value)')
        plt.savefig(os.path.join(folder, 'p-values-final.png'), dpi=300)
        plt.clf()
        plt.plot(p_values)
        plt.xlabel('Bin')
        plt.ylabel('p-value')
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
    except (ValueError, IndexError):
        pass

    return X1_corrected
