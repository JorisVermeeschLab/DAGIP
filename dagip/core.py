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
from typing import Tuple, Union, Callable, List

import numpy as np
import ot
import scipy.stats
import torch
import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM

from dagip.nipt.binning import ChromosomeBounds
from dagip.nn.adapter import MLPAdapter
from dagip.optimize.loss_scaling import LossScaling
from dagip.plot import plot_ot_plan_degrees
from dagip.retraction import Identity
from dagip.retraction.base import Manifold
from dagip.retraction.probability_simplex import ProbabilitySimplex
from dagip.spatial.base import BaseDistance
from dagip.spatial.euclidean import EuclideanDistance
from dagip.spatial.kl_divergence import KLDivergence
from dagip.utils import log_


def compute_weights(X: np.ndarray) -> np.ndarray:
    return np.full(len(X), 1. / len(X))


def transport_plan(
        distances: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(distances)
    m = distances.shape[1]
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    assert not np.any(np.isnan(distances))
    assert not np.any(np.isnan(distances ** p))
    assert not np.any(np.isinf(distances ** p))

    return ot.emd(a, b, distances ** p)


def default_u_test(x1: np.ndarray, x2: np.ndarray) -> float:
    return float(scipy.stats.ks_2samp(x1, x2).pvalue)


def compute_p_values(X1: np.ndarray, X2: np.ndarray, u_test: Callable) -> np.ndarray:
    p_values = []
    for k in range(X1.shape[1]):
        p_values.append(u_test(X1[:, k], X2[:, k]))
    return np.asarray(p_values)


def compute_s_values(X: torch.Tensor) -> torch.Tensor:
    #scale = 1. / torch.clamp(torch.std(X, dim=0), 1e-9, None).unsqueeze(0)
    #mu = torch.mean(X, dim=0).unsqueeze(0)
    #return scale * (X - mu)
    mu = torch.quantile(X, 0.5, dim=0).unsqueeze(0)
    sigma = (torch.quantile(X, 0.75, dim=0) - torch.quantile(X, 0.25, dim=0)).unsqueeze(0)
    sigma = torch.clamp(sigma, 1e-9, None)
    scale = 1. / sigma
    return scale * (X - mu)


def barycentric_mapping(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    D = cdist(X1, X2)
    gamma = transport_plan(D, p=p)
    gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
    return np.dot(gamma, X2)


def _ot_da(
        folder: str,
        X1: Union[np.ndarray, List[np.ndarray]],
        X2: Union[np.ndarray, List[np.ndarray]],
        manifold: Manifold = Identity(),
        pairwise_distances: BaseDistance = EuclideanDistance(),
        u_test: Callable = default_u_test,
        reg_rate: float = 0.1,
        max_n_iter: int = 4000,  # 1000
        convergence_threshold: float = 0.5,
        lr: float = 0.005,  # 0.0002
        verbose: bool = True
) -> Tuple[np.ndarray, MLPAdapter]:

    X1s = [X1] if (not isinstance(X1, list)) else X1
    X2s = [X2] if (not isinstance(X2, list)) else X2
    assert len(X1s) == len(X2s)

    block_sizes_1 = [len(X1) for X1 in X1s]
    block_sizes_2 = [len(X2) for X2 in X1s]

    X1 = np.concatenate(X1s, axis=0)
    X2 = np.concatenate(X2s, axis=0)

    with torch.no_grad():
        X1_second = manifold(X1)
        target_diffs_1 = compute_s_values(X1_second)
        target_diffs_2 = compute_s_values(torch.FloatTensor(X2))
        target_diffs = torch.cat([target_diffs_1, target_diffs_2], dim=0)

    x2_variance = np.mean(np.var(X2, axis=0))

    # Determine convergence criterion
    p_values = compute_p_values(X1, X2, u_test)
    median_p_value = np.median(p_values)
    log_(f'Median p-value: {median_p_value}', verbose=verbose)

    X2_barycentric = barycentric_mapping(X1, X2)
    avg_deviation = np.mean(np.abs(X1 - X2_barycentric))

    X1 = torch.FloatTensor(X1)
    X2 = torch.FloatTensor(X2)
    X0 = None

    # Compute initial transport plans
    X1_second = manifold.transform(X1)
    gammas = []
    offset1, offset2 = 0, 0
    for size1, size2 in zip(block_sizes_1, block_sizes_2):
        distances = torch.cdist(X1_second[offset1:offset1+size1], X2[offset2:offset2+size2], p=1)
        #distances = pairwise_distances(X1_second[offset1:offset1+size1], X2[offset2:offset2+size2])
        gammas.append(torch.FloatTensor(transport_plan(distances.cpu().data.numpy())))

    n_hidden = 32
    adapter = MLPAdapter(X1.size()[0], X1.size()[1], (n_hidden, n_hidden, n_hidden, n_hidden), manifold, eta=avg_deviation)
    best_state_dict = adapter.state_dict()
    best_global_obj = 0.0

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    loss1_scaling = LossScaling()
    loss2_scaling = LossScaling()
    loss3_scaling = LossScaling()

    variances = []
    losses = []
    median_p_values = []
    pbar = tqdm.tqdm(range(max_n_iter), desc='OT')
    for iteration in pbar:

        optimizer.zero_grad()

        # Adaptation on the manifold
        X1_second = adapter(X1)
        if X0 is None:
            X0 = X1_second.cpu().data.numpy()

        # Define targets
        #if (iteration % 10 == 0) or (X2_prime is None):
        #    X2_prime = torch.FloatTensor(ot_mapping(X1_second.cpu().data.numpy(), X2))

        # Compensate the reduction of variance caused by the OT mapping
        #factor = torch.mean(torch.std(torch.FloatTensor(X2), dim=0)) / torch.mean(torch.std(X2_prime, dim=0))
        #mu = torch.mean(X2_prime, dim=0).unsqueeze(0)
        #X2_prime = factor * (X2_prime - mu) + mu

        # Wasserstein distance
        loss1, norm = 0.0, 0.0
        offset1, offset2 = 0, 0
        for size1, size2, gamma in zip(block_sizes_1, block_sizes_2, gammas):

            # Compute pairwise distances
            distances = pairwise_distances(X1_second[offset1:offset1+size1], X2[offset2:offset2+size2])

            # OT distance
            mask = (gamma > 1e-15)
            gamma_ = gamma[mask]
            distances_ = distances[mask]

            # Re-weight cohort by its size
            loss1 = loss1 + torch.sum(gamma_ * torch.square(distances_)) * len(gamma_)
            norm = norm + len(gamma_)
        loss1 = loss1 / norm

        loss1 = loss1_scaling(loss1)

        # Encourage the variances to be identical
        x1_variance = torch.mean(torch.var(X1_second, dim=0))
        loss2 = torch.square(torch.sqrt(x1_variance) - np.sqrt(x2_variance))
        loss2 = loss2_scaling(loss2)

        # Regularization function
        X12 = torch.cat([X1_second, X2], dim=0)
        diffs = compute_s_values(X12)
        loss3 = torch.mean(torch.square(diffs - target_diffs))
        loss3 = loss3_scaling(loss3)

        # Regularization of the bias function
        #X2_second = adapter(X2, manifold)
        #loss4 = torch.mean(torch.square(X2_second - X2))
        #loss4 = loss4_scaling(loss4)

        # Compute total loss function
        total_loss = loss1 + 0.01 * loss2 + reg_rate * loss3

        losses.append([loss1.item(), loss2.item(), loss3.item()])

        variances.append(x1_variance.item())

        # Check convergence
        p_values = compute_p_values(X1_second.cpu().data.numpy(), X2, u_test)
        median_p_values.append(float(np.median(p_values)))
        pbar.set_description(f'p={median_p_values[-1]:.3f}')
        if median_p_values[-1] >= convergence_threshold:
            break
        if (iteration > 10) and (iteration % 4 == 0):
            reg_rate *= 0.9

        # Check global objective
        if median_p_values[-1] > best_global_obj:
            best_global_obj = median_p_values[-1]
            best_state_dict = adapter.state_dict()

        # Update parameters
        optimizer.step()

        # Backpropagation
        total_loss.backward()

        # Update parameters
        optimizer.step()

    # Restore best NN model
    adapter.load_state_dict(best_state_dict)
    adapter.eval()
    X1_second = adapter(X1)
    X1_second = X1_second.cpu().data.numpy()

    losses = np.asarray(losses, dtype=float)
    gamma = gamma.cpu().data.numpy()

    X1_barycentric = barycentric_mapping(X1.cpu().data.numpy(), X2.cpu().data.numpy())

    # Save figures
    try:
        if not os.path.isdir(folder):
            os.makedirs(folder)

        plt.plot(variances)
        plt.axhline(y=x2_variance)
        plt.xlabel('Iterations')
        plt.ylabel('Total X1 variance')
        plt.savefig(os.path.join(folder, 'total-variance.png'), dpi=300)
        plt.clf()

        p_values = compute_p_values(X1_second, X2, u_test)
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
        plt.imshow(cdist(X1_second, X1_second))
        plt.savefig(os.path.join(folder, 'D11.png'), dpi=400)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(cdist(X1, X2))
        plt.subplot(1, 2, 2)
        plt.imshow(cdist(X1_second, X2))
        plt.savefig(os.path.join(folder, 'D12.png'), dpi=400)
        plt.clf()
        plt.plot(median_p_values)
        plt.xlabel('Iterations')
        plt.ylabel('Median p-value')
        plt.savefig(os.path.join(folder, 'median-p-values.png'), dpi=300)
        plt.clf()
        plt.plot(losses[:, 0], label='Loss')
        plt.plot(losses[:, 1], label='Variance penalty')
        plt.plot(losses[:, 2], label='Regularization')
        #plt.plot(losses[:, 3], label='Bias function regularization')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss function')
        plt.savefig(os.path.join(folder, 'loss.png'), dpi=300)
        plt.clf()

        plt.imshow(gamma)
        plt.savefig(os.path.join(folder, 'transport-plan.png'), dpi=300)
        plt.clf()

        ax = plt.subplot(1, 1, 1)
        plot_ot_plan_degrees(ax, gamma)
        plt.savefig(os.path.join(folder, 'transport-plan-degrees.png'), dpi=300)
        plt.clf()

        for transformer, filename in zip([KernelPCA(), TSNE()], ['kpca.png', 'tsne.png']):
            plt.subplot(1, 2, 1)
            coords = transformer.fit_transform(np.concatenate([X1, X2, X1_barycentric], axis=0))
            plt.scatter(coords[:len(X1), 0], coords[:len(X1), 1], alpha=0.4)
            plt.scatter(coords[len(X1):len(X1)+len(X2), 0], coords[len(X1):len(X1)+len(X2), 1], alpha=0.4)
            plt.scatter(coords[len(X1) + len(X2):, 0], coords[len(X1) + len(X2):, 1], alpha=0.4, marker='x')
            plt.subplot(1, 2, 2)
            coords = transformer.fit_transform(np.concatenate([X1_second, X2, X1_barycentric], axis=0))
            plt.scatter(coords[:len(X1), 0], coords[:len(X1), 1], alpha=0.4)
            plt.scatter(coords[len(X1):len(X1)+len(X2), 0], coords[len(X1):len(X1)+len(X2), 1], alpha=0.4)
            plt.scatter(coords[len(X1) + len(X2):, 0], coords[len(X1) + len(X2):, 1], alpha=0.4, marker='x')
            plt.savefig(os.path.join(folder, filename), dpi=300)
            plt.clf()

        transformer = KernelPCA()
        coords = transformer.fit_transform(X1)
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4, label='Original X1')
        coords = transformer.transform(X0)
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4, label='X1 at t=1')
        plt.legend()
        plt.savefig(os.path.join(folder, 'initial-X1.png'), dpi=300)
        plt.clf()

    except (ValueError, IndexError):
        pass

    # Fix numerical issues
    mask = np.logical_or(np.isnan(X1_second), np.isinf(X1_second))
    if np.any(mask):
        print('WARNING: Output contains NaNs. Replacing them with original data.')
    X1_second[mask] = X1[mask]

    return X1_second, adapter


def ot_da(*args, **kwargs) -> np.ndarray:
    X1_second, adapter = _ot_da(*args, **kwargs)
    return X1_second


def train_adapter(*args, **kwargs) -> MLPAdapter:
    X1_second, adapter = _ot_da(*args, **kwargs)
    return adapter
