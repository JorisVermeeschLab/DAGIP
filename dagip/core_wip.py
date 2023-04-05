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
import math
import os.path
from typing import List, Tuple

import numpy as np
import ot
import torch.nn
import tqdm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder

from dagip.ichorcna.model import safe_division, IchorCNA
from dagip.nipt.binning import ChromosomeBounds
from dagip.tools.ichor_cna import create_ichor_cna_normal_panel, load_ichor_cna_results, ichor_cna


def wasserstein_distance(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    distances = cdist(X1, X2)
    gamma = ot.emd(a, b, distances ** p)
    return np.sum(gamma * distances) ** (1. / p)


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
    distances = pairwise_distances(X1, X2)
    np.int = int  # Fix numpy version issue
    gamma = ot.emd(a, b, distances ** p)
    gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
    return np.dot(gamma, X2)


def log_ratios(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.log(safe_division(n, d))


def quantile_transform(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    inv_idx1 = np.argsort(np.argsort(x1))
    idx2 = np.argsort(x2)
    return x2[idx2[inv_idx1]]


def robustness_scores(
        r1: np.ndarray,
        r2: np.ndarray,
        x1: np.ndarray,
        x2_prime: np.ndarray
) -> np.ndarray:
    c11 = log_ratios(x1, r1)
    c22 = log_ratios(x2_prime, r2)
    c12 = log_ratios(x1, r2)
    c21 = log_ratios(x2_prime, r1)

    """
    plt.plot(c11, label='x1 with panel 1')
    plt.plot(c22, label='x2 with panel 2')
    plt.plot(c12, label='x1 with panel 2')
    plt.plot(c21, label='x2 with panel 1')
    plt.legend()
    plt.show()
    """

    mean = 0.25 * c11 + 0.25 * c22 + 0.25 * c12 + 0.25 * c21
    ss_res = 0.5 * (c11 - c12) ** 2 + 0.5 * (c22 - c21) ** 2
    ss_tot = 0.25 * (c11 - mean) ** 2 + 0.25 * (c22 - mean) ** 2
    ss_tot += 0.25 * (c12 - mean) ** 2 + 0.25 * (c21 - mean) ** 2
    scores = np.ones(len(mean))
    mask = np.logical_and(ss_res > 0, ss_tot > 0)
    scores[mask] = np.maximum(1. - ss_res[mask] / ss_tot[mask], 0.)
    return scores


def median_and_rmsd(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    median = torch.median(X, dim=0).values
    # rmsd = torch.sqrt(torch.mean((X - median.unsqueeze(0)) ** 2, dim=0)) + 1e-7
    rmsd = torch.std(X, dim=0) + 1e-7
    return median, rmsd


def define_target(
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


def ot_da(
        ichor_cna_location: str,
        folder: str,
        gc_codes: List[str],
        X1: np.ndarray,
        X2: np.ndarray,
        side_info: np.ndarray,
        alpha: float = 0.4,
        reg_rate: float = 0,
        t_max: int = 1,
        p: int = 1,
        positive: bool = True
) -> np.ndarray:

    X1 = np.copy(X1)

    print(X1.dtype)

    r1 = np.median(X1, axis=0)
    r2 = np.median(X2, axis=0)

    gc_content = side_info[:, 0]
    mappability = side_info[:, 1]
    centromeric = side_info[:, 2].astype(bool)

    side_info = side_info[:, :2]

    model_filepath = 'icna.torch'
    icna = IchorCNA(3)
    if not os.path.exists(model_filepath):

        # Run ichorCNA on samples from domain 1
        if not os.path.isdir(folder):
            os.makedirs(folder)
        normal_panel_filepath = os.path.join(folder, 'normal-panel_median.rds')
        if not os.path.exists(normal_panel_filepath):
            create_ichor_cna_normal_panel(ichor_cna_location, folder, X1, gc_content, mappability)
        states = []
        log_ratios = []
        for i in tqdm.tqdm(range(len(X1)), desc='Loading ichorCNA results'):
            sample_folder = os.path.join(folder, gc_codes[i])
            results = load_ichor_cna_results(sample_folder)
            if not results['success']:
                ichor_cna(ichor_cna_location, normal_panel_filepath, sample_folder, X1[i, :], gc_content, mappability)
            results = load_ichor_cna_results(sample_folder)
            states.append(results['copy-number'])
            log_ratios.append(results['log-r'])
        states = np.asarray(states)
        log_ratios = np.asarray(log_ratios)

        icna.fit(X1, side_info, states)
        icna.save(model_filepath)
    else:
        icna.load(model_filepath)
    Y = torch.FloatTensor(icna.predict(X1, side_info))

    with torch.no_grad():
        r1, std = median_and_rmsd(torch.FloatTensor(X1))
        target_diffs = torch.distributions.Normal(r1, std).cdf(torch.FloatTensor(X1))
    X1_adapted = torch.nn.Parameter(torch.FloatTensor(np.copy(X1)))
    X1 = torch.FloatTensor(X1)
    optimizer = torch.optim.Adam([X1_adapted], lr=1e-5)


    side_info = torch.FloatTensor(side_info)
    for _ in range(100):

        idx = np.arange(len(X1))
        np.random.shuffle(idx)

        X2_prime = torch.FloatTensor(ot_mapping(X1_adapted.cpu().data.numpy(), X2))

        total_loss = 0.
        for i in idx:

            optimizer.zero_grad()

            r1_adapted = torch.median(X1_adapted, dim=0).values

            y_hat = icna.decode(torch.clamp(X1_adapted[i, :], 0), r1_adapted, side_info)

            loss = torch.mean((X1_adapted - X2_prime) ** 2)
            total_loss += loss.item()

            reg = 0.15 * torch.mean((y_hat - Y[i, :]) ** 2)
            (loss + reg).backward()
            optimizer.step()

            # print(loss.item(), reg.item())

            # print((loss + reg).item(), loss.item(), reg.item(), len(X1), len(X2))
        print(f'Total loss: {total_loss}')

    return np.maximum(X1_adapted.cpu().data.numpy(), 0)
