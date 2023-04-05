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

from dagip.ichorcna.model import safe_division
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


def log_ratios(X: torch.Tensor) -> torch.Tensor:
    X = X + 0.01
    r = torch.median(X, dim=0).values.unsqueeze(0)
    return torch.log2(X / r)


def median_and_rmsd(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    median = torch.median(X, dim=0).values
    rmsd = torch.sqrt(torch.mean((X - median.unsqueeze(0)) ** 2))
    return median, rmsd


def ot_da(
        ichor_cna_location: str,
        folder: str,
        sample_names: List[str],
        X1: np.ndarray,
        X2: np.ndarray,
        side_info: np.ndarray,
        alpha: float = 0.4,
        reg_rate: float = 0,
        t_max: int = 5,
        p: int = 1,
        positive: bool = True
) -> np.ndarray:

    with torch.no_grad():
        r1, std = median_and_rmsd(torch.FloatTensor(X1))
        target_diffs = torch.distributions.Normal(r1, std).cdf(torch.FloatTensor(X1))
    X1_adapted = torch.nn.Parameter(torch.FloatTensor(np.copy(X1)))
    X1 = torch.FloatTensor(X1)
    optimizer = torch.optim.Adam([X1_adapted], lr=1e-3 * 0.5)

    for _ in range(5000):

        optimizer.zero_grad()
        X2_prime = torch.FloatTensor(ot_mapping(X1_adapted.cpu().data.numpy(), X2))

        #r1_adapted = torch.median(X1_adapted, dim=0).values.unsqueeze(0)
        #diffs = X1_adapted - r1_adapted
        r1_adapted, std = median_and_rmsd(X1_adapted)
        diffs = torch.distributions.Normal(r1_adapted, std).cdf(X1_adapted)

        loss = torch.mean((X1_adapted - X2_prime) ** 2)
        reg = 0.02 * torch.mean((diffs - target_diffs) ** 2)
        (loss + reg).backward()
        optimizer.step()

        print((loss + reg).item(), loss.item(), reg.item(), len(X1), len(X2))

    return np.maximum(X1_adapted.cpu().data.numpy(), 0)
