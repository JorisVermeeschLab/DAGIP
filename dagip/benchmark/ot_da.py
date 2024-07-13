# -*- coding: utf-8 -*-
#
#  ot_da.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import os
import uuid

import numpy as np
from matplotlib import pyplot as plt

from dagip.benchmark.base import BaseMethod
from dagip.core import ot_da, train_adapter
from dagip.correction.gc import gc_correction
from dagip.plot import scatter_plot
from dagip.retraction import GIPManifold
from dagip.retraction.base import Manifold
from dagip.spatial.base import BaseDistance
from dagip.utils import log_


class OTDomainAdaptation(BaseMethod):

    def __init__(self, ret: Manifold, pairwise_distances: BaseDistance, folder: str, per_label: bool = False, gc_correction: bool = True):
        super().__init__(False, gc_correction)
        self.ret: Manifold = ret
        self.pairwise_distances: BaseDistance = pairwise_distances
        self.folder: str = folder
        self.per_label: bool = per_label

    def adapt(
            self,
            X: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            sample_names: np.ndarray,
            target_domain: int = 0
    ):

        X_adapted = np.copy(X)
        for domain in np.unique(d):
            if domain != target_domain:

                if self.per_label:

                    # Define cohorts
                    X1s, X2s = [], []
                    for label in np.unique(y):
                        mask_source = np.logical_and(d == domain, y == label)
                        mask_target = np.logical_and(d == target_domain, y == label)
                        if np.sum(mask_target) == 0:
                            continue
                        if np.sum(mask_source) == 0:
                            continue
                        X1s.append(X_adapted[mask_source])
                        X2s.append(X_adapted[mask_target])

                    # Train adapter
                    folder = os.path.join(self.folder, str(uuid.uuid4()))
                    if not os.path.isdir(folder):
                        os.makedirs(folder)
                    adapter = train_adapter(folder, X1s, X2s, manifold=self.ret, pairwise_distances=self.pairwise_distances)

                    # Adapt samples
                    mask_source = (d == domain)
                    if np.sum(mask_source) == 0:
                        continue
                    X_adapted[mask_source, :] = adapter.adapt(X_adapted[mask_source])

                else:
                    mask_source = (d == domain)
                    mask_target = (d == target_domain)
                    if np.sum(mask_target) == 0:
                        continue
                    if np.sum(mask_source) == 0:
                        continue

                    # Train model
                    X_adapted[mask_source, :] = self.adapt_(
                        X_adapted[mask_source],
                        X_adapted[mask_target],
                        sample_names[mask_source]
                    )

        plt.subplot(1, 2, 1)
        scatter_plot(X, y, d)
        plt.subplot(1, 2, 2)
        scatter_plot(X_adapted, y, d)
        plt.savefig(os.path.join(self.folder, f'scatter-{str(uuid.uuid4())}.png'), dpi=300)
        plt.clf()

        return X_adapted

    def adapt_(self, X1: np.ndarray, X2: np.ndarray, sample_names: np.ndarray) -> np.ndarray:
        folder = os.path.join(self.folder, str(uuid.uuid4()))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        return ot_da(folder, X1, X2, manifold=self.ret, pairwise_distances=self.pairwise_distances)

    def adapt_sample_wise(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()

    def name(self) -> str:
        if self.gc_correction:
            return 'OT'
        else:
            return 'OT (no GC-correction)'
