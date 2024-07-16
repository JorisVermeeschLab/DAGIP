# -*- coding: utf-8 -*-
#
#  no_source.py
#
#  Copyright 2024 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import numpy as np

from dagip.benchmark.base import BaseMethod


class NoSourceData(BaseMethod):

    def __init__(self):
        super().__init__(False, False)

    def adapt(
            self,
            X: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            sample_names: np.ndarray,
            target_domain: int = 0
    ):
        X_adapted = np.copy(X)
        for label in np.unique(y):

            target_mask = np.logical_and(d == target_domain, y == label)
            if not np.any(target_mask):
                continue

            for domain in np.unique(d):
                if domain != target_domain:
                    mask = np.logical_and(d == domain, y == label)

                    n = int(np.sum(mask))
                    idx_repl = np.random.randint(0, int(np.sum(target_mask)), size=n)

                    X_adapted[mask, :] = X_adapted[target_mask, :][idx_repl, :]

        return X_adapted

    def adapt_sample_wise(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def name(self) -> str:
        return 'No source data'
