# -*- coding: utf-8 -*-
#
#  centering_scaling.py
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

import numpy as np
from sklearn.preprocessing import RobustScaler

from dagip.benchmark.base import BaseMethod


class CenteringScaling(BaseMethod):

    def __init__(self, with_std: bool = False):
        super().__init__(False, False)
        self.with_std: bool = bool(with_std)

    def adapt(
            self,
            X: np.ndarray,
            X_uncorrected: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            t: np.ndarray,
            side_info: np.ndarray,
            sample_names: np.ndarray,
            target_domain: int = 0
    ):
        target_scaler = RobustScaler(with_scaling=self.with_std)
        target_scaler.fit(X[d == target_domain])

        X_adapted = np.copy(X)
        for domain in np.unique(d):
            if domain != target_domain:
                mask = (d == domain)
                X_adapted[mask, :] = RobustScaler(with_scaling=self.with_std).fit_transform(X_adapted[mask, :])
                X_adapted[mask, :] = target_scaler.inverse_transform(X_adapted[mask, :])
        return X_adapted

    def adapt_sample_wise(
            self,
            X: np.ndarray,
            t: np.ndarray,
            side_info: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()

    def name(self) -> str:
        return 'Centering-scaling'
