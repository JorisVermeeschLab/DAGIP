# -*- coding: utf-8 -*-
#
#  adapter.py
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

from typing import Tuple, Optional

import numpy as np
import torch

from dagip.retraction.base import Manifold


class MLPAdapter(torch.nn.Module):

    def __init__(
            self,
            n: int,
            m: int,
            latent_shape: Tuple[int, ...],
            manifold: Manifold,
            eta: float = 0.001
    ):
        torch.nn.Module.__init__(self)
        self.n: int = n
        self.m: int = m
        self.shape = tuple([self.m] + list(latent_shape) + [self.m])
        self.manifold: Manifold = manifold
        self.eta: float = eta

        layers = [torch.nn.LayerNorm(self.shape[0])]
        for k in range(len(self.shape) - 1):
            n_in = self.shape[k]
            n_out = self.shape[k + 1]
            layers.append(torch.nn.Linear(n_in, n_out))
            if k != len(self.shape) - 2:
                layers.append(torch.nn.LayerNorm(n_out))
                layers.append(torch.nn.LeakyReLU())
        self.bias_estimator = torch.nn.Sequential(*layers)
        self.bias_estimator.apply(MLPAdapter.init_weights)

    def forward(self, X_original: torch.Tensor) -> torch.Tensor:
        X = self.manifold.inverse_transform(X_original)
        bias = self.bias_estimator(X_original)
        return self.manifold.transform(X + self.eta * bias)

    @torch.no_grad()
    def adapt(self, X: np.ndarray) -> np.ndarray:
        X = torch.FloatTensor(X)
        X = self.forward(X)
        return X.cpu().data.numpy()

    @staticmethod
    def init_weights(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.001)
