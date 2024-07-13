# -*- coding: utf-8 -*-
#
#  kl_divergence.py
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

from abc import abstractmethod

import torch

from dagip.spatial.base import BaseDistance


class KLDivergence(BaseDistance):

    def pairwise_distances(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(0)
        X = torch.clamp(X, eps, 1)
        Y = torch.clamp(Y, eps, 1)
        M = 0.5 * (X + Y)
        D12 = torch.sum(X * torch.log(X / M), dim=2)
        D21 = torch.sum(Y * torch.log(Y / M), dim=2)
        return torch.sqrt(0.5 * (D12 + D21))
