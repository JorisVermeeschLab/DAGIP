# -*- coding: utf-8 -*-
#
#  minkowski_log.py
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

import torch

from dagip.spatial.base import BaseDistance


class MinkowskiDistanceOnLog(BaseDistance):

    def __init__(self, p: float = 1.0, eps: float = 1e-9):
        self.p: float = p
        self.eps: float = eps

    def pairwise_distances(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = torch.log(torch.clamp(X, self.eps, None))
        Y = torch.log(torch.clamp(Y, self.eps, None))
        return torch.cdist(X, Y, p=self.p)
