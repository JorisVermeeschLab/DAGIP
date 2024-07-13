# -*- coding: utf-8 -*-
#
#  base.py
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

from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseDistance(metaclass=ABCMeta):

    def __call__(self, X: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(X):
            X = torch.FloatTensor(X)
        if not torch.is_tensor(Y):
            Y = torch.FloatTensor(Y)
        D = self.pairwise_distances(X, Y)
        assert D.size() == (len(X), len(Y))
        return D

    @abstractmethod
    def pairwise_distances(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        pass
