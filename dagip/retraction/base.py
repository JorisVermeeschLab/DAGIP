# -*- coding: utf-8 -*-
#
#  base.py
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

from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
import torch


class Retraction(metaclass=ABCMeta):
    """Retraction mapping, implemented in two parts.

    Original space ---f1---> Original space ---f2---> Manifold
    """

    def __call__(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return self.f2(self.f1(X))

    def f1(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(X):
            X = torch.FloatTensor(X)
        X_prime = self._f1(X)
        assert X_prime.size() == X.size()
        return X_prime

    @abstractmethod
    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def f2(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(X):
            X = torch.FloatTensor(X)
        return self._f2(X)

    @abstractmethod
    def _f2(self, X: torch.Tensor) -> torch.Tensor:
        pass
