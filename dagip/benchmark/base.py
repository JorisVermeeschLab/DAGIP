# -*- coding: utf-8 -*-
#
#  base.py
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

from abc import abstractmethod, ABCMeta

import numpy as np

from dagip.retraction.base import Manifold


class BaseMethod(metaclass=ABCMeta):

    def __init__(
            self,
            sample_wise: bool,
            gc_correction: bool
    ):
        self.sample_wise: bool = bool(sample_wise)
        self.gc_correction: bool = bool(gc_correction)

    @abstractmethod
    def adapt(
            self,
            X: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            sample_names: np.ndarray,
            target_domain: int = 0
    ) -> np.ndarray:
        pass

    @abstractmethod
    def adapt_sample_wise(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
