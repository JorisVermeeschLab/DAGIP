# -*- coding: utf-8 -*-
#
#  gip.py
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

import numpy as np
import torch

from dagip.nn.gc_correction import diff_gc_correction
from dagip.retraction.base import Retraction


class GIPRetraction(Retraction):

    def __init__(self, gc_content: np.ndarray):
        self.gc_content = torch.FloatTensor((np.round(gc_content * 1000).astype(int) // 10).astype(float))

    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.clamp(X, 0)
        return X / torch.median(X, dim=1).values.unsqueeze(1)

    def _f2(self, X: torch.Tensor) -> torch.Tensor:
        return diff_gc_correction(X, self.gc_content)