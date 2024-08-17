# -*- coding: utf-8 -*-
#
#  kmm.py
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

import os
from typing import Tuple

import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from adapt.instance_based import KMM

from dagip.benchmark.base import BaseMethod


class KernelMeanMatching(BaseMethod):

    def normalize_(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return X

    def adapt_(self, Xs: np.ndarray, Xt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights_source = KMM(verbose=0).fit_weights(Xs, Xt)
        weights_source = weights_source * len(weights_source) / np.sum(weights_source)
        weights_target = np.ones(len(Xt))
        return Xs, weights_source, weights_target

    def name(self) -> str:
        return 'Kernel mean matching'