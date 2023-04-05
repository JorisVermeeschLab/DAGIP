# -*- coding: utf-8 -*-
#
#  gc_correction.py: Differentiable GC-content bias correction
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
from typing import Tuple

import numpy as np
import torch
from statsmodels.nonparametric.smoothers_lowess import lowess


class Lowess(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            endog: torch.Tensor,
            exog: torch.Tensor,
            frac: float
    ) -> torch.Tensor:

        out = torch.clone(endog)
        for i in range(len(endog)):
            out[i, :] = torch.FloatTensor(lowess(
                endog[i, :].cpu().data.numpy(), exog.cpu().data.numpy(), frac=frac, return_sorted=False
            ))

        ctx.save_for_backward(endog, exog)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:

        Y, x = ctx.saved_tensors
        m, n = Y.size()

        grad_input = torch.clone(grad_output)
        for i in range(n):
            w = np.abs(x - x[i])
            x_bar = np.average(x, weights=w)

            # Partial derivative of the slope w.r.t. Y[k, i]
            numerator = np.average((x - x_bar) * ((np.arange(n) == i) - 1. / n), weights=w)
            denominator = np.average((x - x_bar) ** 2, weights=w)
            grad_a = numerator / denominator

            grad_input[:, i] *= ((x[i] - x_bar) * grad_a + 1. / n)

        return grad_input, None, None


def diff_gc_correction(X: torch.Tensor, exog: torch.Tensor, frac: float = 0.3) -> torch.Tensor:
    y_pred = Lowess.apply(X, exog, frac)
    mask = (y_pred == 0)
    y_pred = mask + (~mask) * y_pred

    X_adapted = X / y_pred

    mask = (exog == 0).unsqueeze(0)
    X_adapted = mask * X + (~mask) * X_adapted

    return X_adapted
