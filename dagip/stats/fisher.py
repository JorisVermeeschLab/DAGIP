# -*- coding: utf-8 -*-
#
#  fisher.py
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

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


def reglog_fisher_kernel(X: np.ndarray, y: np.ndarray, d: np.ndarray) -> np.ndarray:

    # Train and predict with logistic regression
    #X = RobustScaler().fit_transform(X)
    model = LogisticRegression()
    model.fit(X, y)
    y_proba = model.predict_proba(X)[:, 1]

    # Compute Fisher information matrix
    idx = np.where(d == 0)[0]
    I = np.einsum('i,ik,il->kl', (y_proba * (1 - y_proba))[idx], X[idx, :], X[idx, :]) / len(idx)
    I_inv = np.linalg.inv(I)

    # Compute Fisher scores
    scores = (y - y_proba)[:, np.newaxis] * X

    # Compute Fisher kernels
    K = scores @ I_inv @ scores.T
    scores = np.diagonal(K)
    scores = np.abs(scores)
    scores = np.log1p(scores)
    print(scores)
    return scores


def reglog_fisher_info(X: np.ndarray, y: np.ndarray, d: np.ndarray) -> np.ndarray:
    X = RobustScaler().fit_transform(X)
    model = SVC(probability=True)
    model.fit(X[d == 0], y[d == 0])
    y_proba = model.predict_proba(X)[:, 1]

    # Compute Fisher information matrices
    fisher_info = np.sum((y_proba * (1. - y_proba))[:, np.newaxis] * np.square(X), axis=1)
    #fisher_info = (y_proba * (1. - y_proba))

    return fisher_info
