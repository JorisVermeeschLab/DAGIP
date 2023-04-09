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

import warnings
from abc import abstractmethod, ABCMeta
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from dagip.benchmark.base import BaseMethod
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.utils import LaTeXTable


class CrossValidation(metaclass=ABCMeta):

    PIPELINES = {
        'svm': SVC(C=1, class_weight='balanced', probability=True),
        'rf': RandomForestClassifier(max_depth=5, class_weight='balanced'),  # 5
        # 'knn': KNeighborsClassifier(weights='distance'),
        'reglog': LogisticRegression(C=1, class_weight='balanced', max_iter=10000)
    }

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            t: np.ndarray,
            side_info: np.ndarray,
            sample_names: np.ndarray,
            target_domain: int = 0,
            redo_binning: bool = True
    ):
        self.X: np.ndarray = X
        self.side_info: np.ndarray = side_info
        self.X_gc_corrected: np.ndarray = gc_correction(self.X, self.side_info[0, :])
        self.y: np.ndarray = y
        self.d: np.ndarray = d
        self.t: np.ndarray = t
        self.sample_names: np.ndarray = sample_names
        self.target_domain: int = int(target_domain)
        self.redo_binning: bool = bool(redo_binning)

        # Results
        self.table: LaTeXTable = LaTeXTable()

    def adapt(
            self,
            idx_train: np.ndarray,
            idx_test: np.ndarray,
            da_method: BaseMethod
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        assert len(set(list(idx_train)).intersection(set(list(idx_test)))) == 0

        t_train, t_test = self.t[idx_train], self.t[idx_test]
        d_train, d_test = self.d[idx_train], self.d[idx_test]
        assert len(np.unique(d_test)) == 1
        y_train, y_test = self.y[idx_train], self.y[idx_test]
        sample_names_train = self.sample_names[idx_train]

        if da_method.gc_correction:
            X_train, X_test = self.X_gc_corrected[idx_train], self.X_gc_corrected[idx_test]
        else:
            X_train, X_test = self.X[idx_train], self.X[idx_test]
        X_train_uncorrected = self.X[idx_train]

        # Shuffle training set
        six_train = np.arange(len(X_train))
        np.random.shuffle(six_train)
        X_train = X_train[six_train, :]
        X_train_uncorrected = X_train_uncorrected[six_train, :]
        y_train = y_train[six_train]
        d_train = d_train[six_train]
        t_train = t_train[six_train]
        sample_names_train = sample_names_train[six_train]

        # Shuffle test set
        six_test = np.arange(len(X_test))
        np.random.shuffle(six_test)
        X_test = X_test[six_test, :]
        y_test = y_test[six_test]
        t_test = t_test[six_test]

        # Domain adaptation
        if da_method.sample_wise:
            X_train = da_method.adapt_sample_wise(X_train, t_train, self.side_info)
            X_test = da_method.adapt_sample_wise(X_test, t_test, self.side_info)
        else:
            X_train = da_method.adapt(
                X_train,
                X_train_uncorrected,
                y_train,
                d_train,
                t_train,
                self.side_info,
                sample_names_train,
                self.target_domain
            )

        # Inverse-shuffle the training set
        six_train = np.argsort(six_train)
        X_train = X_train[six_train, :]
        del X_train_uncorrected
        y_train = y_train[six_train]

        # Inverse-shuffle the test set
        six_test = np.argsort(six_test)
        X_test = X_test[six_test, :]
        y_test = y_test[six_test]

        # Remove low-mappability bins
        mappability = self.side_info[1, :]
        mask = (mappability >= 0.8)
        X_train = X_train[:, mask]
        X_test = X_test[:, mask]

        # Standard scaling
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    @abstractmethod
    def validate(self, da_method: BaseMethod) -> dict:
        pass

    @staticmethod
    def best_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        scores = []
        for threshold in y_pred:
            scores.append(matthews_corrcoef(y_pred > threshold, y_true))
        return y_pred[np.argmax(scores)]

    @staticmethod
    def compute_evaluation_metrics(y_target: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            tn, fp, fn, tp = confusion_matrix(y_target, y_pred > 0.5).ravel()
            sensitivity = np.nan_to_num(tp / (tp + fn))
            specificity = np.nan_to_num(tn / (tn + fp))
            mcc = matthews_corrcoef(y_target, y_pred > 0.5)

            threshold = CrossValidation.best_threshold(y_target, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_target, y_pred > threshold).ravel()
            sensitivity_best = np.nan_to_num(tp / (tp + fn))
            specificity_best = np.nan_to_num(tn / (tn + fp))
            mcc_best = matthews_corrcoef(y_target, y_pred > threshold)

        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'mcc': mcc,
            'sensitivity-best': sensitivity_best,
            'specificity-best': specificity_best,
            'mcc-best': mcc_best,
            'auroc': roc_auc_score(y_target, y_pred),
            'aupr': average_precision_score(y_target, y_pred)
        }
