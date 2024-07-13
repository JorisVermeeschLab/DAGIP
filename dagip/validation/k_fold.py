# -*- coding: utf-8 -*-
#
#  k_fold.py
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

from typing import Optional, Tuple, List

import numpy as np
import statsmodels.stats.api as sms
from sklearn.model_selection import KFold, GroupKFold

from dagip.benchmark.base import BaseMethod
from dagip.validation.base import CrossValidation


class KFoldValidation(CrossValidation):

    def __init__(
            self,
            *args,
            n_splits: int = 5,
            n_repeats: int = 10,
            average_results: bool = True,
            groups: Optional[np.ndarray] = None,
            random_state: Optional[int] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_splits: int = int(n_splits)
        self.n_repeats: int = int(n_repeats)
        self.average_results: bool = average_results
        self.groups: Optional[np.ndarray] = groups
        self.random_state: Optional[int] = random_state
        self.splits: List[Tuple[np.ndarray, np.ndarray]] = self.make_splits()

    def make_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        idx = np.where(self.d == self.target_domain)[0]
        if self.groups is not None:
            kf = GroupKFold(n_splits=self.n_splits)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        splits = []
        for idx_train, idx_test in kf.split(idx, groups=(None if (self.groups is None) else self.groups[idx])):
            idx_train = idx[idx_train]
            idx_test = idx[idx_test]

            mask_train = np.zeros(len(self.d), dtype=bool)
            mask_test = np.zeros(len(self.d), dtype=bool)
            mask_train[idx_train] = True
            mask_test[idx_test] = True

            # Any sample that is not part of the target domain goes to the training set,
            # unless that sample is part of a group that is present in the test set.
            if self.groups is not None:
                blacklisted_groups = set(self.groups[mask_test])
                for i in range(len(self.d)):
                    if (self.d[i] != self.target_domain) and (self.groups[i] not in blacklisted_groups):
                        mask_train[i] = True
            else:
                mask_train[self.d != self.target_domain] = True

            # Split data into training/held-out sets
            idx_train = np.where(mask_train)[0]
            idx_test = np.where(mask_test)[0]

            assert len(set(list(idx_train)).intersection(set(list(idx_test)))) == 0

            splits.append((idx_train, idx_test))

            assert not np.any(self.d[idx_test] != self.target_domain)
        return splits

    def validate_by_concatenating(self, da_method: BaseMethod):

        results_ = {}
        for model_name in CrossValidation.PIPELINES.keys():
            results_[model_name] = []

        for _ in range(self.n_repeats):

            y_target_train = []
            y_target = []
            y_pred_train = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}
            y_pred = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}

            self.splits = self.make_splits()
            for idx_train, idx_test in self.splits:

                X_train, X_test, y_train, y_test = self.adapt(idx_train, idx_test, da_method)

                y_target_train.append(y_train)
                y_target.append(y_test)

                for model_name, pipeline in CrossValidation.PIPELINES.items():

                    # Train supervised model
                    pipeline.fit(X_train, y_train)

                    # Predict on training set
                    y_pred_train[model_name].append(pipeline.predict_proba(X_train)[:, 1])

                    # Predict on the held-out samples
                    y_pred[model_name].append(pipeline.predict_proba(X_test)[:, 1])

            y_target_train = np.squeeze(np.concatenate(y_target_train, axis=0))
            y_target = np.squeeze(np.concatenate(y_target, axis=0))
            for model_name in CrossValidation.PIPELINES.keys():
                y_pred_train[model_name] = np.squeeze(np.concatenate(y_pred_train[model_name], axis=0))
                y_pred[model_name] = np.squeeze(np.concatenate(y_pred[model_name], axis=0))
                results_[model_name].append(CrossValidation.compute_evaluation_metrics(y_target, y_pred[model_name]))
                results_[model_name][-1].update(CrossValidation.compute_evaluation_metrics(y_target_train, y_pred_train[model_name], train=True))

        results = {}
        for model_name in CrossValidation.PIPELINES.keys():
            results[model_name] = {}
            for key in results_[model_name][0].keys():
                values = [float(results_[model_name][i][key]) for i in range(len(results_[model_name]))]
                results[model_name][key] = float(np.mean(values))
                results[model_name][key + 'confint'] = sms.DescrStatsW(values).tconfint_mean()
            print(f'Validation results for "{model_name}": {results[model_name]}')

        self.table.add(da_method.name(), results)

        return results

    def validate_by_averaging(self, da_method: BaseMethod):

        results_ = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}

        for _ in range(self.n_repeats):
            self.splits = self.make_splits()
            for idx_train, idx_test in self.splits:

                X_train, X_test, y_train, y_test = self.adapt(idx_train, idx_test, da_method)

                for model_name, pipeline in CrossValidation.PIPELINES.items():

                    # Train supervised model
                    pipeline.fit(X_train, y_train)

                    # Predict on the held-out sample
                    y_pred = pipeline.predict_proba(X_test)[:, 1]

                    results_[model_name].append(CrossValidation.compute_evaluation_metrics(y_test, y_pred))

        results = {}
        for model_name in CrossValidation.PIPELINES.keys():
            results[model_name] = {}
            for key in results_[model_name][0].keys():
                values = [float(results_[model_name][i][key]) for i in range(len(results_[model_name]))]
                results[model_name][key] = float(np.mean(values))
                print(sms.DescrStatsW(values).tconfint_mean())
                results[model_name][key + 'confint'] = sms.DescrStatsW(values).tconfint_mean()
            print(f'Validation results for "{model_name}": {results[model_name]}')
        self.table.add(da_method.name(), results)

        return results

    def validate(self, da_method: BaseMethod):
        if self.average_results:
            return self.validate_by_averaging(da_method)
        else:
            return self.validate_by_concatenating(da_method)
