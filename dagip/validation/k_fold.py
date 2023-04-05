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

from typing import Optional

import numpy as np
from sklearn.model_selection import KFold

from dagip.benchmark.base import BaseMethod
from dagip.validation.base import CrossValidation


class KFoldValidation(CrossValidation):

    def __init__(
            self,
            *args,
            n_splits: int = 5,
            average_results: bool = True,
            groups: Optional[np.ndarray],
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_splits: int = int(n_splits)
        self.average_results: bool = average_results

        idx = np.where(self.d == self.target_domain)[0]

        kf = KFold(n_splits=self.n_splits, shuffle=True)
        self.splits = []
        for idx_train, idx_test in kf.split(idx, groups=(None if (groups is None) else groups[idx])):
            idx_train = idx[idx_train]
            idx_test = idx[idx_test]

            mask_train = np.zeros(len(self.d), dtype=bool)
            mask_test = np.zeros(len(self.d), dtype=bool)
            mask_train[idx_train] = True
            mask_test[idx_test] = True

            # Any sample that is not part of the target domain goes to the training set,
            # unless that sample is part of a group that is present in the test set.
            if groups is not None:
                blacklisted_groups = set(groups[mask_test])
                for i in range(len(self.d)):
                    if (self.d[i] != self.target_domain) and (groups[i] not in blacklisted_groups):
                        mask_train[i] = True
            else:
                mask_train[self.d != self.target_domain] = True

            # Split data into training/held-out sets
            idx_train = np.where(mask_train)[0]
            idx_test = np.where(mask_test)[0]

            assert len(set(list(idx_train)).intersection(set(list(idx_test)))) == 0

            self.splits.append((idx_train, idx_test))

            assert not np.any(self.d[idx_test] != self.target_domain)

    def validate_by_concatenating(self, da_method: BaseMethod):

        results = {}

        y_target = []
        y_pred = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}

        for idx_train, idx_test in self.splits:

            X_train, X_test, y_train, y_test = self.adapt(idx_train, idx_test, da_method)

            y_target.append(y_test)

            for model_name, pipeline in CrossValidation.PIPELINES.items():

                # Train supervised model
                pipeline.fit(X_train, y_train)

                # Predict on the held-out sample
                y_pred[model_name].append(pipeline.predict_proba(X_test)[:, 1])

        y_target = np.squeeze(np.concatenate(y_target, axis=0))
        for model_name in CrossValidation.PIPELINES.keys():
            y_pred[model_name] = np.squeeze(np.concatenate(y_pred[model_name], axis=0))
            results[model_name] = CrossValidation.compute_evaluation_metrics(y_target, y_pred[model_name])
            print(f'Validation results for "{model_name}": {results[model_name]}')
        self.table.add(da_method.name(), results)

        return results

    def validate_by_averaging(self, da_method: BaseMethod):

        results_ = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}

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
                results[model_name][key] = 0
                for i in range(len(results_[model_name])):
                    results[model_name][key] += float(results_[model_name][i][key])
                results[model_name][key] /= len(results_[model_name])
            print(f'Validation results for "{model_name}": {results[model_name]}')
        self.table.add(da_method.name(), results)

        return results

    def validate(self, da_method: BaseMethod):
        if self.average_results:
            return self.validate_by_averaging(da_method)
        else:
            return self.validate_by_concatenating(da_method)
