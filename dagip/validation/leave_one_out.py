# -*- coding: utf-8 -*-
#
#  leave_one_out.py
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

from dagip.benchmark.base import BaseMethod
from dagip.validation.base import CrossValidation


class LeaveOneOutValidation(CrossValidation):

    def __init__(self, *args, target_domain: int = 0):
        super().__init__(*args, target_domain=target_domain)

    def validate(self, da_method: BaseMethod):

        results = {}

        y_target = []
        y_pred = {model_name: [] for model_name in CrossValidation.PIPELINES.keys()}
        for i in np.where(self.d == self.target_domain)[0]:

            # Split data into training/held-out sets
            idx_train = np.where(np.arange(len(self.d)) != i)[0]
            idx_test = np.array([i], dtype=int)
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
