# -*- coding: utf-8 -*-
#
#  train_test.py
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
from dagip.utils import LaTeXTable
from dagip.validation.base import CrossValidation


class TrainTestValidation(CrossValidation):

    def __init__(self, *args, test_fraction: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_fraction: float = float(test_fraction)
        mask = np.ones(len(self.X), dtype=bool)
        mask[self.d == self.target_domain] = ~(np.random.rand(int(np.sum(self.d == self.target_domain))) < self.test_fraction)
        self.idx_train: np.ndarray = np.where(mask)[0]
        self.idx_test: np.ndarray = np.where(~mask)[0]

        # Results
        self.table: LaTeXTable = LaTeXTable()

    def validate(self, da_method: BaseMethod):

        X_train, X_test, y_train, y_test = self.adapt(self.idx_train, self.idx_test, da_method)

        results = {}
        for model_name, pipeline in CrossValidation.PIPELINES.items():
            pipeline.fit(X_train, y_train)
            y_pred = np.squeeze(pipeline.predict_proba(X_test)[:, 1])

            results[model_name] = CrossValidation.compute_evaluation_metrics(y_test, y_pred)
            print(f'Validation results for "{model_name}": {results[model_name]}')
        self.table.add(da_method.name(), results)

        return results
