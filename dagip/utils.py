# -*- coding: utf-8 -*-
#
#  utils.py
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

from typing import Dict

import numpy as np
import statsmodels.stats.api as sms


def log_(message: str, verbose: bool = True):
    if verbose:
        print(f'[DAGIP] {message}')


class LaTeXTable:

    def __init__(self, avg: bool = True):
        self.avg: bool = avg
        self.results = []

    def add(self, model_name: str, res: Dict[str, dict]):
        self.results.append((model_name, res))

    def __str__(self) -> str:

        pretty_names = {
            'baseline': 'No correction',
            'center-and-scale': 'Center-and-scale',
            'gc-correction': 'GC-correction',
            'ot-without-gc-correction': 'DA (no GC-correction)',
            'ot': 'DA'
        }

        s = ''

        # Header
        s += f'{"": <20}'
        for key in ['sensitivity', 'specificity', 'mcc', 'auroc', 'aupr']:
            s += f' & {key: <12}'
        s += ' \\\\\n'

        for model_name, res in self.results:
            print(model_name, res.keys())
            if model_name in pretty_names:
                model_name = pretty_names[model_name]
            s += f'{model_name: <20}'
            if self.avg:
                f = lambda x: f'{x:.3f}'.replace('0.', '.')
                for key in ['sensitivity-best', 'specificity-best', 'mcc-best', 'auroc', 'aupr']:
                    values = []
                    for k in ['svm']:
                        print(key, res[k].keys())
                        if key in res[k]:
                            values.append(res[k][key])
                        else:
                            values.append([])
                    values = np.mean(values, axis=0)
                    confint = sms.DescrStatsW(values).tconfint_mean()
                    mean = np.mean(values)
                    #s += f' & {f(mean)} ({f(confint[0])}, {f(confint[1])})'
                    s += f' & {f(mean) : <12}'

            else:
                f = lambda x: f'{(int(round(1000 * x)) * 0.1):.1f}'
                for k in ['reglog', 'rf', 'svm']:
                    sensitivity = f(res[k]['sensitivity-best-mean'])
                    if res[k]['sensitivity-best-mean'] >= max_values[(k, 'sensitivity-best-mean')]:
                        sensitivity = f'\\textbf{{{sensitivity}}}'
                    specificity = f(res[k]['specificity-best-mean'])
                    if res[k]['specificity-best-mean'] >= max_values[(k, 'specificity-best-mean')]:
                        specificity = f'\\textbf{{{specificity}}}'
                    mcc = f(res[k]['mcc-best-mean'])
                    if res[k]['mcc-best-mean'] >= max_values[(k, 'mcc-best-mean')]:
                        mcc = f'\\textbf{{{mcc}}}'
                    auroc = f(res[k]['auroc-mean'])
                    if res[k]['auroc-mean'] >= max_values[(k, 'auroc-mean')]:
                        auroc = f'\\textbf{{{auroc}}}'
                    aupr = f(res[k]['aupr-mean'])
                    if res[k]['aupr-mean'] >= max_values[(k, 'aupr-mean')]:
                        aupr = f'\\textbf{{{aupr}}}'
                    s += f' & {sensitivity} \\% & {specificity} \\% & {mcc} \\% & {auroc} \\% & {aupr} \\%'
            s += ' \\\\\n'

        return s
