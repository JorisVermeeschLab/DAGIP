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


def log_(message: str, verbose: bool = True):
    if verbose:
        print(f'[DAGIP] {message}')


class LaTeXTable:

    def __init__(self):
        self.results = []

    def add(self, model_name: str, res: Dict[str, dict]):
        self.results.append((model_name, res))

    def __str__(self) -> str:
        f = lambda x: f'{(int(round(1000 * x)) * 0.1):.1f}'

        max_values = {}
        for model_name, res in self.results:
            for k in res.keys():
                for metric in ['sensitivity-best', 'specificity-best', 'mcc-best', 'auroc', 'aupr']:
                    if (k, metric) not in max_values:
                        max_values[(k, metric)] = res[k][metric]
                    max_values[(k, metric)] = max(max_values[(k, metric)], res[k][metric])

        pretty_names = {
            'baseline': 'No correction',
            'center-and-scale': 'Center-and-scale',
            'gc-correction': 'GC-correction',
            'ot': 'Domain adaptation'
        }

        s = ''
        for model_name, res in self.results:
            if model_name in pretty_names:
                model_name = pretty_names[model_name]
            s += model_name
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
