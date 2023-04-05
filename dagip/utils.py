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


def log_(message: str):
    print(f'[DAGIP] {message}')


class LaTeXTable:

    def __init__(self):
        self.s: str = ''

    def add(self, model_name: str, results: Dict[str, dict]):
        f = lambda x: f'{(int(round(1000 * x)) * 0.1):.1f}'

        self.s += model_name
        for k in ['reglog', 'rf', 'svm']:
            sensitivity = f(results[k]['sensitivity-best'])
            specificity = f(results[k]['specificity-best'])
            auroc = f(results[k]['auroc'])
            aupr = f(results[k]['aupr'])
            mcc = f(results[k]['mcc-best'])
            self.s += f' & {sensitivity} \\% & {specificity} \\% & {mcc} \\% & {auroc} \\% & {aupr} \\%'
        self.s += ' \\\\\n'

    def __str__(self) -> str:
        return self.s
