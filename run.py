# -*- coding: utf-8 -*-
#
#  run.py
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

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, choices=['random-forest', 'domain-adaptation'],
                    help='Bias correction algorithm')
parser.add_argument('samples', type=str,
                    help='Path to the file describing the samples')
parser.add_argument('gc_content', type=str,
                    help='Path to the WIG file containing the reference GC content')
parser.add_argument('mappability', type=str,
                    help='Path to the WIG file containing the mappability scores')
parser.add_argument('--normalize', action='store_true', default=True,
                    help='Whether to normalise data based on the median')
args = parser.parse_args()

X, d, t = [], [], []
with open(args.samples, 'r') as f:
    for line in f.readlines():
        elements = line.rstrip().split(',')
        if len(elements) != 3:
            continue





