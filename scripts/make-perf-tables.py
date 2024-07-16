import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import scipy.stats
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

from dagip.utils import LaTeXTable


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results')

settings = [
    'end-motif-frequencies', 'fragment-length-distributions', 'long-fragment-ratio-profiles', 
    'nucleosome-positioning-score-profiles', 'HL', 'DLBCL', 'MM', 'OV-forward', 'OV-backward'
]

for setting in settings:
    table = LaTeXTable()

    filepath = os.path.join(RESULTS_FOLDER, 'supervised-learning', f'{setting}.json')

    with open(filepath) as f:
        data = json.load(f)

    for key in data.keys():
        table.add(key, data[key])

    print('')
    print(setting)
    print(table)
    print('')
