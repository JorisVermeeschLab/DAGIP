import argparse
import sys
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dagip.benchmark.baseline import BaselineMethod
from dagip.benchmark.gc_correction import GCCorrection
from dagip.benchmark.centering_scaling import CenteringScaling
from dagip.benchmark.ot_da import OTDomainAdaptation
from dagip.nipt.binning import ChromosomeBounds
from dagip.spatial.euclidean import EuclideanDistance
from dagip.retraction.positive import Positive
from dagip.retraction.gip import GIPManifold
from dagip.validation.k_fold import KFoldValidation


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'supervised-learning')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'HL', 'DLBCL', 'MM', 'OV-forward', 'OV-backward'
    ],
    help='Dataset name'
)
args = parser.parse_args()
DATASET = args.dataset

# Load reference GC content
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
gc_content = df['MEAN'].to_numpy()

# Load data
filename = 'OV.npz' if (DATASET in {'OV-forward', 'OV-backward'}) else 'HEMA.npz'
data = np.load(os.path.join(ROOT, DATA_FOLDER, 'numpy', filename), allow_pickle=True)
gc_codes = data['gc_codes']
gc_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
X = data['X']
y = data['y']
d = np.squeeze(LabelEncoder().fit_transform(data['d'][:, np.newaxis]))
t = data['t']
groups = np.squeeze(LabelEncoder().fit_transform(data['groups'][:, np.newaxis]))

# Define labels
if DATASET == 'OV-forward':
    y = (y == 'OV').astype(int)
elif DATASET == 'OV-backward':
    d = 1 - d
    y = (y == 'OV').astype(int)
elif DATASET == 'HEMA':
    y = (y != 'Healthy').astype(int)
else:
    mask = np.asarray([label in {'Healthy', DATASET} for label in y], dtype=bool)
    X, y, d, t, groups, gc_codes = X[mask], y[mask], d[mask], t[mask], groups[mask], gc_codes[mask]
    y = (y == DATASET).astype(int)
assert np.all(d < 2)

# Ensure there is at least one cancer case in the reference domain.
# Otherwise, swap domains.
if not np.any(y[d == 0]):
    d = 1 - d

# Remove problematic regions
mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
X = X[:, mask]
gc_content = gc_content[mask]

# Cross-validation
validation = KFoldValidation(
    X, y, d, gc_codes,
    target_domain=0, redo_binning=False, average_results=True, n_splits=5, n_repeats=30,
    groups=groups,
    gc_content=gc_content
)
ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', DATASET)
results = {}
results['baseline'] = validation.validate(BaselineMethod())
results['center-and-scale'] = validation.validate(CenteringScaling())
results['gc-correction'] = validation.validate(GCCorrection())
results['ot-without-gc-correction'] = validation.validate(
    OTDomainAdaptation(Positive(), EuclideanDistance(), 'tmp', per_label=True, gc_correction=False)
)
results['ot'] = validation.validate(
    OTDomainAdaptation(GIPManifold(gc_content), EuclideanDistance(), 'tmp', per_label=True, gc_correction=True)
)

print(validation.table)

with open(os.path.join(RESULTS_FOLDER, f'{DATASET}.json'), 'w') as f:
    json.dump(results, f)
