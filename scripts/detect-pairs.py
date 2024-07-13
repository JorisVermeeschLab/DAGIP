import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler

from dagip.core import ot_da, transport_plan
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction.gip import GIPManifold
from dagip.stats.bounds import compute_theoretical_bounds
from dagip.stats.r2 import r2_coefficient


DATA_FOLDER = os.path.join(ROOT, '..', 'data')


parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'OV-forward', 'OV-backward', 'NIPT-chemistry',
        'NIPT-lib', 'NIPT-adapter',
        'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000'
    ],
    help='Dataset name'
)
parser.add_argument(
    'method',
    type=str,
    choices=['none', 'centering-scaling', 'gc-correction+centering-scaling', 'gc-correction', 'domain-adaptation'],
    help='Correction method'
)
parser.add_argument(
    '--noreg',
    action='store_true',
    help='Whether to disable early stopping and regularization'
)
parser.add_argument(
    '--gamma',
    action='store_true',
    help='Whether to rank based on transport plan'
)
args = parser.parse_args()

METHOD = args.method
DATASET = args.dataset

# Load reference GC content
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
gc_content = df['MEAN'].to_numpy()

# Load data
if DATASET in {'OV-forward', 'OV-backward'}:
    data = np.load(os.path.join(DATA_FOLDER, 'numpy', 'OV.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(DATA_FOLDER, 'numpy', 'NIPT.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
paired_with = data['paired_with']
X = data['X']
y = data['y']
d = data['d']

# Remove problematic regions
mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
X = X[:, mask]
gc_content = gc_content[mask]

# Load sample pairs
idx1, idx2, used = [], [], set()
for i in range(len(X)):
    if paired_with[i]:
        j = gc_code_dict[paired_with[i]]
        valid = (i not in used)
        if (DATASET in {'OV-forward', 'OV-backward'}) and (y[i] != 'OV'):
            valid = False
        if (DATASET == 'NIPT-lib') and (d[i] not in {'D1a', 'D1b'}):
            valid = False
        if (DATASET == 'NIPT-adapter') and (d[i] not in {'D2a', 'D2b'}):
            valid = False
        if (DATASET == 'NIPT-hs2000') and (d[i] not in {'D3a', 'D3b'}):
            valid = False
        if (DATASET == 'NIPT-hs2500') and (d[i] not in {'D4a', 'D4b'}):
            valid = False
        if (DATASET == 'NIPT-hs4000') and (d[i] not in {'D5a', 'D5b'}):
            valid = False
        if (DATASET == 'NIPT-chemistry') and (d[i] not in {'D6a', 'D6b'}):
            valid = False
        if valid:
            if DATASET == 'OV-forward':
                i, j = j, i
            idx1.append(i)
            idx2.append(j)
            used.add(i)
            used.add(j)
idx1 = np.asarray(idx1, dtype=int)
idx2 = np.asarray(idx2, dtype=int)

print(f'Number of pairs: {len(idx1)}')

X_adapted = np.copy(X)
if METHOD == 'domain-adaptation':
    X_adapted[idx1, :] = gc_correction(X[idx1, :], gc_content)
    X_adapted[idx2, :] = gc_correction(X[idx2, :], gc_content)
    folder = os.path.join(ROOT, 'tmp', 'ot-da-tmp', DATASET)
    kwargs = {
        'manifold': GIPManifold(gc_content),
        'max_n_iter': 500
    }
    if args.noreg:
        kwargs['reg_rate'] = 0
        kwargs['convergence_threshold'] = 1.0
    X_adapted[idx1] = ot_da(folder, X_adapted[idx1], X_adapted[idx2], **kwargs)
elif METHOD == 'gc-correction':
    X_adapted[idx1, :] = gc_correction(X[idx1, :], gc_content)
    X_adapted[idx2, :] = gc_correction(X[idx2, :], gc_content)
elif METHOD == 'centering-scaling':
    X_adapted[idx1, :] = RobustScaler().fit_transform(X_adapted[idx1])
    X_adapted[idx2, :] = RobustScaler().fit_transform(X_adapted[idx2])
elif METHOD == 'gc-correction+centering-scaling':
    X_adapted[idx1, :] = gc_correction(X[idx1, :], gc_content)
    X_adapted[idx2, :] = gc_correction(X[idx2, :], gc_content)
    X_adapted[idx1, :] = RobustScaler().fit_transform(X_adapted[idx1])
    X_adapted[idx2, :] = RobustScaler().fit_transform(X_adapted[idx2])
elif METHOD == 'none':
    X_adapted = X
else:
    raise NotImplementedError(f'Unknown correction method "{METHOD}"')

if args.gamma:
    print('Using transport plan.')
    D = cdist(X_adapted[idx1], X_adapted[idx2])
    gamma = transport_plan(D)
    D = -gamma
else:
    D = cdist(X_adapted[idx1], X_adapted[idx2])

r2 = r2_coefficient(X_adapted[idx1], X_adapted[idx2])

correct = (np.arange(len(idx1)) == np.argmin(D, axis=0))
misassigned_pairs = []
for i in range(len(correct)):
    if not correct[i]:
        misassigned_pairs.append((gc_codes[idx1[i]], gc_codes[idx2[i]]))
# print(f'Misassigned pairs: {misassigned_pairs}')
print(f'Precision: {np.mean(correct)} ({np.sum(correct)}/{len(correct)})')
print(f'R2 coefficient: {r2}')

settings = [(0, X[idx1, :], X[idx2, :], 'No correction')]
settings.append((1, X_adapted[idx1, :], X_adapted[idx2, :], 'GC correction'))

alpha = 0.5
size = 5
f, ax = plt.subplots(1, 2, figsize=(16, 8))

for i, X1, X2, title in settings:
    pca = KernelPCA(n_components=2)
    pca.fit(np.concatenate((X1, X2), axis=0))
    coords = pca.transform(X1)
    ax[i].scatter(coords[:, 0], coords[:, 1], color='orangered', alpha=alpha, s=size)
    ax[i].set_xlabel('First principal component')
    ax[i].set_ylabel('Second principal component')
    ax[i].set_title(title)
    coords = pca.transform(X2)
    ax[i].scatter(coords[:, 0], coords[:, 1], color='green', alpha=alpha, s=size)
    if (i,) == (0,):
        plt.legend(prop={'size': 5})

# plt.savefig('pairs.png')
plt.show()
