import argparse
import collections
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from sklearn.decomposition import KernelPCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler

from dagip.core import ot_da
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction import GIPRetraction
from dagip.stats.bounds import compute_theoretical_bounds
from dagip.stats.r2 import r2_coefficient


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')


parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'OV-forward', 'OV-backward', 'NIPT-chemistry',
        'NIPT-lib', 'NIPT-adapter', 'NIPT-sequencer',
        'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000'
    ],
    help='Dataset name'
)
parser.add_argument(
    'method',
    type=str,
    choices=['none', 'centering-scaling', 'gc-correction', 'ot'],
    help='Correction method'
)
args = parser.parse_args()

METHOD = args.method
DATASET = args.dataset
SHUFFLE = False

ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

# Load data
if DATASET in {'OV-forward', 'OV-backward'}:
    data = np.load(os.path.join(DATA_FOLDER, 'OV.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(DATA_FOLDER, 'NIPT.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
paired_with = data['paired_with']
X = data['X']
X /= np.median(X, axis=1)[:, np.newaxis]

# Load sample pairs
idx1, idx2 = [], []
for i in range(len(X)):
    if paired_with[i]:
        j = gc_code_dict[paired_with[i]]
        idx1.append(i)
        idx2.append(j)
idx1 = np.asarray(idx1, dtype=int)
idx2 = np.asarray(idx2, dtype=int)

print(f'Number of pairs: {len(idx1)}')

bounds = compute_theoretical_bounds(
    ChromosomeBounds.bin_from_10kb_to_k_mb(X[idx1, :], 200),
    ChromosomeBounds.bin_from_10kb_to_k_mb(gc_content, 200)
)

print(f'Theoretical bounds: {bounds}')


gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
chrids = np.round(ChromosomeBounds.bin_from_10kb_to_1mb(chrids)).astype(int)
centromeric = (ChromosomeBounds.bin_from_10kb_to_1mb(centromeric) > 0)
X = ChromosomeBounds.bin_from_10kb_to_1mb(X)

# Shuffle data
if SHUFFLE:
    six = np.arange(len(idx2))
    np.random.shuffle(six)
    X[idx2, :] = X[idx2[six], :]

if METHOD == 'rf-da':
    X_adapted = gc_correction(X, gc_content)
    side_info = np.asarray([gc_content, mappability, centromeric, chrids]).T
    folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', DATASET)
    X_adapted[idx1] = ot_da(
        folder,
        X[idx1],
        X_adapted[idx2],
        ret=GIPRetraction(side_info[:, 0]),
        # reg_rate=0,
        # convergence_threshold=1.0,
        max_n_iter=50
    )
elif METHOD == 'gc-correction':
    X_adapted = gc_correction(X, gc_content)
elif METHOD == 'centering-scaling':
    X_adapted = np.copy(X)
    X_adapted[idx1, :] = RobustScaler().fit_transform(X_adapted[idx1])
    X_adapted[idx2, :] = RobustScaler().fit_transform(X_adapted[idx2])
elif METHOD == 'none':
    X_adapted = X
else:
    raise NotImplementedError(f'Unknown correction method "{METHOD}"')

# Inverse shuffle the data
if SHUFFLE:
    six = np.argsort(six)
    X[idx2, :] = X[idx2[six], :]
    X_adapted[idx2, :] = X_adapted[idx2[six], :]

D = cdist(X_adapted[idx1], X_adapted[idx2])
D_old = cdist(X[idx1, :], X[idx2, :])

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
