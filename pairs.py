import argparse
import collections
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler

from dagip.core import ot_da
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.stats.bounds import compute_theoretical_bounds

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')


parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=['OV', 'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter', 'NIPT-sequencer'],
    help='Dataset name'
)
parser.add_argument(
    'method',
    type=str,
    choices=['none', 'centering-scaling', 'gc-correction', 'rf-da'],
    help='Correction method'
)
args = parser.parse_args()

METHOD = args.method
DATASET = args.dataset
SHUFFLE = False

NIPT_SUB_DOMAIN_POOL_IDS = [

    # manual Truseq Nano Kit,Truseq Nano indexes,HS4000
    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,HS4000
    [
        ('GC060381', 'GC060477'),
        ('GC060387', 'GC060453'),
        ('GC060393', 'GC060435'),
        ('GC060399', 'GC060441')
    ],

    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,HS4000,
    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,NovaSeq
    [
        ('GC075804', 'GC076241'),
        ('GC075810', 'GC076241'),
        ('GC075816', 'GC076241'),
        ('GC075822', 'GC076241')
    ],

    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,HS2000
    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,NovaSeq
    [
        ('GC075867', 'GC076247'),
        ('GC075873', 'GC076247'),
        ('GC075867', 'GC078075')
    ],

    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,HS2500
    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,NovaSeq
    [
        ('GC075879', 'GC076247'),
        ('GC075885', 'GC076247'),
        ('GC075885', 'GC078228')
    ],

    # Hamilton - Kapa HyperPrep kit,Kapa Dual Indexed Adapters,HS4000
    # Hamilton - Kapa HyperPrep kit,IDT indexes,HS4000
    [
        ('GC076265', 'GC090342'),
        ('GC076271', 'GC090348'),
        ('GC076277', 'GC090354'),
        ('GC076283', 'GC090360'),
        ('GC087675', 'GC090592'),
        ('GC087681', 'GC090598'),
        ('GC087687', 'GC090604'),
        ('GC087693', 'GC090610')
    ],

    # Hamilton - Kapa HyperPrep kit,IDT indexes,NovaSeq V1 chemistry
    # identical pool was sequenced with V1 + V1.5 standard/custom chemistry,,NovaSeq V1.5 standard recipe
    [
        ('GC102740', 'GC102770'),
        ('GC102746', 'GC102782'),
        ('GC102752', 'GC102794'),
        ('GC104274', 'GC107158'),
        ('GC104280', 'GC107161'),
        ('GC104412', 'GC107164')
    ],

    # Hamilton - Kapa HyperPrep kit,IDT indexes,NovaSeq V1 chemistry
    # identical pool was sequenced with V1 + V1.5 standard/custom chemistry,,NovaSeq V1.5 custom recipe
    [
        ('GC102740', 'GC102764'),
        ('GC102746', 'GC102776'),
        ('GC102752', 'GC102788'),
        ('GC104274', 'GC107157'),
        ('GC104280', 'GC107160'),
        ('GC104412', 'GC107163')
    ]
]

ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

if DATASET == 'OV':
    data = np.load(os.path.join(DATA_FOLDER, 'ov.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(DATA_FOLDER, 'valpp.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
X = data['X']
medians = np.median(X, axis=1)
mask = (medians > 0)
X[mask] /= medians[mask, np.newaxis]

assert len(gc_codes) == len(set(gc_codes))

"""
pools1 = ['GC102740', 'GC102746', 'GC102752', 'GC104274', 'GC104280', 'GC104412', 'GC102740',
    'GC102746', 'GC102752', 'GC104274', 'GC104280', 'GC104412']
pools2 = ['GC102770', 'GC102782', 'GC102794', 'GC107158', 'GC107161', 'GC107164',
    'GC102764', 'GC102776', 'GC102788', 'GC107157', 'GC107160', 'GC107163']

for gc_code in gc_codes:
    pool_id = gc_code.split('-')[0]
    bar_code = gc_code.split('-')[1]
    if pool_id in pools1:
        i = pools1.index(pool_id)
        print(gc_code, f'{pools2[i]}-{bar_code}')

import sys; sys.exit(0)
"""

# Load sample pairs
idx1, idx2 = [], []
if DATASET == 'OV':
    with open(os.path.join(DATA_FOLDER, 'control-and-ov-pairs.txt'), 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        elements = line.rstrip().split()
        if len(elements) > 1:
            if not ((elements[0] in gc_code_dict) and (elements[1] in gc_code_dict)):
                # print(f'Could not load sample pair {elements}')
                continue
            if elements[0].startswith('healthy_control'):  # TODO
                continue
            i = gc_code_dict[elements[0]]
            j = gc_code_dict[elements[1]]
            idx1.append(i)
            idx2.append(j)
elif DATASET == 'NIPT':

    bar_codes = set([gc_code.split('-')[1] for gc_code in gc_codes])

    for pool_id1, pool_id2 in NIPT_SUB_DOMAIN_POOL_IDS[NIPT_SUB_DOMAIN]:
        for bar_code in bar_codes:
            gc_code1 = f'{pool_id1}-{bar_code}'
            gc_code2 = f'{pool_id2}-{bar_code}'
            if (gc_code1 in gc_code_dict) and (gc_code2 in gc_code_dict):
                i = gc_code_dict[gc_code1]
                j = gc_code_dict[gc_code2]
                idx1.append(i)
                idx2.append(j)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
elif DATASET == 'NIPT-lib':
    nipt_mapping = {}
    with open(os.path.join(DATA_FOLDER, 'nano-vs-kapa.tsv'), 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        line = line.rstrip()
        if len(line) > 2:
            elements = line.split()
            if (elements[0] in gc_code_dict) and (elements[1] in gc_code_dict):
               idx1.append(gc_code_dict[elements[0]])
               idx2.append(gc_code_dict[elements[1]])
elif DATASET == 'NIPT-adapter':
    nipt_mapping = {}
    with open(os.path.join(DATA_FOLDER, 'kapa-vs-idt.tsv'), 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        line = line.rstrip()
        if len(line) > 2:
            elements = line.split()
            if (elements[0] in gc_code_dict) and (elements[1] in gc_code_dict):
               idx1.append(gc_code_dict[elements[0]])
               idx2.append(gc_code_dict[elements[1]])
elif DATASET == 'NIPT-chemistry':
    nipt_mapping = {}
    with open(os.path.join(DATA_FOLDER, 'chemistry-validation.tsv'), 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        line = line.rstrip()
        if len(line) > 2:
            elements = line.split()
            if (elements[0] in gc_code_dict) and (elements[1] in gc_code_dict):
               idx1.append(gc_code_dict[elements[0]])
               idx2.append(gc_code_dict[elements[1]])
else:
    raise NotImplementedError(f'Unknown dataset "{DATASET}"')
idx1 = np.asarray(idx1)
idx2 = np.asarray(idx2)
assert len(idx1) == len(idx2)

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
        ichor_cna_location, folder, gc_codes[idx1], X[idx1], X_adapted[idx2], side_info,
        convergence_threshold=1.0,
        reg_rate=0,
        max_n_iter=100
    )
elif METHOD == 'gc-correction':
    X_adapted = gc_correction(X, gc_content)
elif METHOD == 'centering-scaling':
    X_adapted = np.empty_like(X)
    X_adapted[idx1, :] = RobustScaler().fit_transform(X[idx1])
    X_adapted[idx2, :] = RobustScaler().fit_transform(X[idx2])
elif METHOD == 'quantiles':
    X_adapted = QuantileTransformer(output_distribution='normal').fit_transform(X)
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

ss_res = np.sum((X_adapted[idx1] - X_adapted[idx2]) ** 2.)
ss_tot = np.sum((X_adapted[idx2] - np.mean(X_adapted[idx2], axis=0)[np.newaxis, :]) ** 2.)
r2 = 1. - ss_res / ss_tot

correct = np.arange(len(idx1)) == np.argmin(D, axis=0)
misassigned_pairs = []
for i in range(len(correct)):
    if not correct[i]:
        misassigned_pairs.append((gc_codes[idx1[i]], gc_codes[idx2[i]]))
print(f'Misassigned pairs: {misassigned_pairs}')
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

plt.savefig('pairs.png')
plt.show()
