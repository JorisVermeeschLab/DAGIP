import collections
import os

import numpy as np
import ot
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from dagip.correction.gc import gc_correction

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

METHOD = 'da'

df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

# Load sample pairs
with open(os.path.join(DATA_FOLDER, 'metadata.csv'), 'r') as f:
    lines = f.readlines()[1:]
tech_mapping = {}
gc_code_mapping = {}
for line in lines:
    elements = line.rstrip().split(',')
    if len(elements) < 2:
        continue
    gc_code_mapping[elements[0]] = elements[4]
    tech_mapping[elements[0]] = '-'.join(elements[1:4])
    tech_mapping[elements[4]] = '-'.join(elements[5:8])

# Load sequencer information
with open(os.path.join(ROOT, DATA_FOLDER, 'nipt_adaptor_machine.csv'), 'r') as f:
    lines = f.readlines()[1:]
sequencer_machine_info = {}
for line in lines:
    elements = line.rstrip().split(',')
    if len(elements) > 1:
        try:
            sequencer_machine_info[elements[0]] = (elements[1], elements[2])
        except NotImplementedError:
            pass

data = np.load(os.path.join(DATA_FOLDER, 'val.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
X = data['X']
labels = data['labels']

# Remove outliers
diff = np.mean((X - np.median(X, axis=0)[np.newaxis, :]) ** 2., axis=1)
idx = (diff < 3)
X = X[idx]
gc_codes = gc_codes[idx]
labels = labels[idx]

medians = np.median(X, axis=1)
mask = (medians > 0)
X[mask, :] /= medians[mask, np.newaxis]
X[X >= 2] = 2

print(X.shape)

X = gc_correction(X, gc_content)


#   np.random.shuffle(X)

print(collections.Counter(labels))
assert len(gc_codes) == len(set(gc_codes))
assert len(gc_codes) == len(X)

settings = [
    ((0, 0), 'indexes', 'manual', 'IDT'),
    ((0, 1), 'indexes', 'KAPA', 'manual'),
    ((0, 2), 'sequencer', 'HiSeq2000', 'HiSeq2500'),
    ((1, 0), 'sequencer', 'HiSeq4000', 'HiSeq2500'),
    ((1, 1), 'sequencer', 'NovaSeq1', 'HiSeq2500'),
    ((1, 2), 'sequencer', 'NovaSeq2', 'NovaSeq1'),
]

pretty_names = {
    'manual': 'Truseq Nano indexes',
    'IDT': 'IDT indexes',
    'KAPA': 'Kapa Dual Indexed Adapters',
    'HiSeq2000': 'HiSeq 2000',
    'HiSeq2500': 'HiSeq 2500',
    'HiSeq4000': 'HiSeq 4000',
    'NovaSeq1': 'NovaSeq V1',
    'NovaSeq2': 'NovaSeq V2',
}

fig, ax = plt.subplots(2, 3, figsize=(15.5, 10), constrained_layout=True)

colorbar_set = False
for data in settings:
    i, j = data[0]
    tech_id = (1 if (data[1].lower().strip() == 'sequencer') else 0)
    tech_a = data[2]
    tech_b = data[3]

    X1, X2 = [], []
    for gc_code, x in zip(gc_codes, X):
        pool_id = gc_code.split('-')[0]
        info = sequencer_machine_info[pool_id][tech_id]
        if info == tech_a:
            X1.append(x)
        elif info == tech_b:
            X2.append(x)
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    n, m = len(X1), len(X2)
    gamma = ot.emd(np.full(n, 1. / n), np.full(m, 1. / m), cdist(X1, X2))
    gamma /= gamma.sum(axis=0)[np.newaxis, :]
    X1 = gamma.T.dot(X1)

    xs = np.mean(X, axis=0)
    ys = gc_content
    zs = np.mean(X2 - X1, axis=0)
    mask = (gc_content > 0.2)
    xs, ys, zs = xs[mask], ys[mask], zs[mask]

    cmap = plt.get_cmap('BrBG')
    mappable = ax[i, j].scatter(xs, ys, c=zs, s=1, cmap=cmap)
    #ax[i, j].plot([0, 1.2], [0, 1.2], color='black')
    ax[i, j].spines['right'].set_visible(False)
    ax[i, j].spines['top'].set_visible(False)
    ax[i, j].set_xlabel('Average normalised read count')
    ax[i, j].set_ylabel('GC content')
    ax[i, j].set_title(pretty_names[tech_b] + ' / ' + pretty_names[tech_a], fontsize=12)

    if not colorbar_set:
        fig.colorbar(mappable, ax=ax[:, 2], shrink=0.6, location='right')
        colorbar_set = True

plt.savefig('gc-biases.png', dpi=200)
plt.show()
