import collections
import os
import pickle

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from dagip.correction.gc import gc_correction
from dagip.ichorcna.model import IchorCNA
from dagip.nipt.binning import ChromosomeBounds
from dagip.core import ot_da
from dagip.stats.bounds import compute_theoretical_bounds

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

data = np.load(os.path.join(DATA_FOLDER, 'ov.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
X = data['X']
medians = np.median(X, axis=1)
mask = (medians > 0)
X[mask] /= medians[mask, np.newaxis]

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

counter = collections.Counter(data['labels'])
print(counter)

assert len(gc_codes) == len(set(gc_codes))

y, d = [], []
for label, gc_code in zip(data['labels'], data['gc_codes']):
    d.append(gc_code.startswith('GC'))
    y.append(label != 'control')
y = np.asarray(y, dtype=int)
d = np.asarray(d, dtype=int)

# Load sample pairs
gc_codes_idx = {gc_code: i for i, gc_code in enumerate(gc_codes)}

gc_codes = data['gc_codes']
gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
with open(os.path.join(DATA_FOLDER, 'control-and-ov-pairs.txt'), 'r') as f:
    lines = f.readlines()[1:]
idx1, idx2 = [], []
for i, gc_code in enumerate(gc_codes):
    if gc_code.startswith('GC'):
        idx1.append(i)
    else:
        idx2.append(i)
idx1 = np.asarray(idx1)
idx2 = np.asarray(idx2)

gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
chrids = np.round(ChromosomeBounds.bin_from_10kb_to_1mb(chrids)).astype(int)
centromeric = (ChromosomeBounds.bin_from_10kb_to_1mb(centromeric) > 0)
X = ChromosomeBounds.bin_from_10kb_to_1mb(X)


def process(METHOD):
    if METHOD == 'rf-da':
        ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
        folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', 'NIPT')
        X_adapted = gc_correction(X, gc_content)
        side_info = np.asarray([gc_content, mappability, centromeric, chrids]).T
        X_adapted[idx1] = ot_da(folder, X_adapted[idx1], X_adapted[idx2], side_info)
    elif METHOD == 'rf':
        side_info = np.asarray([gc_content, mappability]).T
        X_adapted = np.copy(X)
        for i in tqdm.tqdm(range(len(X))):
            model = RandomForestRegressor(n_jobs=8)
            model.fit(side_info, X[i, :])
            x_pred = model.predict(side_info)
            mask = (x_pred > 0)
            X_adapted[i, mask] = X[i, mask] / x_pred[mask]
    elif METHOD == 'gc-correction':
        X_adapted = gc_correction(X, gc_content)
    elif METHOD == 'centering-scaling':
        X_adapted = np.empty_like(X)
        X_adapted[idx1, :] = StandardScaler().fit_transform(X[idx1])
        X_adapted[idx2, :] = StandardScaler().fit_transform(X[idx2])
    elif METHOD == 'none':
        X_adapted = X
    else:
        raise NotImplementedError(f'Unknown correction method "{METHOD}"')
    return X_adapted


for method in ['none', 'centering-scaling', 'gc-correction', 'rf-da']:
    if not os.path.exists(f'ov-{method}.npy'):
        X_adapted = process(method)
        np.save(f'ov-{method}.npy', X_adapted)


settings = [(0, 0, X[idx1, :], X[idx2, :], 'No correction')]
X_gc_corrected = np.load(f'ov-gc-correction.npy')
settings.append((0, 1, X_gc_corrected[idx1, :], X_gc_corrected[idx2, :], 'GC correction'))
X_adapted = np.load(f'ov-centering-scaling.npy')
settings.append((1, 0, X_adapted[idx1, :], X_adapted[idx2, :], 'Centering-scaling'))
X_adapted = np.load(f'ov-rf-da.npy')
settings.append((1, 1, X_adapted[idx1, :], X_adapted[idx2, :], 'Optimal transport'))

alpha = 0.5
size = 5
f, ax = plt.subplots(2, 6, figsize=(16, 8), gridspec_kw=dict(width_ratios=[6,6,2,6,2,6], hspace=0.3))

numb_fontsize = 14
ax[0, 0].text(0.07, 0.90, '(a)', fontsize=numb_fontsize, transform=plt.gcf().transFigure)
ax[0, 0].text(0.45, 0.90, '(b)', fontsize=numb_fontsize, transform=plt.gcf().transFigure)
ax[0, 0].text(0.70, 0.90, '(c)', fontsize=numb_fontsize, transform=plt.gcf().transFigure)

for i, j, X1, X2, title in settings:
    print(len(X1), len(X2))
    pca = KernelPCA(n_components=2)
    pca.fit(np.concatenate((X1, X2), axis=0))
    coords = pca.transform(X1)
    ax[i, j].scatter(coords[:, 0], coords[:, 1], color='darkgoldenrod', alpha=alpha, s=size, label=r'Healthy ($\mathcal{D}_4$)')
    ax[i, j].set_xlabel('First principal component')
    if j == 0:
        ax[i, j].set_ylabel('Second principal component')
    ax[i, j].set_title(title)
    ax[i, j].spines['right'].set_visible(False)
    ax[i, j].spines['top'].set_visible(False)
    coords = pca.transform(X2)
    ax[i, j].scatter(coords[:, 0], coords[:, 1], color='darkcyan', alpha=alpha, s=size, label=r'Healthy ($\mathcal{D}_3$)')
    if (i, j) == (0, 0):
        ax[i, j].legend(prop={'size': 8})

ax[0, 2].remove()
ax[1, 2].remove()

import scipy.stats
res = []
titles = []
for i, j, X1, X2, title in settings[::-1]:
    pvalues = []
    for k in range(X.shape[1]):
        if mappability[k] < 0.8:
            continue
        pvalues.append(scipy.stats.ranksums(X1[:, k], X2[:, k]).pvalue)
    res.append(pvalues)
    titles.append(title)
r = ax[0, 3].violinplot(res, vert=False, showmeans=True, showextrema=True)
color = 'b'
r['cbars'].set_color(color)
r['cmins'].set_color(color)
r['cmaxes'].set_color(color)
r['cmeans'].set_color(color)
for body in r['bodies']:
    body.set_color(color)
# ax[0, 3].set_xscale('log')
ax[0, 3].spines['right'].set_visible(False)
ax[0, 3].spines['top'].set_visible(False)
ax[0, 3].set_xlabel('Wilcoxon rank-sum p-value')
ax[0, 3].set_yticks(range(1, len(res) + 1), titles)
ax[0, 3].axvline(x=0.5, color='black', linestyle='--', linewidth=0.5)

r = ax[1, 3].violinplot([np.log(x) for x in res], vert=False, showmeans=True, showextrema=True)
color = 'b'
r['cbars'].set_color(color)
r['cmins'].set_color(color)
r['cmaxes'].set_color(color)
r['cmeans'].set_color(color)
for body in r['bodies']:
    body.set_color(color)
# ax[1, 3].set_xscale('log')
ax[1, 3].spines['right'].set_visible(False)
ax[1, 3].spines['top'].set_visible(False)
ax[1, 3].set_xlabel('Wilcoxon rank-sum log(p-value)')
ax[1, 3].set_yticks(range(1, len(res) + 1), titles)
ax[1, 3].axvline(x=np.log(0.5), color='black', linestyle='--', linewidth=0.5)

ax[0, 4].remove()
ax[1, 4].remove()

zscore_lim = 3
_, _, X1, _, _ = settings[0]
Z1 = StandardScaler().fit_transform(X1)
Z1 = Z1[:, mappability >= 0.8]
i, j, X1_adapted, _, title = settings[1]
Z1_adapted = StandardScaler().fit_transform(X1_adapted)
Z1_adapted = Z1_adapted[:, mappability >= 0.8]
ax[0, 5].plot([-zscore_lim, zscore_lim], [-zscore_lim, zscore_lim], color='black', linestyle='--', linewidth=0.5)
ax[0, 5].scatter(Z1.flatten()[::20], Z1_adapted.flatten()[::20], s=4, alpha=0.03, color='black')
ax[0, 5].set_xlim(-zscore_lim, zscore_lim)
ax[0, 5].set_ylim(-zscore_lim, zscore_lim)
ax[0, 5].spines['right'].set_visible(False)
ax[0, 5].spines['top'].set_visible(False)
ax[0, 5].set_title('GC-correction')
ax[0, 5].set_xlabel('Z-score (no correction)')
ax[0, 5].set_ylabel('Z-score (GC-correction)')

_, _, X1_adapted, _, _ = settings[3]
Z1_adapted = StandardScaler().fit_transform(X1_adapted)
Z1_adapted = Z1_adapted[:, mappability >= 0.8]
ax[1, 5].plot([-zscore_lim, zscore_lim], [-zscore_lim, zscore_lim], color='black', linestyle='--', linewidth=0.5)
ax[1, 5].scatter(Z1.flatten()[::20], Z1_adapted.flatten()[::20], s=4, alpha=0.03, color='black')
ax[1, 5].set_xlim(-zscore_lim, zscore_lim)
ax[1, 5].set_ylim(-zscore_lim, zscore_lim)
ax[1, 5].spines['right'].set_visible(False)
ax[1, 5].spines['top'].set_visible(False)
ax[1, 5].set_title('Optimal transport')
ax[1, 5].set_xlabel('Z-score (no correction)')
ax[1, 5].set_ylabel('Z-score (optimal transport)')

# plt.tight_layout()
plt.savefig('fig2third.png', transparent=True)
plt.show()
