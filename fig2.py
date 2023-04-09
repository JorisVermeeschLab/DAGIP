import os
import pickle

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, ranksums
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from dagip.correction.gc import gc_correction
from dagip.ichorcna.model import IchorCNA
from dagip.nipt.binning import ChromosomeBounds
from dagip.core import ot_da

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

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

# Load detailed annotations
annotations = {}
with open(os.path.join(ROOT, DATA_FOLDER, 'Sample_Disease_Annotation_toAntoine_20211026.tsv'), 'r') as f:
    lines = f.readlines()[1:]
for line in lines:
    elements = line.rstrip().split('\t')
    if len(elements) > 1:
        annotations[elements[0]] = elements[2]

data = np.load(os.path.join(DATA_FOLDER, 'hema.npz'), allow_pickle=True)
whitelist = {'HL', 'DLBCL', 'MM', 'GRP', 'GRP_newlib'}
idx = []
for i in range(len(data['X'])):
    if data['labels'][i] in whitelist:
        idx.append(i)
idx = np.asarray(idx)
data_ = {}
for attr in ['X', 'gc_codes', 'labels', 'metadata']:
    data_[attr] = data[attr]
data = data_
for attr in ['X', 'gc_codes', 'labels', 'metadata']:
    data[attr] = data[attr][idx]

gc_codes = data['gc_codes']
X = data['X']
assert len(gc_codes) == len(set(gc_codes))

X /= np.median(X, axis=1)[:, np.newaxis]

t, y, d = [], [], []
for label, gc_code in zip(data['labels'], data['gc_codes']):
    pool_id = gc_code.split('-')[0]
    indexes, sequencer = sequencer_machine_info[pool_id]
    t.append(f'{indexes}-{sequencer}')
    d.append('_newlib' in label)
    y.append(label in {'HL', 'DLBCL', 'MM'})
t = np.asarray(t, dtype=object)
y = np.asarray(y, dtype=int)
d = np.asarray(d, dtype=int)

idx1 = np.where(np.logical_and(d == 1, y == 0))[0]
idx2 = np.where(np.logical_and(d == 0, y == 0))[0]

X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
centromeric = ChromosomeBounds.bin_from_10kb_to_1mb(centromeric)
chrids = ChromosomeBounds.bin_from_10kb_to_1mb(chrids)


def process(METHOD):
    if METHOD == 'rf-da':
        ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
        folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', 'HEMA')
        X_adapted = gc_correction(X, gc_content)
        side_info = np.asarray([gc_content, mappability, centromeric, chrids]).T
        X_adapted[idx1] = ot_da(folder, X[idx1], X_adapted[idx2], side_info)
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



for method in ['rf-da']:
    if not os.path.exists(f'hema-{method}.npy'):
        X_adapted = process(method)
        np.save(f'hema-{method}.npy', X_adapted)


"""
x1 = X[idx1, :][1]
x1_new = X_adapted[idx1, :][1]

mask = np.logical_and(np.logical_and(gc_content > 0.3, gc_content < 0.7), x1 > 0)
#mask = np.logical_and(mask, mappability > 0.9)
xs = x1[mask][::20]
ys = x1_new[mask][::20]
zs = gc_content[mask][::20]
plt.scatter(xs, ys, c=zs, alpha=1, s=0.5, cmap=plt.get_cmap('BrBG'))
plt.show()

import sys; sys.exit(0)
"""


settings = [(0, 0, X[idx1, :], X[idx2, :], 'No correction')]
X_gc_corrected = np.load(f'hema-gc-correction.npy')
settings.append((0, 1, X_gc_corrected[idx1, :], X_gc_corrected[idx2, :], 'GC correction'))
X_adapted = np.load(f'hema-centering-scaling.npy')
settings.append((1, 0, X_adapted[idx1, :], X_adapted[idx2, :], 'Centering-scaling'))
X_adapted = np.load(f'hema-rf-da.npy')
settings.append((1, 1, X_adapted[idx1, :], X_adapted[idx2, :], 'Optimal transport'))


colors = ['darkcyan', 'darkcyan', 'darkcyan', 'darkcyan']
f, ax = plt.subplots(2, 2, figsize=(16, 8))
for kk, (i, j, X1, X2, title) in enumerate(settings):
    pvalues = np.asarray([ranksums(X1[:, k], X2[:, k]).pvalue for k in range(X.shape[1])])
    ax[i, j].scatter(gc_content, pvalues, alpha=0.1, color=colors[kk])
    ax[i, j].set_yscale('log')
    ax[i, j].set_xlim([0.32, 0.6])
    ax[i, j].set_ylim([[1e-64, 1e-64, 0.05, 1e-6][kk], 1.5 if (kk == 2) else None])
    ax[i, j].spines[['right', 'top']].set_visible(False)
    if kk > 1:
        ax[i, j].set_xlabel('GC content')
    if kk % 2 == 0:
        ax[i, j].set_ylabel('Wilcoxon rank-sum p-value')
    ax[i, j].set_title(title)
    ax[i, j].axhline(y=0.5, linestyle='--', color='darkblue')

    bins = np.linspace(0.335, 0.6, 50)
    for start, end in zip(bins[:-1], bins[1:]):
        mask = np.logical_and(start <= gc_content, gc_content < end)
        if np.sum(mask) > 0:
            ax[i, j].scatter(
                0.5 * (start + end), np.exp(np.median(np.log(pvalues[mask]))),
                color='black', marker='D'
            )
plt.tight_layout()
plt.savefig('gc-bias-by-method.png', dpi=300, transparent=True)

import sys; sys.exit(0)

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
    pvalues = np.asarray([scipy.stats.ranksums(X1[:, k], X2[:, k]).pvalue for k in range(X.shape[1])])

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
i, j, X1_adapted, _, title = settings[1]
Z1_adapted = StandardScaler().fit_transform(X1_adapted)
ax[0, 5].plot([-zscore_lim, zscore_lim], [-zscore_lim, zscore_lim], color='black', linestyle='--', linewidth=0.5)
ax[0, 5].scatter(Z1.flatten()[::20], Z1_adapted.flatten()[::20], s=4, alpha=0.03, color='black')
ax[0, 5].set_xlim(-zscore_lim, zscore_lim)
ax[0, 5].set_ylim(-zscore_lim, zscore_lim)
ax[0, 5].spines['right'].set_visible(False)
ax[0, 5].spines['top'].set_visible(False)
ax[0, 5].set_title('GC-correction')
ax[0, 5].set_xlabel('Z-score (no correction)')
ax[0, 5].set_ylabel('Z-score (GC-correction)')
print('pearson GC correction', pearsonr(Z1.flatten()[::20], Z1_adapted.flatten()[::20]))

_, _, X1_adapted, _, _ = settings[3]
Z1_adapted = StandardScaler().fit_transform(X1_adapted)
ax[1, 5].plot([-zscore_lim, zscore_lim], [-zscore_lim, zscore_lim], color='black', linestyle='--', linewidth=0.5)
ax[1, 5].scatter(Z1.flatten()[::20], Z1_adapted.flatten()[::20], s=4, alpha=0.03, color='black')
ax[1, 5].set_xlim(-zscore_lim, zscore_lim)
ax[1, 5].set_ylim(-zscore_lim, zscore_lim)
ax[1, 5].spines['right'].set_visible(False)
ax[1, 5].spines['top'].set_visible(False)
ax[1, 5].set_title('Optimal transport')
ax[1, 5].set_xlabel('Z-score (no correction)')
ax[1, 5].set_ylabel('Z-score (optimal transport)')
print('pearson OT', pearsonr(Z1.flatten()[::20], Z1_adapted.flatten()[::20]))

# plt.tight_layout()
plt.savefig('fig2.png')
plt.show()
