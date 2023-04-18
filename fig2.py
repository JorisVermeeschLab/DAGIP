import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, ranksums
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from dagip.core import ot_da, transport_plan, piecewise_transport_plans
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
OUT_FOLDER = os.path.join(ROOT, 'figures')

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'HEMA', 'OV', 'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter', 'NIPT-sequencer',
        'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000'
    ],
    help='Dataset name'
)
args = parser.parse_args()
DATASET = args.dataset

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

if DATASET == 'HEMA':
    data = np.load(os.path.join(DATA_FOLDER, 'hema.npz'), allow_pickle=True)
elif DATASET == 'OV':
    data = np.load(os.path.join(DATA_FOLDER, 'ov.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(DATA_FOLDER, 'valpp.npz'), allow_pickle=True)

gc_codes = data['gc_codes']
X = data['X']
labels = data['labels']

# Remove failed samples
n_reads = np.asarray([int(x['unpaired-reads']) for x in data['metadata']], dtype=int)
mask = (n_reads > 3000000)
X, gc_codes, labels = X[mask], gc_codes[mask], labels[mask]

gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
medians = np.median(X, axis=1)
mask = (medians > 0)
X[mask] /= medians[mask, np.newaxis]

assert len(gc_codes) == len(set(gc_codes))

# Load sample pairs
idx1, idx2 = [], []
if DATASET == 'HEMA':
    for i, (label, gc_code) in enumerate(zip(labels, gc_codes)):
        pool_id = gc_code.split('-')[0]
        if label == 'GRP':
            idx2.append(i)
        elif label == 'GRP_newlib':
            idx1.append(i)
elif DATASET == 'OV':
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
else:
    print(DATASET, 'NIPT-hs2000', DATASET == 'NIPT-hs2000')
    if DATASET == 'NIPT-chemistry':
        filename = 'chemistry-validation.tsv'
    elif DATASET == 'NIPT-adapter':
        filename = 'kapa-vs-idt.tsv'
    elif DATASET == 'NIPT-lib':
        filename = 'nano-vs-kapa.tsv'
    elif DATASET == 'NIPT-hs2000':
        filename = 'hs2000-vs-novaseq.tsv'
    elif DATASET == 'NIPT-hs2500':
        filename = 'hs2500-vs-novaseq.tsv'
    elif DATASET == 'NIPT-hs4000':
        filename = 'hs4000-vs-novaseq.tsv'
    else:
        raise NotImplementedError(f'Unknown dataset "{DATASET}"')
    nipt_mapping = {}
    with open(os.path.join(DATA_FOLDER, filename), 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        line = line.rstrip()
        if len(line) > 2:
            elements = line.split()
            if (elements[0] in gc_code_dict) and (elements[1] in gc_code_dict):
               idx1.append(gc_code_dict[elements[0]])
               idx2.append(gc_code_dict[elements[1]])
idx1 = np.asarray(idx1)
idx2 = np.asarray(idx2)

X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
centromeric = ChromosomeBounds.bin_from_10kb_to_1mb(centromeric)
chrids = ChromosomeBounds.bin_from_10kb_to_1mb(chrids)

# GC correction
X_gc_corrected = gc_correction(X, gc_content)

# Center-and-scale
X_ces = np.copy(X)
X_ces[idx1, :] = RobustScaler().fit_transform(X[idx1])
X_ces[idx2, :] = RobustScaler().fit_transform(X[idx2])

# Domain adaptation
if not os.path.exists(f'{DATASET}-corrected.npy'):
    folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', DATASET)
    X_adapted = np.copy(X_gc_corrected)
    side_info = np.asarray([gc_content, mappability, centromeric, chrids]).T
    X_adapted[idx1] = ot_da(folder, X[idx1], X_adapted[idx2], side_info)
    np.save(f'{DATASET}-corrected.npy', X_adapted)
X_adapted = np.load(f'{DATASET}-corrected.npy')

# Compute optimal transport plan (after inference)
scaler = RobustScaler()
scaler.fit(X_adapted[idx2])
# gammas = piecewise_transport_plans(scaler.transform(X_adapted[idx1]), scaler.transform(X_adapted[idx2]))
gamma = transport_plan(scaler.transform(X_adapted[idx1]), scaler.transform(X_adapted[idx2]))

settings = [(0, 0, X[idx1, :], X[idx2, :], 'No correction')]
settings.append((0, 1, X_gc_corrected[idx1, :], X_gc_corrected[idx2, :], 'GC correction'))
settings.append((1, 0, X_ces[idx1, :], X_ces[idx2, :], 'Centering-scaling'))
settings.append((1, 1, X_adapted[idx1, :], X_adapted[idx2, :], 'Optimal transport'))


colors = ['darkcyan', 'darkcyan', 'darkcyan', 'darkcyan']
f, ax = plt.subplots(2, 2, figsize=(16, 8))
for kk, (i, j, X1, X2, title) in enumerate(settings):
    pvalues = np.asarray([ranksums(X1[:, k], X2[:, k]).pvalue for k in range(X.shape[1])])
    ax[i, j].scatter(gc_content, pvalues, alpha=0.1, color=colors[kk])
    ax[i, j].set_yscale('log')
    ax[i, j].set_xlim([0.32, 0.6])
    ax[i, j].set_ylim([[1e-64, 1e-64, 0.05, 1e-6][kk], 1.5 if (kk == 2) else None])
    # ax[i, j].spines[['right', 'top']].set_visible(False)
    ax[i, j].spines['right'].set_visible(False)
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
plt.savefig(os.path.join(OUT_FOLDER, f'{DATASET}-gc-bias-by-method.png'), dpi=300, transparent=True)
plt.clf()
plt.close()

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
plt.savefig(os.path.join(OUT_FOLDER, f'{DATASET}-fig2.png'))
plt.show()
