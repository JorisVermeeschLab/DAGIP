import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, ks_2samp
import scipy.spatial
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, LabelEncoder

from dagip.core import ot_da, transport_plan
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction import GIPManifold, Positive
from dagip.plot import plot_end_motif_freqs, scatterplot_with_sample_importances, loo_influence_analysis


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
OUT_FOLDER = os.path.join(ROOT, '..', 'figures')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'corrected')

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'HEMA', 'OV', 'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter',
        'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000'
    ],
    help='Dataset name'
)
args = parser.parse_args()
DATASET = args.dataset

# Load reference GC content
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
gc_content = df['MEAN'].to_numpy()

# Load data
if DATASET in {'OV-forward', 'OV-backward'}:
    filename = 'OV.npz'
elif DATASET == 'HEMA':
    filename = 'HEMA.npz'
else:
    filename = 'NIPT.npz'
data = np.load(os.path.join(ROOT, DATA_FOLDER, 'numpy', filename), allow_pickle=True)
gc_codes = data['gc_codes']
gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
paired_with = data['paired_with']
X = data['X']
labels = data['y']
d = data['d']

print('labels: ', set(labels))

# Remove problematic regions
mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
X = X[:, mask]
gc_content = gc_content[mask]

if DATASET == 'HEMA':
    idx1 = np.where(np.logical_and(d == 'D7', labels == 'Healthy'))[0]
    idx2 = np.where(np.logical_and(d == 'D8', labels == 'Healthy'))[0]
else:
    idx1, idx2 = [], []
    for i in range(len(X)):
        if paired_with[i]:
            if (DATASET == 'OV') and (labels[i] != 'OV'):
                continue
            j = gc_code_dict[paired_with[i]]
            idx1.append(i)
            idx2.append(j)
    idx1 = np.asarray(idx1, dtype=int)
    idx2 = np.asarray(idx2, dtype=int)

print(X.shape, gc_content.shape)

# GC correction
X_gc_corrected = gc_correction(X, gc_content)

# Center-and-scale
scaler = RobustScaler()
scaler.fit(X[idx1])
X_ces = scaler.transform(X)
X_ces[idx2, :] = RobustScaler().fit_transform(X[idx2])

# Domain adaptation
if not os.path.exists(os.path.join(RESULTS_FOLDER, f'{DATASET}-corrected.npy')):
    folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', DATASET)
    X_adapted = np.copy(X_gc_corrected)
    ret = Positive()  #ret = GIPManifold(gc_content)
    X_adapted[idx1] = ot_da(X_adapted[idx1], X_adapted[idx2], manifold=ret)
    np.save(os.path.join(RESULTS_FOLDER, f'{DATASET}-corrected.npy'), X_adapted)
X_adapted = np.load(os.path.join(RESULTS_FOLDER, f'{DATASET}-corrected.npy'))

#X = RobustScaler().fit_transform(X)
#X_adapted = RobustScaler().fit_transform(X_adapted)
#X_ces = RobustScaler().fit_transform(X_ces)
#X_gc_corrected = RobustScaler().fit_transform(X_gc_corrected)

settings = [(0, 0, X, 'No correction')]
settings.append((0, 1, X_gc_corrected, 'GC correction'))
settings.append((1, 0, X_ces, 'Centering-scaling'))
settings.append((1, 1, X_adapted, 'Optimal transport'))


colors = ['darkcyan', 'darkcyan', 'darkcyan', 'darkcyan']
f, ax = plt.subplots(2, 2, figsize=(16, 8))
for kk, (i, j, X_corrected, title) in enumerate(settings):
    X1 = X_corrected[idx1, :]
    X2 = X_corrected[idx2, :]
    pvalues = np.asarray([ks_2samp(X1[:, k], X2[:, k]).pvalue for k in range(X.shape[1])])
    ax[i, j].scatter(gc_content, pvalues, alpha=0.1, color=colors[kk])
    ax[i, j].set_yscale('log')
    ax[i, j].set_xlim([0.32, 0.6])
    ax[i, j].set_ylim([[1e-64, 1e-64, 0.05, 1e-6][kk], 1.5 if (kk == 2) else None])
    # ax[i, j].spines[['right', 'top']].set_visible(False)
    ax[i, j].spines['right'].set_visible(False)
    if kk > 1:
        ax[i, j].set_xlabel('GC content')
    if kk % 2 == 0:
        ax[i, j].set_ylabel('Two-sample KS p-value')
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
if not os.path.isdir(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)
plt.savefig(os.path.join(OUT_FOLDER, f'{DATASET}-gc-bias-by-method.png'), dpi=150, transparent=True)
plt.clf()
plt.close()

alpha = 0.5
size = 5
f, ax = plt.subplots(2, 2, figsize=(16, 8))

numb_fontsize = 14

if DATASET == 'HEMA':
    d1, d2 = '7', '8'
elif DATASET == 'OV':
    d1, d2 = '9', '10'
elif DATASET == 'NIPT-chemistry':
    d1, d2 = '6,a', '6,b'
elif DATASET == 'NIPT-lib':
    d1, d2 = '1,a', '1,b'
elif DATASET == 'NIPT-adapter':
    d1, d2 = '2,a', '2,b'
elif DATASET == 'NIPT-hs2000':
    d1, d2 = '3,a', '3,b'
elif DATASET == 'NIPT-hs2500':
    d1, d2 = '4,a', '4,b'
elif DATASET == 'NIPT-hs4000':
    d1, d2 = '5,a', '5,b'
else:
    raise NotImplementedError(f'Unknown dataset "{DATASET}"')
for i, j, X_corrected, title in settings:
    X1 = X_corrected[idx1, :]
    X2 = X_corrected[idx2, :]

    #transformer = KernelPCA(n_components=2)
    transformer = TSNE(early_exaggeration=80, perplexity=7)
    coords = transformer.fit_transform(X_corrected)

    palette = {}
    combined_labels = np.asarray([label + str(domain) for label, domain in zip(labels, d)], dtype=object)
    scatterplot_with_sample_importances(ax[i, j], X_corrected, labels != 'Healthy', d, combined_labels, palette)
    ax[i, j].set_xlabel('First principal component')
    if j == 0:
        ax[i, j].set_ylabel('Second principal component')
    ax[i, j].set_title(title)
    ax[i, j].spines['right'].set_visible(False)
    ax[i, j].spines['top'].set_visible(False)
    if (i, j) == (0, 0):
        ax[i, j].legend(prop={'size': 8})

plt.show()
import sys; sys.exit(0)



import scipy.stats
res = []
titles = []
for i, j, X_corrected, title in settings[::-1]:
    X1 = X_corrected[idx1, :]
    X2 = X_corrected[idx2, :]

    pvalues = np.asarray([scipy.stats.ks_2samp(X1[:, k], X2[:, k]).pvalue for k in range(X.shape[1])])
    print(title, pvalues)

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
ax[0, 3].set_xlabel('Two-sample KS p-value')
print(titles)
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
ax[1, 3].set_xlabel('Two-sample KS log(p-value)')
ax[1, 3].set_yticks(range(1, len(res) + 1), titles)
ax[1, 3].axvline(x=np.log(0.5), color='black', linestyle='--', linewidth=0.5)

ax[0, 4].remove()
ax[1, 4].remove()

Z1 = RobustScaler().fit_transform(X[idx1, :])
Z1_corrected = RobustScaler().fit_transform(X_gc_corrected[idx1, :])
Z1_adapted = RobustScaler().fit_transform(X_adapted[idx1, :])

zscore_lim = 3
ax[0, 5].plot([-zscore_lim, zscore_lim], [-zscore_lim, zscore_lim], color='black', linestyle='--', linewidth=0.5)
ax[0, 5].scatter(Z1_corrected.flatten()[::20], Z1_adapted.flatten()[::20], s=4, alpha=0.03, color='black')
ax[0, 5].set_xlim(-zscore_lim, zscore_lim)
ax[0, 5].set_ylim(-zscore_lim, zscore_lim)
ax[0, 5].spines['right'].set_visible(False)
ax[0, 5].spines['top'].set_visible(False)
r = pearsonr(Z1_corrected.flatten()[::20], Z1_adapted.flatten()[::20])[0]
ax[0, 5].set_title(f'Pearson r = {r:.3f}')
ax[0, 5].set_xlabel('Z-score (GC-correction)')
ax[0, 5].set_ylabel('Z-score (optimal transport)')
print('pearson GC correction', r)

# Compute optimal transport plan (after inference)
scaler = RobustScaler()
scaler.fit(X_adapted[idx2])
distances = scipy.spatial.distance.cdist(scaler.transform(X_adapted[idx1]), scaler.transform(X_adapted[idx2]))
gamma = transport_plan(distances)
unique, counts = np.unique(np.sum(gamma > 1e-5, axis=0), return_counts=True)
ax[1, 5].bar(
    unique - 0.15, counts, color='darkgoldenrod', width=0.3,
    label=r'$\mathcal{D}_{' + str(d1) + r'}$' + r' $\rightarrow$ ' + r'$\mathcal{D}_{' + str(d2) + r'}$'
)
print({u: c for u, c in zip(unique, counts)})
unique2, counts = np.unique(np.sum(gamma > 1e-5, axis=1), return_counts=True)
ax[1, 5].bar(
    unique2 + 0.15, counts, color='darkcyan', width=0.3,
    label=r'$\mathcal{D}_{' + str(d2) + r'}$' + r' $\rightarrow$ ' + r'$\mathcal{D}_{' + str(d1) + r'}$'
)
print({u: c for u, c in zip(unique2, counts)})
ax[1, 5].legend()
ax[1, 5].set_xticks(range(1, max(max(unique), max(unique2)) + 1), range(1, max(max(unique), max(unique2)) + 1))
ax[1, 5].set_xlabel(r'Number of similar patients in source domain')
ax[1, 5].set_ylabel(r'Patients in target domain')
ax[1, 5].spines['right'].set_visible(False)
ax[1, 5].spines['top'].set_visible(False)

# plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, f'{DATASET}-fig2.png'), dpi=150)
# plt.show()
