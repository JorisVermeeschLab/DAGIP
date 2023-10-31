import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection

from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
OUT_FOLDER = os.path.join(ROOT, 'figures')

if not os.path.isdir(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

# Load data
data = np.load(os.path.join(DATA_FOLDER, 'NIPT.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
X = data['X']
labels = data['y']
assert len(gc_codes) == len(set(gc_codes))
X /= np.median(X, axis=1)[:, np.newaxis]
plasma_sep = data['plasma_sep_delay']

mask = ~np.isnan(plasma_sep)
X = X[mask]
gc_codes = gc_codes[mask]
labels = labels[mask]
plasma_sep = plasma_sep[mask]

X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)

X_sub = X

p_values, correlations, p_values2 = [], [], []
for j in range(X_sub.shape[1]):
    if np.mean(X_sub[:, j]) > 0.001:
        res = pearsonr(X_sub[:, j], plasma_sep)
        p_values.append(res[1])
        correlations.append(res[0])
        res = spearmanr(X_sub[:, j], plasma_sep)
        p_values2.append(res[1])
p_values = np.asarray(p_values)
p_values2 = np.asarray(p_values2)

j = np.argmin(p_values)
print(j, p_values[j], correlations[j])

plt.figure(figsize=(10, 10))
ax = plt.subplot(2, 1, 1)
ax.text(0.03, 0.90, '(A)', weight='bold', fontsize=14, transform=plt.gcf().transFigure)
ax.scatter(plasma_sep, X_sub[:, j], alpha=0.7, color='grey')
m, b = np.polyfit(plasma_sep, X_sub[:, j], 1)
plt.plot(plasma_sep, plasma_sep * m + b, color='black')
for k in range(1, 12):

    ax.axvline(k*24*60, linestyle='--', color='black', alpha=0.7, linewidth=0.5)
    ax.annotate(xy=(k*24*60+20, 1.055), xytext=(k*24*60+20, 1.055), text='1 day' if (k == 1) else f'{k} days')
ax.set_xlabel('Plasma separation delay (in minutes)')
ax.set_ylabel('GC-corrected read counts')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

threshold = 0.01
print(f'Significant bins: {np.sum(p_values < threshold)} / {len(p_values)}')
print(f'Benjamini/Hochberg correction: {np.sum(fdrcorrection(p_values, 0.01, method="indep")[0])} / {len(p_values)}')
print(f'Bonferroni correction: {np.sum(p_values < threshold / len(p_values))} / {len(p_values)}')
ax = plt.subplot(2, 1, 2)
ax.text(0.03, 0.45, '(B)', weight='bold', fontsize=14, transform=plt.gcf().transFigure)
_, corrected = fdrcorrection(p_values, 0.01, method='indep')
ax.hist(np.log(p_values), color='darkslateblue', alpha=0.4, bins=200, label='Pearson')
ax.hist(np.log(p_values2), color='crimson', alpha=0.4, bins=200, label='Spearman')
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Log(p-value)')
ax.set_ylabel('Number of bins')
ax.axvline(np.log(threshold), linestyle='--', color='black', alpha=0.7, linewidth=0.5)
ax.annotate(xy=(np.log(threshold)+0.1, 90), xytext=(np.log(threshold)+0.1, 90), text='log(significance level)')
ax.axvline(np.log(threshold / len(p_values)), linestyle='--', color='black', alpha=0.7, linewidth=0.5)
ax.annotate(xy=(np.log(threshold / len(p_values))+0.1, 90), xytext=(np.log(threshold / len(p_values))+0.1, 90), text='FDR correction')
plt.tight_layout()

# plt.show()
plt.savefig(os.path.join(OUT_FOLDER, 'plasmasepdelay.png'), dpi=300, transparent=True)
