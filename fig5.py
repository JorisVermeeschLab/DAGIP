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

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

plasma_dict = {}
with open(os.path.join(DATA_FOLDER, 'GC-code_plasmasepdelay.csv'), 'r') as f:
    for line in f.readlines()[1:]:
        line = line.rstrip()
        elements = line.split(',')
        if len(elements) == 2:
            plasma_dict[elements[0]] = float(elements[1])

data = np.load(os.path.join(DATA_FOLDER, 'valpp.npz'), allow_pickle=True)

gc_codes = data['gc_codes']
X = data['X']
labels = data['labels']
assert len(gc_codes) == len(set(gc_codes))

X /= np.median(X, axis=1)[:, np.newaxis]

X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)

X_sub, plasma_sep = [], []
for i in range(len(X)):
    if gc_codes[i] in plasma_dict:
        X_sub.append(X[i])
        plasma_sep.append(plasma_dict[gc_codes[i]])
X_sub = np.asarray(X_sub)
X_sub = gc_correction(X_sub, gc_content)
plasma_sep = np.asarray(plasma_sep)

p_values, correlations, p_values2 = [], [], []
for j in range(X_sub.shape[1]):
    if np.mean(X_sub[:, j]) > 0.001:
        res = pearsonr(X_sub[:, j], plasma_sep)
        p_values.append(res.pvalue)
        correlations.append(res.statistic)
        res = spearmanr(X_sub[:, j], plasma_sep)
        p_values2.append(res.pvalue)
p_values = np.asarray(p_values)
p_values2 = np.asarray(p_values2)

j = np.argmin(p_values)
print(p_values[j], correlations[j])

plt.figure(figsize=(10, 10))
ax = plt.subplot(2, 1, 1)
ax.text(0.03, 0.90, '(A)', weight='bold', fontsize=14, transform=plt.gcf().transFigure)
ax.scatter(plasma_sep, X_sub[:, j], alpha=0.7, color='lightseagreen')
m, b = np.polyfit(plasma_sep, X_sub[:, j], 1)
plt.plot(plasma_sep, plasma_sep * m + b, color='black')
for k in range(1, 12):
    ax.axvline(k*24*60, linestyle='--', color='black', alpha=0.7, linewidth=0.5)
    ax.annotate(xy=(k*24*60+20, 1.055), xytext=(k*24*60+20, 1.055), text='1 day' if (k == 1) else f'{k} days')
ax.set_xlabel('Plasma separation delay (in minutes)')
ax.set_ylabel('GC-corrected read counts')
ax.spines[['right', 'top']].set_visible(False)

threshold = 0.01
print(f'Significant bins: {np.sum(p_values < threshold)} / {len(p_values)}')
print(f'Benjamini/Hochberg correction: {np.sum(fdrcorrection(p_values, 0.01, method="indep")[0])} / {len(p_values)}')
print(f'Bonferroni correction: {np.sum(p_values < threshold / len(p_values))} / {len(p_values)}')
ax = plt.subplot(2, 1, 2)
ax.text(0.03, 0.45, '(B)', weight='bold', fontsize=14, transform=plt.gcf().transFigure)
_, corrected = fdrcorrection(p_values, 0.01, method='indep')
ax.hist(np.log(p_values), color='royalblue', alpha=0.4, bins=200, label='Pearson')
ax.hist(np.log(p_values2), color='salmon', alpha=0.4, bins=200, label='Spearman')
ax.legend()
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Log(p-value)')
ax.set_ylabel('Number of bins')
ax.axvline(np.log(threshold), linestyle='--', color='black', alpha=0.7, linewidth=0.5)
ax.annotate(xy=(np.log(threshold)+0.1, 90), xytext=(np.log(threshold)+0.1, 90), text='log(significance level)')
ax.axvline(np.log(threshold / len(p_values)), linestyle='--', color='black', alpha=0.7, linewidth=0.5)
ax.annotate(xy=(np.log(threshold / len(p_values))+0.1, 90), xytext=(np.log(threshold / len(p_values))+0.1, 90), text='FDR correction')
plt.tight_layout()

# plt.show()
plt.savefig('plasmasepdelay.png', dpi=300, transparent=True)
