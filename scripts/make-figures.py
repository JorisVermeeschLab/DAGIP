import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import numpy as np
import scipy.stats
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

from dagip.plot import plot_end_motif_freqs, scatterplot_with_sample_importances, loo_influence_analysis
from dagip.stats.fisher import reglog_fisher_info


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)


def load_folder(folder: str) -> np.ndarray:
    X = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath)
        x = df.to_numpy()[:, -1]
        if ('end-motif-frequencies' in folder) or ('fragment-length-distributions' in folder):
            x = x / x.sum()
        X.append(x)
    return np.asarray(X)


#MODALITY = 'end-motif-frequencies'
MODALITY = 'long-fragment-ratio-profiles'


X_D12_controls_adapted = load_folder(os.path.join(RESULTS_FOLDER, 'corrected', 'D12', 'controls', MODALITY))


X, d, labels, y = [], [], [], []
X.append(load_folder(os.path.join(DATA_FOLDER, 'D11', 'cases', MODALITY)))
y += [1] * len(X[-1])
d += [0] * len(X[-1])
labels += ['D11 cases'] * len(X[-1])
X.append(load_folder(os.path.join(DATA_FOLDER, 'D11', 'controls', MODALITY)))
y += [0] * len(X[-1])
d += [0] * len(X[-1])
labels += ['D11 controls'] * len(X[-1])
X.append(load_folder(os.path.join(DATA_FOLDER, 'D12', 'controls', MODALITY)))
y += [0] * len(X[-1])
d += [1] * len(X[-1])
labels += ['D12 controls'] * len(X[-1])
X = np.concatenate(X, axis=0)
y = np.asarray(y, dtype=int)
d = np.asarray(d, dtype=object)
labels = np.asarray(labels, dtype=object)

from dagip.core import DomainAdapter
from dagip.retraction import ProbabilitySimplex

#model = DomainAdapter(manifold=ProbabilitySimplex())
#X[d == 'D12 controls'] = model.fit_transform(X[d == 'D12 controls'], X[d == 'D11 controls'])



palette = {'D11 cases': 'palevioletred', 'D11 controls': 'darkslateblue', 'D12 controls': 'teal'}
ax = plt.subplot(1, 2, 1)
scatterplot_with_sample_importances(ax, X, y, d, labels, palette)
ax = plt.subplot(1, 2, 2)
scaler = RobustScaler()
scaler.fit(X[labels == 'D11 controls'])
X[labels == 'D12 controls'] = scaler.inverse_transform(RobustScaler().fit_transform(X[labels == 'D12 controls']))
#X[labels == 'D12 controls'] = X_D12_controls_adapted
scatterplot_with_sample_importances(ax, X, y, d, labels, palette)
#plt.show()


fisher_info = loo_influence_analysis(X, y)

print(np.mean(fisher_info[labels == 'D12 controls']))


"""
# Plot end motif frequencies in D11
plt.figure(figsize=(16, 6))
ax = plt.subplot(1, 1, 1)
X = load_folder(os.path.join(DATA_FOLDER, 'D11', 'controls', 'end-motif-frequencies'))
x = np.median(X, axis=0)
plot_end_motif_freqs(ax, x)
ax.set_ylim([None, 0.02])
ax.set_title(r'Median end motif frequencies (NEBNext Enzymatic Methyl-seq)', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'end-motifs-d11.png'), dpi=300)

# Plot end motif frequencies in D12
plt.figure(figsize=(16, 6))
ax = plt.subplot(1, 1, 1)
X = load_folder(os.path.join(DATA_FOLDER, 'D12', 'controls', 'end-motif-frequencies'))
x = np.median(X, axis=0)
plot_end_motif_freqs(ax, x)
#ax.set_title(r'Median end motif frequencies ($\mathcal{D}_{12}$ controls)', fontsize=20)
ax.set_title(r'Median end motif frequencies (KAPA HyperPrep)', fontsize=20)
ax.set_ylim([None, 0.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'end-motifs-d12.png'), dpi=300)

# Plot fragment length histograms
plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
plot_params = dict(alpha=0.6)
X = load_folder(os.path.join(DATA_FOLDER, 'D11', 'cases', 'fragment-length-distributions'), lengths=True)
plt.plot(np.median(X, axis=0), label=r'Cases (NEBNext Enzymatic Methyl-seq)', color='mediumvioletred', **plot_params)
X = load_folder(os.path.join(DATA_FOLDER, 'D11', 'controls', 'fragment-length-distributions'), lengths=True)
ax.plot(np.median(X, axis=0), label=r'Controls (NEBNext Enzymatic Methyl-seq)', color='mediumslateblue', **plot_params)
X = load_folder(os.path.join(DATA_FOLDER, 'D12', 'controls', 'fragment-length-distributions'), lengths=True)
ax.plot(np.median(X, axis=0), label=r'Controls (KAPA HyperPrep)', color='goldenrod', **plot_params)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.grid(alpha=0.4, color='grey', linestyle='--', linewidth=0.5)
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('Fragment length (bp)', fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
ax.set_title('Median fragment length histograms', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'fragment-lengths.png'), dpi=300)
"""