import os
import numpy as np
import scipy.stats
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)


def load_folder(folder: str, lengths: bool = False) -> np.ndarray:
    X = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath)
        x = df['Count'].to_numpy()
        if lengths:
            x[:40] = 0
        X.append(x / x.sum())
    return np.asarray(X)


def plot_end_motif_freqs(ax, x: np.ndarray):
    xs, ys = np.arange(len(x)) + 0.5, x
    plt.bar(xs[:64], ys[:64], color='royalblue')
    plt.bar(xs[64:128], ys[64:128], color='gold')
    plt.bar(xs[128:192], ys[128:192], color='coral')
    plt.bar(xs[192:], ys[192:], color='mediumseagreen')
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    ax.set_xticks(range(0, 257, 16), minor=True)
    ax.set_xticklabels(['' for _ in range(17)], minor=True)
    ax.set_xticks(np.arange(0, 256, 16) + 8, minor=False)
    labels = ['AANN', 'ATNN', 'ACNN', 'AGNN', 'TANN', 'TTNN', 'TCNN', 'TGNN', 'CANN', 'CTNN', 'CCNN', 'CGNN', 'GANN', 'GTNN', 'GCNN', 'GGNN']
    ax.set_xticklabels(labels, minor=False)
    ax.grid(which='minor', alpha=0.8, linestyle='--', linewidth=0.8, color='black')
    plt.axhline(y=0, linestyle='--', color='black', linewidth=0.5)
    ax.set_ylabel('5\' end-motif frequency', fontsize=15)


X, y = [], []
X.append(load_folder(os.path.join(DATA_FOLDER, 'D11', 'cases', 'end-motif-frequencies')))
y += [0.0] * len(X[-1])
X.append(load_folder(os.path.join(DATA_FOLDER, 'D11', 'controls', 'end-motif-frequencies')))
y += [0.4] * len(X[-1])
X.append(load_folder(os.path.join(DATA_FOLDER, 'D12', 'controls', 'end-motif-frequencies')))
y += [0.8] * len(X[-1])
X = np.concatenate(X, axis=0)
coords = TSNE().fit_transform(X)

seaborn.set_theme(style='whitegrid')
g = seaborn.relplot(
    x=coords[:, 0],
    y=coords[:, 1],
    hue=y,
    size=np.random.rand(len(coords)),
    palette=seaborn.cubehelix_palette(rot=-0.2, as_cmap=True),
)
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)
plt.show()


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