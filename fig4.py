import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
OUT_FOLDER = os.path.join(ROOT, 'figures')


data = np.load('fractions.npz')
fb = data['before']
fa = data['after']
data = np.load('fractions-noref.npz')
fnrb = data['before']
fnra = data['after']

plt.figure(figsize=(6, 6))
ax = plt.subplot(1, 1, 1)
ax.plot([0, 0.3], [0, 0.3], linestyle='--', linewidth=0.5, color='grey', alpha=0.6)
r, p_value = pearsonr(fb, fa)
print(np.std(fb))
print(r, p_value)
ax.scatter(fb, fa, color='darkslateblue', alpha=0.4, label=f'With controls')
r, p_value = pearsonr(fnrb, fnra)
print(r, p_value)
ax.scatter(fnrb, fnra, color='crimson', alpha=0.4, label=f'Reference-free')
ax.legend()
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('No correction')
ax.set_ylabel('OT-based correction')
plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, 'tumour-fractions.png'), dpi=300)
plt.show()
