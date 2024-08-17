import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'supervised-learning')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')


#METHODS = ['baseline', 'dryclean', 'center-and-scale', 'kmm', 'mapping-transport', 'da']
METHODS = ['baseline', 'center-and-scale', 'kmm', 'mapping-transport', 'da']
COLORS = ['#a883ef', '#f996ab', '#f35ee2', 'teal', '#f4c45a', '#b1e468']
TITLES = [
    'HL (coverage profiles)',
    'DLBCL (coverage profiles)',
    'MM (coverage profiles)',
    'OV (coverage profiles)',
    'BRCA (LFRP)',
    'BRCA (NPSP)',
    'BRCA (EMF)',
    'BRCA (FLD)'
]
#DATASETS = ['HL', 'DLBCL', 'MM', 'OV-forward', 'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions']
DATASETS = ['HL', 'DLBCL', 'MM', 'OV-forward']

print('Computation time:')
for method_name in METHODS:
    row = []
    for dataset in DATASETS:
        if not os.path.exists(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json')):
            continue
        with open(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json'), 'r') as f:
            data = json.load(f)
            data = data[method_name]
            row.append(data['extra']['computation-time'])
    metric = np.mean(row)
    print(method_name, metric)
print('')

roc = False

plt.figure(figsize=(20, 8))

all_results = {method: [] for method in METHODS}

for k, (title, dataset) in enumerate(zip(TITLES, DATASETS)):
    results = []

    ax = plt.subplot(2, 5, (k + 1) if (k < 4) else (k + 2))

    for i, method_name in enumerate(METHODS):

        if (method_name == 'dryclean') and (dataset not in {'HL', 'DLBCL', 'MM', 'OV-forward'}):
            continue

        with open(os.path.join(RESULTS_FOLDER, f'{dataset}-{method_name}.json'), 'r') as f:
            data = json.load(f)
            data = data[method_name]

        for ml_model in ['rf', 'svm', 'reglog']:
            all_results[method_name] += [x['mcc'] for x in data['supervised-learning'][ml_model]]

        y, y_hat = [], []
        for ml_model in ['rf', 'svm', 'reglog']:
            for res in data['supervised-learning'][ml_model]:
                y.append(res['y'])
                y_hat.append(res['y-pred'])
        y, y_hat = np.concatenate(y, axis=0), np.concatenate(y_hat, axis=0)

        if roc:
            fpr, tpr, _ = roc_curve(y, y_hat)
            ax.plot(fpr, tpr, label=method_name, color=COLORS[i], alpha=0.8)
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
        else:
            precision, recall, _ = precision_recall_curve(y, y_hat)
            ax.plot(recall, precision, label=method_name, color=COLORS[i], alpha=0.8)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')

        ax.set_title(title)

    ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='black')
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)

ax = plt.subplot(2, 5, 5)
handles = []
for color, method_name in zip(COLORS, METHODS):
    handles.append(mpatches.Patch(color=color, label=method_name))
ax.legend(handles=handles)
ax.set_axis_off()

plt.tight_layout()

plt.savefig(os.path.join(FIGURES_FOLDER, 'roc-curves.png'), dpi=400)
plt.show()

print('')
for method_name in METHODS:
    values = all_results[method_name]
    confint = sms.DescrStatsW(values).tconfint_mean()
    print(method_name, np.mean(values), confint)

