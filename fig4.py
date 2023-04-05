import os
import pickle

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
# X = gc_correction(X, mappability)


stages, t, y, d = [], [], [], []
for label, gc_code in zip(data['labels'], data['gc_codes']):
    pool_id = gc_code.split('-')[0]
    indexes, sequencer = sequencer_machine_info[pool_id]
    t.append(f'{indexes}-{sequencer}')
    d.append('_newlib' in label)
    y.append(label in {'HL', 'DLBCL', 'MM'})

    print(annotations[gc_code])
    if ';' in annotations[gc_code]:
        elements = annotations[gc_code].split(';')
        if elements[0] in {'HL', 'DLBCL', 'MM'}:
            j = 2 if (len(elements) == 3) else 1
            if elements[j] not in {'I', 'II', 'III', 'IV'}:
                stages.append('NA')
            else:
                stages.append(elements[j])
        else:
            stages.append('control')
    else:
        stages.append('control')
t = np.asarray(t, dtype=object)
y = np.asarray(y, dtype=int)
d = np.asarray(d, dtype=int)
stages = np.asarray(stages, dtype=object)

y = (y > 0).astype(int)

"""
all_stages, y_pred = [], []
for i, (train_index, test_index) in enumerate(KFold(n_splits=5, shuffle=True).split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred.append(model.predict_proba(X_test)[:, 1])
    all_stages.append(stages[test_index])
y_pred = np.concatenate(y_pred, axis=0)
all_stages = np.concatenate(all_stages, axis=0)
"""
all_stages = stages


X = StandardScaler().fit_transform(X)
model = SVC(probability=True, class_weight='balanced')
model.fit(X, y)
y_pred = model.predict_proba(X)[:, 1]
plt.subplot(2, 1, 1)
for i, label in enumerate(['control', 'I', 'II', 'III', 'IV']):
    mask = (all_stages == label)
    plt.scatter(np.full(np.sum(mask), i), y_pred[mask], alpha=0.3)

X = np.load(f'hema-rf-da.npy')
X = StandardScaler().fit_transform(X)
model = SVC(probability=True, class_weight='balanced')
model.fit(X, y)
y_pred = model.predict_proba(X)[:, 1]
plt.subplot(2, 1, 2)
for i, label in enumerate(['control', 'I', 'II', 'III', 'IV']):
    mask = (all_stages == label)
    plt.scatter(np.full(np.sum(mask), i), y_pred[mask], alpha=0.3)
plt.show()
