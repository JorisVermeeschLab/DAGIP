import argparse
import collections
import os
import json
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import ot.da
import numpy as np
import scipy.stats
import pandas as pd
import lightgbm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

from dagip.core import ot_da, DomainAdapter
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction.positive import Positive
from dagip.retraction.gip import GIPManifold
from dagip.stats.bounds import compute_theoretical_bounds
from dagip.stats.r2 import r2_coefficient
from dagip.tools.dryclean import run_dryclean


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'pairs-raw')
os.makedirs(RESULTS_FOLDER, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'OV-forward', 'NIPT-chemistry',
        'NIPT-lib', 'NIPT-adapter',
        'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000'
    ],
    help='Dataset name'
)
args = parser.parse_args()

DATASET = args.dataset


def main(METHOD: str, DATASET: str) -> None:

    # Load reference GC content
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
    gc_content = df['MEAN'].to_numpy()
    bin_chr_names = df['CHR'].to_numpy()
    bin_starts = df['START'].to_numpy()
    bin_ends = df['END'].to_numpy()

    # Load data
    if DATASET in {'OV-forward', 'OV-backward'}:
        data = np.load(os.path.join(DATA_FOLDER, 'numpy', 'OV.npz'), allow_pickle=True)
    else:
        data = np.load(os.path.join(DATA_FOLDER, 'numpy', 'NIPT.npz'), allow_pickle=True)
    gc_codes = data['gc_codes']
    gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
    paired_with = data['paired_with']
    X = data['X']
    y = data['y']
    d = data['d']

    # Remove problematic regions
    mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
    X = X[:, mask]
    gc_content = gc_content[mask]
    bin_chr_names = bin_chr_names[mask]
    bin_starts = bin_starts[mask]
    bin_ends =  bin_ends[mask]

    # Load sample pairs
    idx1, idx2, used = [], [], set()
    for i in range(len(X)):
        if paired_with[i]:
            j = gc_code_dict[paired_with[i]]
            valid = (i not in used)
            if (DATASET in {'OV-forward', 'OV-backward'}) and (y[i] != 'OV'):
                valid = False
            if (DATASET == 'NIPT-lib') and (d[i] not in {'D1a', 'D1b'}):
                valid = False
            if (DATASET == 'NIPT-adapter') and (d[i] not in {'D2a', 'D2b'}):
                valid = False
            if (DATASET == 'NIPT-hs2000') and (d[i] not in {'D3a', 'D3b'}):
                valid = False
            if (DATASET == 'NIPT-hs2500') and (d[i] not in {'D4a', 'D4b'}):
                valid = False
            if (DATASET == 'NIPT-hs4000') and (d[i] not in {'D5a', 'D5b'}):
                valid = False
            if (DATASET == 'NIPT-chemistry') and (d[i] not in {'D6a', 'D6b'}):
                valid = False
            if valid:
                if DATASET == 'OV-forward':
                    i, j = j, i
                idx1.append(i)
                idx2.append(j)
                used.add(i)
                used.add(j)
    idx1 = np.asarray(idx1, dtype=int)
    idx2 = np.asarray(idx2, dtype=int)
    idx_combined = np.concatenate((idx1, idx2), axis=0)

    # GC-correction
    if METHOD == 'baseline-loess':
        from dagip.tools.loess import loess
        for i in list(idx1) + list(idx2):
            X[i, :] /= loess(X[i, :], gc_content)
    elif METHOD == 'baseline-polynomial':
        for i in list(idx1) + list(idx2):
            X[i, :] /= np.polyval(np.polyfit(gc_content, X[i, :], 5), gc_content)
    else:
        X[idx_combined] = gc_correction(X[idx_combined], gc_content)

    print(f'Number of pairs: {len(idx1)}')


    if METHOD == 'dryclean':
        X[idx1, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends, X[idx1, :], X[idx1, :], os.path.join('tmp-dryclean', DATASET))
        X[idx2, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends, X[idx2, :], X[idx2, :], os.path.join('tmp-dryclean', DATASET))


    Y_target, Y_pred = [], []
    for train_index, test_index in KFold(n_splits=5, random_state=0xCAFE, shuffle=True).split(X[idx1, :]):

        idx1_train, idx2_train = idx1[train_index], idx2[train_index]
        idx1_test, idx2_test = idx1[test_index], idx2[test_index]

        X_adapted = np.copy(X)
        if METHOD == 'da':
            folder = os.path.join(ROOT, 'tmp', 'ot-da-tmp', DATASET)
            kwargs = {'manifold': Positive()}
            adapter = DomainAdapter(folder=folder, **kwargs)
            adapter.fit(X_adapted[idx1_train], X_adapted[idx2_train])
            X_adapted[idx1] = adapter.transform(X_adapted[idx1])
        elif METHOD == 'centering-scaling':
            target_scaler = RobustScaler()
            target_scaler.fit(X_adapted[idx2_train])
            source_scaler = RobustScaler()
            source_scaler.fit(X_adapted[idx1_train])
            X_adapted[idx1, :] = target_scaler.inverse_transform(source_scaler.transform(X_adapted[idx1, :]))
        elif METHOD == 'dryclean':
            pass
        elif METHOD == 'mapping-transport':
            model = ot.da.MappingTransport()
            model.fit(
                Xs=X_adapted[idx1_train, :],
                ys=np.zeros(len(idx1_train), dtype=int),
                Xt=X_adapted[idx2_train, :]
            )
            X_adapted[idx1, :] = model.transform(Xs=X_adapted[idx1, :])
        elif METHOD in {'baseline', 'baseline-polynomial', 'baseline-loess', 'no-correction'}:
            X_adapted = X
        else:
            raise NotImplementedError(f'Unknown correction method "{METHOD}"')

        Y_pred.append(X_adapted[idx1_test, :])
        Y_target.append(X_adapted[idx2_test, :])

    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_target = np.concatenate(Y_target, axis=0)

    np.savez(os.path.join(RESULTS_FOLDER, f'{DATASET}-{METHOD}.json.npz'), ypred=Y_pred, ytarget=Y_target)


METHODS = ['mapping-transport']
#METHODS = ['baseline', 'centering-scaling', 'dryclean', 'mapping-transport', 'da']
for method in METHODS:
    #if not os.path.exists(os.path.join(RESULTS_FOLDER, f'{DATASET}-{method}.json')):
    main(method, DATASET)
