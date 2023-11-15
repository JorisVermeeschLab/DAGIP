import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dagip.benchmark.baseline import BaselineMethod
from dagip.benchmark.centering_scaling import CenteringScaling
from dagip.benchmark.gc_correction import GCCorrection
from dagip.benchmark.rf_da import RFDomainAdaptation
from dagip.nipt.binning import ChromosomeBounds
from dagip.validation.k_fold import KFoldValidation


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'HL', 'DLBCL', 'MM', 'OV-forward', 'OV-backward'
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

# Load data
filename = 'OV.npz' if (DATASET in {'OV-forward', 'OV-backward'}) else 'HEMA.npz'
data = np.load(os.path.join(ROOT, DATA_FOLDER, filename), allow_pickle=True)
gc_codes = data['gc_codes']
gc_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
X = data['X']
X /= np.median(X, axis=1)[:, np.newaxis]
y = data['y']
d = np.squeeze(LabelEncoder().fit_transform(data['d'][:, np.newaxis]))
t = data['t']
groups = np.squeeze(LabelEncoder().fit_transform(data['groups'][:, np.newaxis]))

# Define labels
if DATASET == 'OV-forward':
    y = (y == 'OV').astype(int)
elif DATASET == 'OV-backward':
    d = 1 - d
    y = (y == 'OV').astype(int)
elif DATASET == 'HEMA':
    y = (y != 'Healthy').astype(int)
else:
    mask = np.asarray([label in {'Healthy', DATASET} for label in y], dtype=bool)
    X, y, d, t, groups, gc_codes = X[mask], y[mask], d[mask], t[mask], groups[mask], gc_codes[mask]
    y = (y == DATASET).astype(int)
assert np.all(d < 2)

# Ensure there is at least one cancer case in the reference domain.
# Otherwise, swap domains.
if not np.any(y[d == 0]):
    d = 1 - d

X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
chrids = np.round(ChromosomeBounds.bin_from_10kb_to_1mb(chrids)).astype(int)
centromeric = (ChromosomeBounds.bin_from_10kb_to_1mb(centromeric) > 0)
side_info = np.asarray([gc_content, mappability, centromeric, chrids])

validation = KFoldValidation(
    X, y, d, t, side_info, gc_codes,
    target_domain=0, redo_binning=False, average_results=True, n_splits=5, n_repeats=10,
    groups=groups
)
# validation = LoncoValidation(X, y, d, t, side_info, gc_codes, target_domain=0)
# validation = TrainTestValidation(X, y, d, t, side_info, gc_codes, target_domain=0, redo_binning=False)
ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', DATASET)
#validation.validate(BaselineMethod())
#validation.validate(CenteringScaling())
#validation.validate(GCCorrection())
validation.validate(RFDomainAdaptation(ichor_cna_location, folder, per_label=True))

print(validation.table)
