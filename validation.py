import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from dagip.benchmark.baseline import BaselineMethod
from dagip.benchmark.centering_scaling import CenteringScaling
from dagip.benchmark.gc_correction import GCCorrection
from dagip.benchmark.rf_da import RFDomainAdaptation
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.validation.k_fold import KFoldValidation
from dagip.validation.train_test import TrainTestValidation

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

METHOD = 'rf-da'
DATASET = 'MM'

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

if DATASET == 'OV':
    data = np.load(os.path.join(DATA_FOLDER, 'ov.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(DATA_FOLDER, 'hema.npz'), allow_pickle=True)

    if DATASET == 'MM':
        whitelist = {'MM', 'GRP', 'GRP_newlib'}
    elif DATASET == 'DLBCL':
        whitelist = {'DLBCL', 'GRP', 'GRP_newlib'}
    elif DATASET == 'HL':
        whitelist = {'HL', 'GRP', 'GRP_newlib'}
    elif DATASET == 'HEMA':
        whitelist = {'HL', 'DLBCL', 'MM', 'GRP', 'GRP_newlib'}
    else:
        raise NotImplementedError()
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

t, y, d = [], [], []
if DATASET == 'OV':

    gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
    with open(os.path.join(DATA_FOLDER, 'control-and-ov-pairs.txt'), 'r') as f:
        lines = f.readlines()[1:]
    mapping = {}
    for line in lines:
        elements = line.rstrip().split()
        if len(elements) > 1:
            if not ((elements[0] in gc_code_dict) and (elements[1] in gc_code_dict)):
                # print(f'Could not load sample pair {elements}')
                continue
            mapping[elements[0]] = elements[1]
            mapping[elements[1]] = elements[0]

    current_group = 0
    groups = {}
    for label, gc_code in zip(data['labels'], data['gc_codes']):
        if gc_code.startswith('GC'):
            pool_id = gc_code.split('-')[0]
            indexes, sequencer = sequencer_machine_info[pool_id]
            t.append(f'{indexes}-{sequencer}')
        else:
            t.append('external')
        d.append(gc_code.startswith('GC'))
        y.append(label != 'control')
        if (gc_code in mapping) and (mapping[gc_code] in groups):
            groups[gc_code] = groups[mapping[gc_code]]
        else:
            groups[gc_code] = current_group
            current_group += 1
    groups = np.asarray([groups[gc_code] for gc_code in gc_codes])
else:
    for label, gc_code in zip(data['labels'], data['gc_codes']):
        pool_id = gc_code.split('-')[0]
        indexes, sequencer = sequencer_machine_info[pool_id]
        t.append(f'{indexes}-{sequencer}')
        d.append('_newlib' in label)
        y.append(label == DATASET)
        assert DATASET in {'HL', 'MM', 'DLBCL', 'HEMA'}
    groups = None
t = np.asarray(t, dtype=object)
y = np.asarray(y, dtype=int)
d = np.asarray(d, dtype=int)

X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
chrids = np.round(ChromosomeBounds.bin_from_10kb_to_1mb(chrids)).astype(int)
centromeric = (ChromosomeBounds.bin_from_10kb_to_1mb(centromeric) > 0)

print(gc_content.shape, mappability.shape)
side_info = np.asarray([gc_content, mappability, centromeric, chrids])

validation = KFoldValidation(
    X, y, d, t, side_info, gc_codes,
    target_domain=0, redo_binning=False, average_results=False, n_splits=5, groups=groups
)
# validation = LoncoValidation(X, y, d, t, side_info, gc_codes, target_domain=0)
# validation = TrainTestValidation(X, y, d, t, side_info, gc_codes, target_domain=0, redo_binning=False)
ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
if DATASET == 'OV':
    folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', 'OV')
else:
    folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', 'HEMA')
validation.validate(BaselineMethod())
validation.validate(CenteringScaling())
validation.validate(GCCorrection())
validation.validate(RFDomainAdaptation(ichor_cna_location, folder, per_label=True))

print(validation.table)
