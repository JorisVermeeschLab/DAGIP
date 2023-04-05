import collections
import os

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from dagip.ichorcna.model import safe_division
from dagip.nipt.binning import ChromosomeBounds
from dagip.tools.ichor_cna import create_ichor_cna_normal_panel, load_ichor_cna_results, ichor_cna

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

METHOD = 'rf-da'
DATASET = 'OV'
SHUFFLE = True

ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')

# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

if DATASET == 'OV':
    data = np.load(os.path.join(DATA_FOLDER, 'ov.npz'), allow_pickle=True)
else:
    data = np.load(os.path.join(DATA_FOLDER, 'valpp.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
X = data['X']
medians = np.median(X, axis=1)
mask = (medians > 0)
X[mask] /= medians[mask, np.newaxis]
X[X >= 2] = 2

# Load sample pairs
with open(os.path.join(DATA_FOLDER, 'metadata.csv'), 'r') as f:
    lines = f.readlines()[1:]
tech_mapping = {}
gc_code_mapping = {}
for line in lines:
    elements = line.rstrip().split(',')
    if len(elements) < 2:
        continue
    gc_code_mapping[elements[0]] = elements[4]
    tech_mapping[elements[0]] = '-'.join(elements[1:4])
    tech_mapping[elements[4]] = '-'.join(elements[5:8])

counter = collections.Counter(data['labels'])
print(counter)

assert len(gc_codes) == len(set(gc_codes))

y, d = [], []
for label, gc_code in zip(data['labels'], data['gc_codes']):
    d.append(gc_code.startswith('GC'))
    y.append(label != 'control')
y = np.asarray(y, dtype=int)
d = np.asarray(d, dtype=int)

# Load sample pairs
if DATASET == 'OV':
    gc_codes = data['gc_codes']
    gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
    with open(os.path.join(DATA_FOLDER, 'control-and-ov-pairs.txt'), 'r') as f:
        lines = f.readlines()[1:]
    idx1, idx2 = [], []
    for line in lines:
        elements = line.rstrip().split()
        if len(elements) > 1:
            if not ((elements[0] in gc_code_dict) and (elements[1] in gc_code_dict)):
                # print(f'Could not load sample pair {elements}')
                continue
            if elements[0].startswith('healthy_control'):  # TODO
                continue
            i = gc_code_dict[elements[0]]
            j = gc_code_dict[elements[1]]
            idx1.append(i)
            idx2.append(j)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
elif DATASET == 'NIPT':
    gc_codes_idx = {gc_code: i for i, gc_code in enumerate(gc_codes)}

    ts, idx1, idx2 = [], [], []
    for gc_code in gc_codes:
        i = gc_codes_idx[gc_code]
        pool_id = gc_code.split('-')[0]
        ts.append(tech_mapping[pool_id])
        index_id = gc_code.split('-')[1]
        if pool_id in gc_code_mapping:
            counterpart = gc_code_mapping[pool_id] + '-' + index_id
            if counterpart in gc_codes_idx:
                j = gc_codes_idx[counterpart]
                if (np.sum(X[i, :]) == 0) or (np.sum(X[j, :]) == 0):
                    continue

                idx1.append(i)
                idx2.append(j)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    ts = np.asarray(ts)
else:
    raise NotImplementedError(f'Unknown dataset "{DATASET}"')
assert len(idx1) == len(idx2)

print(f'Number of pairs: {len(idx1)}')

gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
centromeric = ChromosomeBounds.bin_from_10kb_to_1mb(centromeric).astype(int).astype(bool)
chrids = ChromosomeBounds.bin_from_10kb_to_1mb(chrids)
X = ChromosomeBounds.bin_from_10kb_to_1mb(X)

X1 = X[idx1, :]
X1 = np.copy(X1)
r1 = np.median(X1, axis=0)

i = idx1[0]


def read_counts_correction(x, gc_content, mappability):
    valid = (x > 0)
    range_ = np.quantile(x[valid], [0, 0.99])
    domain = np.quantile(gc_content[valid], [0.001, 0.999])

    ideal = np.ones(len(valid), dtype=bool)
    ideal[~valid] = False
    ideal[mappability < 0.9] = False
    ideal[np.logical_or(x <= range_[0], x > range_[1])] = False
    ideal[np.logical_or(gc_content < domain[0], gc_content > domain[1])] = False

    # GC correction
    binned = np.linspace(0, 1, 1001)
    import loess.loess_1d
    _, y_out, _ = loess.loess_1d.loess_1d(gc_content[ideal], x[ideal], xnew=binned, frac=0.03, degree=2)
    assert not np.any(np.isnan(y_out))
    _, final, _ = loess.loess_1d.loess_1d(binned, y_out, xnew=gc_content, frac=0.3, degree=2)
    assert not np.any(np.isnan(final))
    assert not np.any(final == 0)
    x = x / final

    # Mappability correction
    range_ = np.quantile(x[valid], [0, 0.99])
    ideal = np.logical_and(x > range_[0], x <= range_[1])
    x = x / lowess(mappability[ideal], x[ideal], xvals=mappability, frac=2./3.)

    return x


def preprocess(x, gc_content, mappability, centromeric):
    x = np.copy(x)
    x[~centromeric] = read_counts_correction(x[~centromeric], gc_content[~centromeric], mappability[~centromeric])
    print(x)

    return x


for i in tqdm.tqdm(idx1):
    X[i, :] = preprocess(X[i, :], gc_content, mappability, centromeric)

r1 = np.median(X[idx1, :], axis=0)
p1 = safe_division(X[idx1[0]], r1)

plt.plot(p1)
plt.show()
# print(np.log2(safe_division(X[i, :], r1)))

# Run ichorCNA on samples from domain 1
folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', )
if not os.path.isdir(folder):
    os.makedirs(folder)
normal_panel_filepath = os.path.join(folder, 'normal-panel_median.rds')
if not os.path.exists(normal_panel_filepath):
    create_ichor_cna_normal_panel(ichor_cna_location, folder, X1, gc_content, mappability)

sample_folder = os.path.join(folder, gc_codes[i])
ichor_cna(ichor_cna_location, normal_panel_filepath, sample_folder, X[i, :], gc_content, mappability)
results = load_ichor_cna_results(sample_folder)

print('Finished')
