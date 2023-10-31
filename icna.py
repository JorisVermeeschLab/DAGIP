import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler

from dagip.core import ot_da
from dagip.correction.gc import gc_correction
from dagip.ichorcna.metrics import ploidy_accuracy, cna_accuracy, sov_refine, absolute_error
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction import GIPRetraction
from dagip.tools.ichor_cna import ichor_cna, create_ichor_cna_normal_panel, load_ichor_cna_results


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')

CORRECTION = False
REFERENCE_FREE = True
BACKWARD = False


# Load reference GC content and mappability
mappability = np.load(os.path.join(DATA_FOLDER, 'mappability.npy'))
centromeric = np.load(os.path.join(DATA_FOLDER, 'centromeric.npy'))
chrids = np.load(os.path.join(DATA_FOLDER, 'chrids.npy'))
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc.hg38.partition.10000.tsv'), sep='\t')
gc_content = df['GC.CONTENT'].to_numpy()

# Load data
data = np.load(os.path.join(DATA_FOLDER, 'OV.npz'), allow_pickle=True)
gc_codes = data['gc_codes']
X = data['X']
X /= np.median(X, axis=1)[:, np.newaxis]
y = (data['y'] == 'OV').astype(int)
t = data['t']
d = (data['d'] == 'D9').astype(int)

# Binning: 10 kb -> 1 mb
X = ChromosomeBounds.bin_from_10kb_to_1mb(X)
gc_content = ChromosomeBounds.bin_from_10kb_to_1mb(gc_content)
mappability = ChromosomeBounds.bin_from_10kb_to_1mb(mappability)
chrids = np.round(ChromosomeBounds.bin_from_10kb_to_1mb(chrids)).astype(int)
centromeric = (ChromosomeBounds.bin_from_10kb_to_1mb(centromeric) > 0)

# Create normal panel using samples from domain 1
ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
folder = os.path.join(ROOT, 'ichor-cna-results', f'domain-1',
                      'ov-backward' if BACKWARD else 'ov-forward', 'normal-panel')
normal_panel_filepath = os.path.join(folder, 'normal-panel_median.rds')
if not os.path.exists(normal_panel_filepath):
    idx = np.where(np.logical_and(y == 0, d == 1))[0]
    create_ichor_cna_normal_panel(ichor_cna_location, folder, X[idx, :], gc_content, mappability)

# Call CNAs in cancer samples from domain 1
log_r, fractions, cna, ploidy, prevalence, subclonal = [], [], [], [], [], []
for i in np.where(np.logical_and(y == 1, d == 1))[0]:
    folder = os.path.join(
        'ichor-cna-results',
        'domain-1' if (not REFERENCE_FREE) else 'domain-1-noref',
        'ov-backward' if BACKWARD else 'ov-forward',
        gc_codes[i]
    )
    results = load_ichor_cna_results(folder)
    if not results['success']:
        ichor_cna(
            ichor_cna_location,
            None if REFERENCE_FREE else normal_panel_filepath,
            folder,
            X[i, :],
            gc_content,
            mappability
        )
        results = load_ichor_cna_results(folder)
    log_r.append(results['log-r'])
    cna.append(results['copy-number'])
    fractions.append(results['tumor-fraction'])
    ploidy.append(results['tumor-ploidy'])
    prevalence.append(results['tumor-cellular-prevalence'])
    subclonal.append(results['proportion-subclonal-cnas'])
cna = np.asarray(cna)
fractions = np.asarray(fractions)
ploidy = np.asarray(ploidy)
prevalence = np.asarray(prevalence)
subclonal = np.asarray(subclonal)
log_r = np.asarray(log_r)

# Domain adaptation
log_r_adapted = {}
cna_adapted = {}
fractions_adapted = {}
ploidy_adapted = {}
prevalence_adapted = {}
subclonal_adapted = {}
for METHOD in ['rf-da']:
    idx1 = np.where(np.logical_and(y == 1, d == 1))[0]
    idx2 = np.where(np.logical_and(y == 1, d == 0))[0]
    if CORRECTION:
        if METHOD == 'rf-da':
            ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
            folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', 'icna')
            X_adapted = gc_correction(X, gc_content)
            side_info = np.asarray([gc_content, mappability, centromeric, chrids]).T
            ret = GIPRetraction(side_info[:, 0])
            X_adapted[idx1] = ot_da(
                folder, X[idx1], X_adapted[idx2], ret=ret
            )
        elif METHOD == 'gc-correction':
            X_adapted = gc_correction(X, gc_content)
        elif METHOD == 'centering-scaling':
            X_adapted = np.copy(X)
            scaler2 = RobustScaler()
            scaler2.fit(X[idx2])
            X_adapted[idx1, :] = np.maximum(0, scaler2.inverse_transform(RobustScaler().fit_transform(X[idx1])))
        elif METHOD == 'none':
            X_adapted = X
        else:
            raise NotImplementedError(f'Unknown correction method "{METHOD}"')
    else:
        X_adapted = X

    # Create normal panel using samples from domain i
    ichor_cna_location = os.path.join(ROOT, 'ichorCNA-master')
    folder = os.path.join(
        ROOT,
        'ichor-cna-results',
        METHOD if (not REFERENCE_FREE) else f'{METHOD}-noref',
        'ov-backward' if BACKWARD else 'ov-forward',
        'normal-panel'
    )
    normal_panel_filepath = os.path.join(folder, 'normal-panel_median.rds')
    if not os.path.exists(normal_panel_filepath):
        idx = np.where(np.logical_and(y == 0, d == 0))[0]
        create_ichor_cna_normal_panel(
            ichor_cna_location,
            folder,
            X_adapted[idx, :],
            gc_content,
            mappability
        )

    # Call CNAs in cancer samples from domain i
    log_r_adapted[METHOD] = []
    cna_adapted[METHOD] = []
    fractions_adapted[METHOD] = []
    ploidy_adapted[METHOD] = []
    prevalence_adapted[METHOD] = []
    subclonal_adapted[METHOD] = []
    for i in np.where(np.logical_and(y == 1, d == 1))[0]:
        folder = os.path.join(
            'ichor-cna-results',
            METHOD if (not REFERENCE_FREE) else f'{METHOD}-noref',
            'ov-backward' if BACKWARD else 'ov-forward',
            gc_codes[i]
        )
        results = load_ichor_cna_results(folder)
        if not results['success']:
            ichor_cna(
                ichor_cna_location,
                None if REFERENCE_FREE else normal_panel_filepath,
                folder,
                X_adapted[i, :],
                gc_content,
                mappability
            )
            results = load_ichor_cna_results(folder)

        log_r_adapted[METHOD].append(results['log-r'])
        cna_adapted[METHOD].append(results['copy-number'])

        fractions_adapted[METHOD].append(results['tumor-fraction'])
        ploidy_adapted[METHOD].append(results['tumor-ploidy'])
        prevalence_adapted[METHOD].append(results['tumor-cellular-prevalence'])
        subclonal_adapted[METHOD].append(results['proportion-subclonal-cnas'])

    cna_adapted[METHOD] = np.asarray(cna_adapted[METHOD])
    fractions_adapted[METHOD] = np.asarray(fractions_adapted[METHOD])
    prevalence_adapted[METHOD] = np.asarray(prevalence_adapted[METHOD])
    subclonal_adapted[METHOD] = np.asarray(subclonal_adapted[METHOD])
    log_r_adapted[METHOD] = np.asarray(log_r_adapted[METHOD])

    f = lambda x: f'{(int(round(1000 * x)) * 0.1):.1f}'

    print('TODO', np.mean([isinstance(x, np.ndarray) for x in cna]))

    print(METHOD)
    print('Accuracy (ploidy)', f(np.mean(ploidy_accuracy(cna, cna_adapted[METHOD]))))
    print('Accuracy (CNA)', f(np.mean(cna_accuracy(cna, cna_adapted[METHOD]))))
    print('SOV_REFINE', np.mean(sov_refine(cna, cna_adapted[METHOD])))
    print('Error on log ratios', np.mean(absolute_error(log_r, log_r_adapted[METHOD])))
    print('Error on tumour fraction', np.mean(absolute_error(fractions, fractions_adapted[METHOD])))
    print('Error on tumour ploidy', np.mean(absolute_error(ploidy, ploidy_adapted[METHOD])))
    print('Error on cellular prevalence', np.mean(absolute_error(prevalence, prevalence_adapted[METHOD])))
    print('Error on proportion of subclonal CNAs', np.mean(absolute_error(subclonal, subclonal_adapted[METHOD])))
    print('')


np.savez('fractions-noref.npz', before=fractions, after=fractions_adapted['rf-da'])
plt.scatter(fractions, fractions_adapted['rf-da'])
plt.show()

# --------------------------------------------
# Make figures
# --------------------------------------------

def cn_state_to_color(state):
    if state == 0:
        return 'green', 0.6
    elif state == 1:
        return 'green', 0.4
    elif state == 2:
        return 'white', 0
    elif state == 3:
        return 'red', 0.3
    elif state == 4:
        return 'red', 0.5
    elif state >= 4:
        return 'red', 0.6
    else:
        assert state == -1
        return 'white', 0

# Initialize circos sectors
from pycirclize import Circos

bounds = ChromosomeBounds.get_1mb()


def circle_plot(method_name: str, sample_id: int):

    sectors = {f'chr {c + 1}': bounds[c + 1] - bounds[c] for c in range(22)}
    circos = Circos(sectors, space=4)

    for c, sector in enumerate(circos.sectors):

        # Plot sector axis & name text
        sector.axis(fc='none', ls='-', lw=0.5, ec='black', alpha=0.5)
        sector.text(sector.name, size=9)

        track1 = sector.add_track((70, 100))
        # track1.axis(fc="tomato", alpha=0.5)
        states = cna_adapted[method_name][sample_id][bounds[c]:bounds[c+1]]
        ys = log_r_adapted[method_name][sample_id][bounds[c]:bounds[c+1]]
        xs = np.arange(len(ys))
        # track1.line(xs, ys)
        for i in range(len(ys)):
            x1, x2 = i, i + 1
            color, alpha = cn_state_to_color(states[i])
            track1.rect(x1, x2, lw=0, color=color, alpha=alpha)

        track2 = sector.add_track((60, 70))
        xs = np.arange(len(ys))
        ys = np.zeros(len(ys))
        track2.line(xs, ys, color='black', lw=0.5, alpha=0.5)

        # Set Track03 (Radius: 15 - 40)
        track3 = sector.add_track((30, 60))
        # track3.axis(fc="lime", alpha=0.5)
        states = cna[sample_id][bounds[c]:bounds[c+1]]
        ys = log_r[sample_id][bounds[c]:bounds[c+1]]
        xs = np.arange(len(ys))
        # track1.line(xs, ys)
        for i in range(len(ys)):
            x1, x2 = i, i + 1
            color, alpha = cn_state_to_color(states[i])
            track3.rect(x1, x2, lw=0, color=color, alpha=alpha)

    fig = circos.plotfig()


i = 189
circle_plot('none', i)
plt.savefig('icna-circle-1.png', dpi=300)
plt.clf()
circle_plot('rf-da', i)
plt.savefig('icna-circle-2.png', dpi=300)
plt.clf()

print(f'Finished')
