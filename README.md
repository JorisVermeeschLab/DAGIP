# DAGIP: A bias correction algorithm for cell-free DNA data

![Zenodo doi badge](https://zenodo.org/badge/DOI/10.5281/zenodo.14503340.svg)

Given two groups of matched (preferentially paired) samples sequenced under different protocols, the tool explicitly learns the bias using a neural network. The approach builds on Optimal Transport theory, and exploits sample-to-sample similarities to define how to perform bias correction.

Documentations are available at [jorisvermeeschlab.github.io/DAGIP/](https://jorisvermeeschlab.github.io/DAGIP/).
The source code has also been archived on Zenodo: [https://zenodo.org/records/14503340](https://zenodo.org/records/14503340).

---

## Install the tool

Install dependencies:
```bash
pip install -r requirements.txt
```

To reproduce the results of our manuscript, ``rpy2`` should be installed, as well the following R packages: ``dplyr``, ``GenomicRanges`` and ``dryclean``.

Install the package:
```bash
python setup.py install --user
```

---

## Basic usage:

```python
from dagip.core import DomainAdapter

# Let X and Y be two numpy arrays, where the rows are (coverage, methylation, fragmentomic) profiles and columns are features (e.g., DMRs, bins). Y and X have been produced under sequencing protocols 1 and 2, respectively.

# Build a model from the matched groups
model = DomainAdapter()
X_adapted = model.fit_transform(X, Y)

# X_adapted is a numpy array of same dimensions as X

# Save the model
model.save('some/location.pt')

...

# Load the model
model.load('some/location.pt')

# Perform bias correction on new independent samples from sequencing protocol 2
X_new_adapted = model.transform(X_new)
```

For more advanced usage, please check the documentation.

---

## Reproduce the results from the manuscript

To reproduce the results presented in our paper, first download the datasets from the FigShare repository (DOI: 10.6084/m9.figshare.24459304) and place its content in the `data` folder. The repository is available at [https://figshare.com/s/5f837a93ea2719ffcaf9](https://figshare.com/s/5f837a93ea2719ffcaf9).

Preprocess the data:
```bash
python data/preprocess.py
python data/50kb-to1mb.py
python data/to-numpy.py
```

#### Cancer detection

```bash
python scripts/validate-cnas.py HL
python scripts/validate-cnas.py DLBCL
python scripts/validate-cnas.py MM
python scripts/validate-cnas.py OV
python scripts/validate-fragmentomics-multimodal.py
```

#### Cross-validation with paired samples

```bash
python scripts/identify-pairs.py OV-forward
python scripts/identify-pairs.py NIPT-chemistry
python scripts/identify-pairs.py NIPT-lib
python scripts/identify-pairs.py NIPT-adapter
python scripts/identify-pairs.py NIPT-hs2000
python scripts/identify-pairs.py NIPT-hs2500
python scripts/identify-pairs.py NIPT-hs4000
```
