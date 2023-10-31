# DAGIP: A domain adaptation algorithm for high-dimensional bioinformatics data

---

Install dependencies:
```bash
pip3 install -r requirements.txt
```

To reproduce the results presented in our paper, first download the datasets from the FigShare repository (DOI: 10.6084/m9.figshare.24459304) and place its content in the `data` folder. The repository is available at [https://figshare.com/s/5f837a93ea2719ffcaf9](https://figshare.com/s/5f837a93ea2719ffcaf9).

Convert the data sets to NumPy arrays for more efficient usage:
```bash
python3 build-datasets.py
```

Experiments on sample pairs identification:
```bash
python3 pairs.py OV-forward gc-correction
...
python3 pairs.py NIPT-adapter gc-correction
python3 pairs.py NIPT-hs4000 gc-correction
```

To experiment with GC-correction instead:
```bash
python3 pairs.py NIPT-hs4000 gc-correction
```
Available methods are `none`, `centering-scaling`, `gc-correction` and `ot`.

Reproduce figure 2:
```bash
python3 fig2.py
```

To perform the experiments on supervised learning:
```bash
python3 validation.py HL
python3 validation.py DLBCL
python3 validation.py MM
python3 validation.py OV-forward
python3 validation.py OV-backward
```

To perform the experiments on CNAs, R should be installed first, and the GitHub repository of ichorCNA should be copied to the root of this repository under the name `ichorCNA-master`. To run the experiments:
```bash
python3 icna.py
```
