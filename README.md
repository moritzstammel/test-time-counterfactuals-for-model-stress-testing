# Test time Counterfactuals for Model Stress Testing

This repository contains the code accompanying the MSc thesis **“Test time counterfactuals for model stress testing.”** It includes dataset loaders, classifier training, counterfactual stress tests, and analysis notebooks.

## Framework overview

The following document shows the high level framework at a glance.

[Open the overview](overview.pdf)

## Quick start

1. Create and activate a Python environment
2. Install dependencies
3. Prepare dataset roots
4. Train classifiers
5. Run counterfactual stress tests
6. Explore results with the notebooks

```bash
# 1. optional but recommended
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate

# 2. install
pip install --upgrade pip
pip install -r requirements.txt
```

## Project structure

```
.
├── requirements.txt
├── overview.pdf
└── src
    ├── classifier-training
    │   ├── train_on_embed.py
    │   └── train_on_padchest.py
    ├── datasets
    │   ├── __init__.py
    │   ├── embed.py
    │   └── padchest.py
    ├── helpers
    │   ├── __init__.py
    │   └── helpers.py
    ├── notebooks
    │   ├── correlation_plots.ipynb
    │   ├── counterfactuals_artifacts.ipynb
    │   ├── generate_embed_csv.ipynb
    │   ├── pca_evaluation.ipynb
    │   └── stress_test_evaluation.ipynb
    └── stress-testing
        ├── stress_test_on_embed.py
        └── stress_test_on_padchest.py
```

## Datasets

This code expects two datasets.

**PadChest**
* Task: pneumonia classification
* Required fields: image path, label, scanner vendor, sex

**EMBED**
* Task: breast density classification A to D
* Required fields: image path, density label, scanner vendor, view

Place the raw data in folders of your choice. Many users keep a layout like:

```
data/
  padchest/
    images/
    metadata.csv
  embed/
    images/
    metadata.csv
```

If needed, use `src/notebooks/generate_embed_csv.ipynb` to build or harmonize the csv for EMBED.

## Training

PadChest example

```bash
python src/classifier-training/train_on_padchest.py \
  --data_root data/padchest \
  --out_dir runs/padchest_baseline
```

EMBED example

```bash
python src/classifier-training/train_on_embed.py \
  --data_root data/embed \
  --out_dir runs/embed_baseline
```

Notes

* Default values inside the scripts are sensible for a first run
* Use the help flag to see all options
  ```bash
  python src/classifier-training/train_on_padchest.py --help
  ```

## Counterfactual stress testing

PadChest example

```bash
python src/stress-testing/stress_test_on_padchest.py \
  --data_root data/padchest \
  --ckpt runs/padchest_baseline/model.pt \
  --out_dir runs/padchest_stresstest
```

EMBED example

```bash
python src/stress-testing/stress_test_on_embed.py \
  --data_root data/embed \
  --ckpt runs/embed_baseline/model.pt \
  --out_dir runs/embed_stresstest
```

Outputs typically include metrics like AUC and TPR at a fixed FPR, plus any plots or tables emitted by the scripts.

## Notebooks

Analysis and visualization

* `stress_test_evaluation.ipynb` compares baselines and counterfactual tests
* `pca_evaluation.ipynb` projects features to inspect distribution alignment
* `correlation_plots.ipynb` explores relationships among metrics
* `counterfactuals_artifacts.ipynb` looks for artifacts in generated images
* `generate_embed_csv.ipynb` builds a clean csv for EMBED if you need it

You can launch Jupyter as follows

```bash
jupyter lab
```

## Helpers and datasets

* `src/datasets/padchest.py` and `src/datasets/embed.py` implement dataset loaders, preprocessing, and any metadata parsing
* `src/helpers/helpers.py` contains common utilities such as logging, seeding, metric computation, and file io


## Acknowledgments

This codebase builds on ideas and components from the following excellent projects

* Causal Gen by BioMedIA  
  https://github.com/biomedia-mira/causal-gen

* Counterfactual Contrastive by BioMedIA  
  https://github.com/biomedia-mira/counterfactual-contrastive/tree/main

Please cite and follow their licenses when using derived parts.

## Citation

If you use this repository in academic work, please cite the thesis

```
@thesis{stammel2025_stress_testing,
  title   = {Test time counterfactuals for model stress testing},
  author  = {Stammel, Moritz Paul},
  school  = {Imperial College London},
  year    = {2025}
}
```

