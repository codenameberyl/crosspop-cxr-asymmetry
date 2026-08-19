# Directional Asymmetry of Cross-Population Generalisation in Lightweight Pediatric Pneumonia Classifiers

A confidence-reliability and performance evaluation framework analysing cross-population transfer between high-resource (Kermany) and low-resource (Togunwa, Nigeria) under-five chest radiograph cohorts using MobileNetV2 and EfficientNet-B0.

MSc Artificial Intelligence Dissertation
School of Computing and Digital Technologies, Sheffield Hallam University

---

## Research overview

This repository investigates whether cross-population generalisation between high-resource and low-resource under-five pediatric chest radiograph cohorts exhibits directional asymmetry in both discrimination and calibration, after strictly controlling for source training-set size.

Primary research question: to what extent is cross-population generalisation between the Kermany (China) and Togunwa (Nigeria) under-five chest-radiograph cohorts directionally asymmetric for MobileNetV2 and EfficientNet-B0, in terms of discrimination (AUROC, specificity at 95% sensitivity) and calibration (ECE, MCE, Brier score), after controlling for source training-set size?

---

## Experimental design

The study executes a fully factorised experimental grid:

- 2 architectures: MobileNetV2, EfficientNet-B0
- 2 transfer directions:
  - K2N: train on Kermany, evaluate on the Togunwa Nigerian holdout.
  - N2K: train on Togunwa Nigerian, evaluate on the Kermany test set.
- 3 matched training sizes: 50, 100, and 190 samples (1:1 class-balanced downsampling, nested per seed).
- 3 random seeds: 42, 43, 44 (yielding 48 model fits in total).

Methodological safeguard: Kermany training subsets are downsampled to match Togunwa size constraints, balanced 1:1, and nested (50 within 100 within 190) at a fixed seed to prevent confounding directional asymmetry with class imbalance or sample volume. Temperature scaling is fitted exclusively on in-fold validation splits, never on the evaluation target.

---

## Repository layout

```text
crosspop-cxr-asymmetry/
├── config.py                 # Single source of truth: paths, design constants, seeds, plot style.
├── requirements.txt          # Environment dependencies.
├── .gitignore                # Excludes datasets, checkpoints, and raw images from git.
├── README.md                 # Project documentation.
│
├── notebooks/                # Sequenced analysis notebooks (run in order)
│   ├── 01_verify_eda.ipynb
│   ├── 02_splits_downsampling.ipynb
│   ├── 03_preprocess_train.ipynb
│   ├── 04_inference.ipynb
│   ├── 05_discrimination.ipynb
│   ├── 06_calibration.ipynb
│   ├── 07_size_interaction.ipynb
│   ├── 08_significance.ipynb
│   ├── 09_ood_musa.ipynb
│   └── 10_aggregate.ipynb    # Assembles dissertation tables & figures
│
├── src/                      # Modular Python package imported by notebooks
│   ├── __init__.py
│   ├── splits.py             # Data collection, stratified holdout, k-fold, nested downsampling.
│   ├── datasets.py           # CXRDataset class and torchvision transforms.
│   ├── models.py             # MobileNetV2 and EfficientNet-B0 factory functions.
│   ├── train.py              # Early-stopping training loops.
│   ├── inference.py          # Batch prediction, logit export, temperature scaling fitting.
│   ├── metrics.py            # AUROC, specificity at 95% sensitivity, ECE, MCE, Brier, OOD scoring.
│   ├── stats.py              # Bootstrap CIs, DeLong test, McNemar test.
│   └── results.py            # Run-ID parsing and result pooling utilities.
│
└── outputs/                  # Git-ignored pipeline outputs (generated via code)
    ├── results/              # CSV artefacts, manifests, and outputs/results/final/
    ├── figures/              # Plot PNGs and outputs/figures/final/
    └── checkpoints/          # Model state dicts and training metadata JSONs.
```

The data/ folder lives outside this repository and is never committed. Only code, configuration and aggregate numerical results are tracked. Splits are stored as manifests (CSV lists of image paths and labels), never as copies of image data.

---

## Execution pipeline

Notebooks are designed to be run sequentially. Data files are kept external to the repository and referenced via config.py.

| Notebook | Phase | Focus / deliverable | Target hardware |
| --- | --- | --- | --- |
| 01_verify_eda | Verification | Integrity checks, image metadata verification, EDA, pipeline go/no-go. | CPU |
| 02_splits_downsampling | Data prep | Stratified Togunwa holdouts, 5-fold CV, nested 1:1 Kermany downsamples. | CPU |
| 03_preprocess_train | Training | Executes the 48 training runs across seeds, sizes, and architectures. | GPU required |
| 04_inference | Evaluation | Runs target-set inference, outputs raw logits and probabilities. | CPU / GPU |
| 05_discrimination | SQ1 | Pooled AUROC and specificity at 95% sensitivity with 95% bootstrap CIs. | CPU |
| 06_calibration | SQ2 | ECE, MCE, Brier, and reliability diagrams (raw vs temperature-scaled). | GPU required |
| 07_size_interaction | SQ3 | Kermany source-size scaling curves (50, 100, 190) vs N2K baseline. | CPU |
| 08_significance | Hypothesis | Direction contrast tests, DeLong and McNemar tests, Holm-Bonferroni correction. | CPU |
| 09_ood_musa | Exploratory | Out-of-distribution confidence on the Musa TB/COVID cohorts. | GPU required |
| 10_aggregate | Synthesis | Consolidates Tables 1–5, Figures 1–5, and a plain-text results digest. | CPU |

---

## Generated artefacts

Executing 10_aggregate.ipynb populates the final evaluation artefacts ready for dissertation inclusion:

```text
outputs/
├── results/final/
│   ├── table_1_discrimination.csv      # SQ1 pooled AUROC & specificity
│   ├── table_2_calibration.csv         # SQ2 ECE & Brier scores
│   ├── table_3_size.csv                # SQ3 size-interaction data
│   ├── table_4_significance.csv        # Consolidated statistical tests
│   ├── table_5_ood.csv                 # Exploratory OOD separation
│   └── results_summary.md              # Plain-text results digest
└── figures/final/
    ├── figure_1_discrimination_calibration.png  # SQ1 & SQ2 bar charts
    ├── figure_2_spread.png                      # Per-run spread across seeds/folds
    ├── figure_3_size_interaction.png            # SQ3 scaling interaction plot
    ├── figure_4_reliability.png                 # SQ2 reliability diagrams
    └── figure_5_ood_confidence.png              # Exploratory OOD confidence
```

---

## Quick start

1. Environment setup:

```bash
git clone https://github.com/your-username/crosspop-cxr-asymmetry.git
cd crosspop-cxr-asymmetry
pip install -r requirements.txt
```

2. Configure data root: edit DATA_ROOT in config.py to point to your dataset directory containing kermany/, togunwa/, and optionally musa/.

3. Pipeline execution:
- Run notebooks/01_verify_eda.ipynb and ensure all go/no-go checks pass.
- Run notebooks 02 through 10 sequentially.

---

## Datasets and citations

1. Kermany pediatric CXR dataset (China, high-resource):
Kermany, D. S., et al. (2018). Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification. Mendeley Data, v3. https://doi.org/10.17632/rscbjbr9sj.3 (CC BY 4.0). NORMAL and PNEUMONIA.

2. Togunwa pediatric CXR cohort (Nigeria, low-resource):
Togunwa, T. O., et al. (2025). Nigerian Paediatric Chest X-Ray Dataset. Zenodo. https://doi.org/10.5281/zenodo.14185822 (CC BY 4.0). 190 under-five radiographs, NORMAL and PNEUMONIA.

3. Musa CXR cohort (Nigeria, exploratory OOD):
Musa, A. Nigeria Chest X-Ray Dataset. Kaggle. https://www.kaggle.com/datasets/aminumusa/nigeria-chest-x-ray-dataset (Apache 2.0). Four classes (NORMAL, PNEUMONIA, TB, COVID). Used solely for exploratory OOD evaluation on unseen pathologies; the test partition only.

---

## Ethics and scope

- Scope: this codebase is an academic investigation into cross-population transfer, deep learning calibration, and dataset asymmetry. It is not intended for clinical diagnosis or medical deployment, and no clinical claim is made.
- Data integrity: all image data is held in external directories and accessed strictly via path manifests (CSVs). No raw patient images are stored or tracked within this repository.
- The out-of-distribution analysis is exploratory: the Musa cohort mixes paediatric and adult radiographs, so its results are reported separately from the core findings. See the ethics documentation for data handling and approval.
