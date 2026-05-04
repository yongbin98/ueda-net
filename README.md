# uEDA-Net: Underwater EDA Denoising Model

## Overview

uEDA-Net is a lightweight deep learning framework for robust electrodermal activity (EDA) denoising under harsh environments, including underwater and motion artifact conditions.

The model is designed to:

* Remove motion artifacts from EDA signals
* Preserve tonic and phasic components
* Generalize across multiple datasets (PUBLIC, CMAD, UMAD)

---

## 📁 Repository Structure

```
data/               # Processed datasets
model/              # Model architecture
weights/            # Pretrained weights
results/            # Output results
Train.ipynb         # Training notebook
Evaluation.ipynb    # Evaluation notebook
data_preprocess.py  # Data preprocessing (including augmentation) pipeline
requirements.txt    # Dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yongbin98/uEDA-Net.git
cd uEDA-Net
pip install -r requirements.txt
```

## 📊 Dataset

This project uses multiple datasets:

* PUBLIC dataset
* CMAD dataset (motion artifacts)
* UMAC dataset (underwater EDA)

PUBLIC dataset:
https://zenodo.org/communities/pspm

UMAC dataset:
https://github.com/jjizza/UMAC-Dataset

---

## 🔬 EDA Decomposition Methods

* ospEDA: https://github.com/yongbin98/ospEDA
* cvxEDA: https://github.com/lciti/cvxEDA

---
