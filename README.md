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

## 📊 Dataset

This project uses multiple datasets:

* PUBLIC dataset
* CMAD dataset (motion artifacts)
* UMAC dataset (underwater EDA)

PUBLIC dataset:
https://zenodo.org/communities/pspm

UMAC dataset:
https://github.com/jzizza/UMAC-Dataset

---

## 🔬 EDA Decomposition Methods

* ospEDA: https://github.com/yongbin98/ospEDA
* cvxEDA: https://github.com/lciti/cvxEDA

---

## Citation

If you find this repository useful for your research, please cite our paper:

**Memory-Efficient EDA Denoising via Knowledge Distillation for Wearable IoT Under Severe Motion Artifacts and Underwater Conditions**  
Yongbin Lee, Andrew Peitzsch, Youngsun Kong, Jarod Zizza, Dong-hee Kang, Farnoush Baghestani, and Ki H. Chon  
arXiv:2605.05246, 2026.  
DOI: https://doi.org/10.48550/arXiv.2605.05246

```bibtex
@article{lee2026memory,
  title={Memory-Efficient EDA Denoising via Knowledge Distillation for Wearable IoT Under Severe Motion Artifacts and Underwater Conditions},
  author={Lee, Yongbin and Peitzsch, Andrew and Kong, Youngsun and Zizza, Jarod and Kang, Dong-hee and Baghestani, Farnoush and Chon, Ki H.},
  journal={arXiv preprint arXiv:2605.05246},
  year={2026},
  doi={10.48550/arXiv.2605.05246}
}
