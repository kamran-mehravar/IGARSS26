
# Hyperspectral Land Surface Classification with Spectral and Spatial CNNs

This repository contains the implementation and experimental workflow for the research paper:

**“The Impact of Spatial Context on Hyperspectral Land Surface Classification with CNNs.”** 

The project investigates the role of **spatial information** in hyperspectral land surface classification by comparing a **1D spectral CNN** with a **3D spatial–spectral CNN (ResNet-based)** using EMIT hyperspectral imagery from the **Cuprite Hills region (Nevada, USA)**.

---

# Authors

* **Kamran Mehravar** — University of Pisa
* **Morteza Safari** — Stony Brook University
* **Nima Esmaeilzadeh** — Western Kentucky University

---

# Research Overview

Hyperspectral imaging is widely used in **geological and mineral mapping**. However, in many geological environments:

* Individual pixels contain **mixtures of minerals**
* Spectral signatures of neighboring pixels are **very similar**
* This creates strong **spatial–spectral continuity**

Traditional hyperspectral classification often treats pixels independently and ignores spatial context.

This work evaluates whether incorporating spatial context significantly improves classification performance.

We compare two architectures:

### 1D Spectral CNN

* Processes **only spectral signatures**
* Input shape: `(B, 1)`
* Focuses purely on spectral absorption characteristics

### 3D CNN (ResNet-18)

* Uses **spectral + spatial context**
* Operates on **3D patches**
* Input shape: `(B, P, P)`
* Captures neighborhood structure in hyperspectral data

The goal is to quantify how much spatial information contributes to classification accuracy in **spectrally mixed geological environments**.

---

# Dataset

The experiments use **EMIT Level-2A reflectance hyperspectral data**.

Location:
**Cuprite Hills, Nevada (USA)**

Characteristics:

* Spectral range: **381 – 2450 nm**
* Mineralogical hyperspectral imaging
* Well-known benchmark site for geological remote sensing

Processing steps:

1. Continuum removal to reduce illumination effects
2. Pseudo-endmember detection for label generation
3. Pixel-level alignment between hyperspectral cube and labels
4. Stratified train/validation/test split

---

# Methodology Pipeline

The workflow follows a deterministic pipeline to guarantee reproducibility.

Pipeline stages:

1. Data preparation and preprocessing
2. Spectral normalization and continuum removal
3. Label generation (pseudo-endmembers)
4. Dataset split (train / validation / test)
5. Model training
6. Sliding-window inference for full-scene prediction
7. Evaluation and visualization

Key evaluation metrics:

* **Pixel Accuracy**
* **Mean Intersection-over-Union (mIoU)**
* **Confusion Matrix**
* **Spatial Difference Maps**

---

# Key Findings

Both models achieve **high classification accuracy (>80%)**.

However:

* Spatial information provides **only modest improvement**
* The main discriminative information lies in **spectral signals**
* Spatial context becomes less important in **highly mixed geological environments**

This suggests that **spectral CNNs may be sufficient** for many mineral classification tasks while being computationally cheaper.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/kamran-mehravar/IGARSS26.git
cd IGARSS26
```

Install dependencies:

* Python 3.9+
* PyTorch
* NumPy
* Scikit-learn
* Rasterio
* Matplotlib

---

# Training

Train the **1D spectral CNN**

```bash
python train_1dcnn.py
```

Train the **3D CNN**

```bash
python train_3dcnn.py
```

---

# Inference

Generate full-scene classification maps:

```bash
python sliding_window_inference.py
```

This produces:

* Pixel-level predictions
* Full classification maps
* Spatial difference visualization

---

# Reproducibility

To ensure reproducibility:

* All **random seeds are fixed**
* Dataset splits are **stratified**
* Training checkpoints are saved
* All preprocessing steps are deterministic

---

# Citation

If you use this code or dataset pipeline, please cite:

```
Safari, M., Mehravar, K., Esmaeilzadeh, N.
The Impact of Spatial Context on Hyperspectral Land Surface Classification with CNNs.
IGARSS 2026.
```

---

# License

This project is released under the **MIT License**.

---

