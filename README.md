
# Hyperspectral Land Surface Classification with Spectral and Spatial CNNs

This repository contains the implementation and experimental workflow for the research paper:

**“The Impact of Spatial Context on Hyperspectral Land Surface Classification with CNNs.”**

The project investigates the role of **spatial information** in hyperspectral land surface classification by comparing a **1D spectral CNN** with a **3D spatial–spectral CNN (ResNet-based)** using EMIT hyperspectral imagery from the **Cuprite Hills region (Nevada, USA)**.

---

# Authors

* **Morteza Safari** — Stony Brook University
* **Kamran Mehravar** — University of Pisa  
* **Nima Esmaeilzadeh** — Western Kentucky University  

---

# Research Overview

Hyperspectral imaging is widely used in **geological and mineral mapping**. However, in many geological environments:

* Individual pixels contain **mixtures of minerals**
* Spectral signatures of neighboring pixels are **very similar**
* This creates strong **spatial–spectral continuity**

Traditional hyperspectral classification often treats pixels independently and ignores spatial context.

This work evaluates whether incorporating spatial context significantly improves classification performance.

Two architectures are compared:

### 1D Spectral CNN

* Processes **only spectral signatures**
* Input shape: `(B)`
* Focuses purely on spectral absorption characteristics

### 3D Spatial–Spectral CNN (ResNet-18)

* Uses **spectral + spatial context**
* Operates on **hyperspectral patches**
* Input shape: `(B, P, P)`
* Captures neighborhood structure in hyperspectral data

The goal is to quantify how much spatial information contributes to classification accuracy in **spectrally mixed geological environments**.

---

# Dataset

The experiments use **EMIT Level-2A hyperspectral reflectance data**.

Location  
**Cuprite Hills, Nevada, USA**

Characteristics:

* Spectral range: **381 – 2450 nm**
* Hyperspectral mineral mapping benchmark
* High spectral dimensionality

Processing steps:

1. Continuum removal
2. Pixel-level alignment between hyperspectral cube and labels
3. Pseudo-endmember label generation
4. Patch extraction
5. Stratified train / validation / test split

---

# Methodology Pipeline

The workflow follows a deterministic pipeline to ensure reproducibility.

Pipeline stages:

1. Hyperspectral cube preprocessing  
2. Label alignment and patch extraction  
3. Dataset generation  
4. Model training  
5. Validation and early stopping  
6. Full-scene inference  
7. Evaluation and visualization  

Evaluation metrics include:

* **Pixel Accuracy**
* **Mean Intersection-over-Union (mIoU)**
* **Confusion Matrix**
* **Prediction Agreement Maps**

---

# Repository Structure

```

project/
│
├── README.md
│
├── data_preparation/
│   └── build_dataset_from_georef_labels.py
│
├── training_inference/
│   └── Model_Supervised_1d_3d.py
│
├── visualization/
│   └── RGB_copritmap_and_diff.py
│
├── utils/
│   ├── analyze_dataset_out.py
│   ├── analyze_dataset_out_v2.py
│   ├── check_align.py
│   └── inspect_labels.py

````

---

# Installation

Clone the repository:

```bash
git clone https://github.com/kamran-mehravar/IGARSS26.git
cd IGARSS26
````

Install dependencies:

```bash
pip install torch torchvision numpy rasterio matplotlib scikit-learn tqdm
```

Requirements:

* Python ≥ 3.9
* PyTorch ≥ 2.0
* NumPy
* Rasterio
* Matplotlib
* Scikit-learn
* tqdm

---

# Dataset Preparation

Generate the training dataset from georeferenced hyperspectral data:

```bash
python data_preparation/build_dataset_from_georef_labels.py
```

This step creates:

```
dataset/
│
├── meta.json
├── patches/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
```

---

# Model Training

Train both the **1D spectral CNN** and **3D spatial–spectral CNN**:

```bash
python training_inference/Model_Supervised_1d_3d.py
```

The script performs:

* Dataset loading
* Model training
* Validation
* Early stopping
* Test evaluation
* Confusion matrix generation

Outputs include:

```
runs_A100_Final/
│
├── best_1D.pt
├── best_3D.pt
├── curve_1d.png
├── curve_3d.png
├── cm_1d_test.png
├── cm_3d_test.png
```

---

# Full Scene Inference

The same script can generate **full-scene prediction maps**.

Outputs include:

```
runs_A100_Final/maps/

pred_1d.tif
pred_3d.tif
diff_1d_3d.tif

visual_map_1d.png
visual_map_3d.png
visual_diff_map.png

matrix_1d_vs_3d.png
```

These maps allow comparison between spectral-only and spatial-spectral predictions.

---

# Key Findings

Both models achieve **high classification accuracy (>80%)**.

However:

* Spatial information provides **only modest improvement**
* Most discriminative information is contained in **spectral signatures**
* Spatial context becomes less critical in **spectrally mixed geological environments**

This suggests that **spectral CNNs may already capture most of the relevant information for mineral classification**, while spatial models introduce additional computational complexity.

---

# Reproducibility

To ensure reproducibility:

* Random seeds are fixed
* Dataset splits are deterministic
* Training checkpoints are saved
* All preprocessing steps are deterministic

---

# Citation

If you use this code, please cite:

```
Safari, M., Mehravar, K., Esmaeilzadeh, N.
The Impact of Spatial Context on Hyperspectral Land Surface Classification with CNNs.
IGARSS 2026.
```

---

# License

This project is released under the **MIT License**.

```

---
