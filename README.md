# Acne-Detection-and-Severity-Grading-using-Deep-Learning


**Author:** Cansu Eyuboglu  
**Institution:** Northumbria University, 2025-2026

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start Guide](#quick-start-guide)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Code Files](#code-files)
- [File Paths & Configuration](#important--file-paths--configuration)
- [Key Results](#key-results-summary)
- [Pipeline Architecture](#pipeline-architecture)
- [Reproducibility Notes](#reproducibility-notes)
- [References](#references)

---

## Project Overview

This project implements a **two-stage deep learning pipeline** for automated acne lesion detection and severity grading using the ACNE04 dataset.

### Stage 1: Object Detection

Two state-of-the-art detection models are trained and evaluated:

- **YOLOv8m** (Ultralytics) — Anchor-free, single-stage detector
- **Faster R-CNN ResNet-50 FPN v2** (TorchVision) — Two-stage detector

### Stage 2: Severity Classification

Detection-derived features (16 features per image) are extracted and used to train a **LightGBM ensemble classifier** with 5-fold stratified cross-validation. Severity is graded under a three-class merged scheme:

| Class | Label    | Criteria     |
| ----- | -------- | ------------ |
| 0     | Mild     | 0-5 lesions  |
| 1     | Moderate | 6-50 lesions |
| 2     | Severe   | >50 lesions  |

**Interpretability:** SHAP analysis is applied to the LightGBM ensemble for post-hoc model explainability.

## Key Results

### Detection Performance (ACNE04 Test Set)

| Model                         | mAP@0.50 | Precision | Recall | MAE  | Pearson r |
| ----------------------------- | -------- | --------- | ------ | ---- | --------- |
| YOLOv8m                       | 0.3165   | 0.4059    | 0.4029 | 3.99 | 0.927     |
| Faster R-CNN ResNet-50 FPN v2 | 0.1939   | 0.1620    | 0.4295 | 4.58 | 0.859     |

### Severity Grading Performance (3-Class Merged Scheme)

| Pipeline                 | Test Accuracy | F1-Score | CV Mean Accuracy |
| ------------------------ | ------------- | -------- | ---------------- |
| YOLOv8m + LightGBM       | 82.91%        | 83.02%   | 86.13% ± 0.87%   |
| Faster R-CNN + LightGBM  | 81.41%        | 81.90%   | 88.59% ± 1.34%   |
| Hayashi Baseline (YOLO)  | 79.90%        | —        | —                |
| Hayashi Baseline (FRCNN) | 68.84%        | —        | —                |

> Both pipelines outperform the Hayashi rule-based baseline.
> SHAP analysis identified **acne_count** and **weighted_severity**
> as the most influential features across both pipelines.

## Quick Start Guide

### Prerequisites

- Google Colab account with GPU access (recommended) or local machine with NVIDIA GPU
- Google Drive account
- Python 3.12+

This project was developed in Google Colab connected to Google Drive.
There is no Git repository — all files are provided via OneDrive submission.

### Step-by-Step Setup

#### Step 1 — Download project files

Download all files from the OneDrive submission link provided with this
project and upload them to your Google Drive.


### Step 2 — Organise folder structure

Place all files according to the folder structure described in the DATASET
section above. The base directory must be:
/content/drive/MyDrive/yolo_acne_detection/

### Step 3- Install Dependencies

pip install torch==2.10.0+cu128 torchvision ultralytics==8.4.38 lightgbm shap scikit-learn pandas matplotlib seaborn

### Step 4 — Run scripts

Run scripts in the following order: 

1. implementation.ipynb 
2. yolo_object_detection_results.ipynb  or fasterrcnn_object_detection_results.ipynb 
3. acne_grading.ipynb 

This will produce:

Annotated detection images
SHAP feature importance plots
Console output with predicted grades and confidence scores

## Dataset

### ACNE04 Dataset Information

| Property             | Value                           |
| -------------------- | ------------------------------- |
| Source               | Wu et al. (2019)                |
| Filtering            | ACNE04v2 (Gazeau et al., 2024)  |
| Total Images         | 1,457 facial images             |
| Total Annotations    | 18,983 bounding box annotations |
| Train/Val/Test Split | 70/15/15 stratified             |
| Train Samples        | 1,345 images                    |
| Val Samples          | 198 images                      |
| Test Samples         | 199 images                      |

> Apply ACNE04v2 filtering before splitting into train/val/test sets.

## Environment Setup

### System Requirements

| Component | Specification                        |
| --------- | ------------------------------------ |
| OS        | Ubuntu 22.04 LTS (or Google Colab)   |
| GPU       | NVIDIA L4 (22.5 GB VRAM) recommended |
| CUDA      | 12.8                                 |
| Python    | 3.12                                 |
| RAM       | 16 GB minimum                        |

### Key Libraries and Versions

torch==2.10.0+cu128
torchvision>=0.15.0
ultralytics==8.4.38
lightgbm>=4.0.0
shap>=0.42.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
Pillow>=9.5.0

## Important Note on Code Files

All code was developed and executed in Google Colab notebooks connected
to Google Drive. The scripts provided in the `source_codes/` folder of
this submission are exported versions of these notebooks.

**Before running any script, the following must be updated:**

1. **Base directory path** — change `BASE_DIR` in each script:

```python
   # Change this to your own Google Drive path
   BASE_DIR = Path('/content/drive/MyDrive/yolo_acne_detection')
```

2. **Dataset paths** — ensure ACNE04 dataset is placed in the correct
   folder structure as described in the DATASET section above.

3. **Model checkpoint paths** — ensure the following files exist:
   - `checkpoints_frcnn/best_model_frcnn.pth` (Faster R-CNN)
   - `runs/yolov8m_acne/weights/best.pt` (YOLOv8m)

The pretrained model checkpoints are available in the `checkpoints/`
folder of the OneDrive submission.

## References

### Core Papers

- Wu et al. (2019): ACNE04 Dataset
- Gazeau et al. (2024): ACNE04v2 Filtering (MICCAI)
- Ren et al. (2015): Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- He et al. (2015): Deep Residual Learning for Image Recognition (ResNet)
- Redmon & Farhadi (2017): YOLO9000: Better, Faster, Stronger
- Smith (2017): A Disciplined Approach to Neural Network Training (Learning Rate Range Test)
- Ke et al. (2017): LightGBM: A Fast, Distributed, High Performance Gradient Boosting Framework
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions (SHAP)

### Datasets & Benchmarks

- **ACNE04:** https://github.com/xpwu95/LDL
- **ACNE04v2:** Gazeau et al. (2024), MICCAI 2024

### Software Frameworks

- **Ultralytics YOLOv8:** https://github.com/ultralytics/ultralytics
- **PyTorch / torchvision:** https://pytorch.org/vision/
- **LightGBM:** https://github.com/Microsoft/LightGBM
- **SHAP:** https://github.com/slundberg/shap

### Related Work
Full reference list with DOIs is provided in the project report (PDF).


---

## Citation

If you use this project in your research, please cite:

```bibtex
@thesis{eyuboglu2026acne,
  author = {Eyuboglu, Cansu},
  title  = {Automated Acne Detection and Severity Grading using Deep Learning},
  school = {Northumbria University},
  year   = {2026},

}
```

---

## License

This project is submitted as a final year project at Northumbria University
for academic assessment purposes.

---

## Contact

| Field           | Details                       |
| --------------- | ----------------------------- |
| **Author**      | Cansu Eyuboglu                |
| **Institution** | Northumbria University        |
| **Email**       | [cansuglneyb@gmail.com] |

For questions regarding this project, please contact the author directly.

---

_Last Updated: May 2026_
