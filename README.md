# RagwortDetection: Automated Computer Vision Pipeline for Real-World Ragwort Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://docs.ultralytics.com/)
[![DINOv2 / DINOv3](https://img.shields.io/badge/Vision%20Transformer-DINOv3-FF6F00.svg)](https://github.com/facebookresearch/dinov2)
[![DEIMv2](https://img.shields.io/badge/DETR-DEIMv2-792EE5.svg)](https://github.com/ShihuaGao/DEIMv2)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)](https://www.docker.com/)
[![TU Berlin Erasmus+](https://img.shields.io/badge/TU%20Berlin-Erasmus%2B-CC0000.svg)](https://www.tu.berlin/)
[![Web Application](https://img.shields.io/badge/Web%20App-TU--BerlinB%2FWebsite-2ea44f.svg?logo=github)](https://github.com/TU-BerlinB/Website)

---

## 1. Introduction & Project Overview

**RagwortDetection** is an applied computer vision project developed at **Technische Universität Berlin (TU Berlin)** as part of the **Erasmus+** exchange program. 

> [!TIP]
> **Companion Web Application**: The interactive web platform and UI developed for this project can be found in the companion repository: [TU-BerlinB/Website](https://github.com/TU-BerlinB/Website).

### The Problem: Toxic Weed Invasion in Agriculture
Common Ragwort (*Jacobaea vulgaris* / *Senecio jacobaea*) is a highly invasive weed that contains toxic pyrrolizidine alkaloids. These alkaloids cause irreversible liver cirrhosis and failure in grazing livestock, particularly horses and cattle. Even when dried into hay or silage, the toxin remains lethal while losing the bitter taste that naturally deters animals from eating it.

Early eradication is critical. However, manual pasture inspection is labor-intensive, error-prone, and difficult across vast fields. Automated detection systems (e.g., mounted on tractors, autonomous mowers, or agricultural drones) require robust computer vision algorithms capable of functioning in harsh outdoor conditions.

### The Challenge & Real-World Validation
Ragwort exhibits extreme morphological variations across growth stages:
- **Rosette / Leaf Stage**: Low-lying, serrated green leaves closely matching surrounding pasture grasses, clover, and harmless weeds.
- **Flowering Stage**: Clusters of vibrant yellow daisy-like inflorescences.
- **Senescence**: Dry, fibrous brown stalks and seed heads.

Our international team engineered, deployed, and tested an **end-to-end computer vision pipeline under real-world agricultural conditions**. Rather than relying purely on clean web datasets, the pipeline was specifically optimized and validated using high-resolution, unconstrained field photography captured directly in pastures and meadows.

---

## 2. Core Technologies & Architecture

- **Deep Learning Frameworks**: [PyTorch](https://pytorch.org/) (2.5+), [Torchvision](https://pytorch.org/vision/), [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- **Object Detection & Foundation Models**:
  - **Ultralytics YOLOv8**: Real-time anchor-free CNN detector optimized for edge inference on agricultural machinery.
  - **Meta DINOv2 / DINOv3**: Self-supervised Vision Transformers (ViT) delivering high-dimensional visual feature extraction, phenotyping, and leaf-vs-flower representation learning.
  - **DEIMv2 / RT-DETR**: Real-Time Detection Transformers pairing dense feature enhancement and Hungarian matching with high-throughput inference.
- **Data Engineering, Annotation & Image Processing**: **CVAT (Computer Vision Annotation Tool)**, OpenCV, Pillow, Albumentations, Segment Anything (SAM), Roboflow API.
- **Analysis & Visualization**: UMAP (Uniform Manifold Approximation and Projection), PCA, scikit-learn, Matplotlib, Faster-COCO-Eval.
- **Infrastructure & Containerization**: Docker, Docker Compose, NVIDIA CUDA 12+, GPU passthrough.

---

## 3. Project Contributors & Academic Affiliation

This project was conceived, implemented, and benchmarked by our international engineering team in collaboration with **Technische Universität Berlin (TU Berlin)**:

- **Szymon Drywa** ([@Szmekowy](https://github.com/Szmekowy))
- **Szymon D. Olkowski** ([@szymonolkowski](https://github.com/szymonolkowski))
- **Mateusz Dadura** ([@Invictiuc](https://github.com/Invictiuc))
- **Felix Sjosted** ([@LAZYCAT33](https://github.com/LAZYCAT33))
- **Joanna Mielczarek** ([@AsiaAl21](https://github.com/AsiaAl21))
- **Anshika Gahlot** ([@agahlot257-gif](https://github.com/agahlot257-gif))

*Academic Institution*: **Technische Universität Berlin (TU-Berlin)**  
*Program*: **Erasmus+**

---

## 4. Repository Structure

```text
RagwortDetection/
├── Dockerfile                      # Unified container image with multi-requirement support
├── docker-compose.yml              # GPU-enabled service definitions (yolo, deim)
├── requirements/
│   ├── requirementsYOLO.txt        # Pinned dependencies for YOLOv8 & PyTorch 2.5
│   └── requirementsDEIM.txt        # Dependencies for DEIMv2, DINOv3 & COCO evaluations
├── data/                           # Data storage (gitignored except placeholders)
│   ├── data_concatenated/          # Multi-source dataset (annotated_data, synthetic, combined)
│   └── synthetic_dataset/          # Generated synthetic training pairs
├── outputs/                        # Output artifacts, model weights, runs, and reports
│   ├── datasets/                   # Weighted YOLO & COCO manifests
│   ├── models/                     # Best exported model checkpoints
│   ├── runs/                       # Training logs, TensorBoard runs
│   └── fine_labeling_results/      # Multi-stage fine-tuning metric reports
└── src/
    ├── dataset/                    # Dataset downloading utilities (Roboflow, sampling)
    ├── embeddings/                 # Feature extraction from DINO foundation models
    ├── fine_tuning/                # Multi-stage fine-tuning pipelines (real-data & leaf geometry)
    │   ├── fine_tuning.py          # Stage 1 (pure real) & Stage 2 (weighted real + synthetic)
    │   └── fine_tuning_leaves.py   # Stage 3 (1024px), Stage 4 (masking), Stage 5 (No-Yellow)
    ├── inspection/                 # Visual inspection and diagnostic tools for DINOv3
    ├── models/                     # Model wrapper definitions (YOLOv8, DINOv3, DEIM, Model base)
    ├── scripts/                    # Preprocessing, data synthesis, weighting, conversions
    │   ├── data_download.py        # Automated Roboflow dataset downloader
    │   ├── download_background.py  # Pasture background downloader
    │   ├── extracting_ragwort_from_photo.py # Plant segmentation cutouts
    │   ├── filter_backgrounds.py   # Corruption and dimension background filter
    │   ├── import_field_images.py  # Field photograph ingest & verification
    │   ├── select_representative.py # Representative sample clustering via embeddings
    │   ├── split_folder.py         # Balanced dataset chunking
    │   ├── weight_dataset.py       # Crucial: Weighted manifest & COCO JSON exporter
    │   └── yolo_to_coco.py         # YOLO-to-COCO bounding box conversion
    ├── train_evaluation/           # End-to-end training and evaluation scripts
    │   ├── yolo_eval.py            # YOLOv8 training, validation, IoU=0.5 analysis, test suite
    │   └── DINO_DEIM_eval.py       # DEIMv2+DINOv3 training, validation, error mining
    └── visualisation/              # 2D UMAP/PCA embedding visualization
```

---

## 5. Data Collection, Synthetic Augmentation & Dataset Weighting

A primary innovation of this project is solving the severe domain shift between web images and real agricultural environments.

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Curated Ground Truth        │       │  Pasture Backgrounds (No Weeds) │
│ (annotated_data - via CVAT)     │       │  (OpenImages / Custom Pastures) │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 │                        ┌────────────────▼────────────────┐
                 │                        │   Cutout Synthesis & Pasting    │
                 │                        │ (extracting_ragwort_from_photo) │
                 │                        └────────────────┬────────────────┘
                 │                                         │
                 │                                         ▼
                 │                        ┌─────────────────────────────────┐
                 │                        │    Synthetic Dataset (Diverse)  │
                 │                        └────────────────┬────────────────┘
                 │                                         │
                 ▼                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     weight_dataset.py Engine                              │
│  - Annotated Data: Weighted 15x (primary gradient influence)              │
│  - Synthetic & Web Data: Weighted 1x (regularization & variety)           │
│  - Unbiased Split: Reserved authentic validation subsets                  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│        YOLO Manifests        │              │       COCO Annotations       │
│  data_weighted.yaml, val.txt │              │       annotations.json       │
└──────────────────────────────┘              └──────────────────────────────┘
```

### 1. Data Ingestion & Downloading
- **Roboflow & Public Repositories**: `src/scripts/data_download.py` automates pulling labeled ragwort datasets in YOLOv8 format using the Roboflow API.
- **Curated Annotated Dataset (`annotated_data`)**: High-quality, verified ground-truth annotations labeled and audited using **CVAT (Computer Vision Annotation Tool)**, capturing ragwort across diverse growth stages and complex grassland environments.

### 2. Plant Segmentation & Synthetic Generation
To expose models to thousands of variations without manual labeling bottlenecks:
1. `src/scripts/download_background.py` and `filter_backgrounds.py`: Pull weed-free grassland backgrounds.
2. `src/scripts/extracting_ragwort_from_photo.py`: Extracts pixel-precise transparent cutouts of ragwort plants.
3. Compositing: Cutouts are scaled, rotated, color-jittered, and blended onto natural pasture backgrounds to generate diverse synthetic training pairs.

### 3. Gradient Balancing via `weight_dataset.py`
Training models naively on pooled synthetic and web images causes models to overfit to synthetic rendering artifacts while failing in real pastures.

`src/scripts/weight_dataset.py` resolves this:
- **Primary Annotated Data (`annotated_data`)** is upweighted **15x**, ensuring authentic ground-truth features dominate parameter updates.
- **Synthetic & Web Data** is kept at **1x** to provide background variety and prevent overfitting.
- Generates zero-copy manifests for YOLO (`outputs/datasets/data_weighted/data_weighted.yaml`) and COCO format (`outputs/datasets/coco/`) for transformer models.

---

## 6. Models Overview

| Model | Type | Backbone / Architecture | Primary Role in Pipeline | Key Advantages |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8** | Object Detector | CSPDarknet + C2f + PAN-FPN | Real-time field detection | High FPS (>60 FPS on edge GPUs), lightweight deployment, robust bounding boxes. |
| **DINOv3 / DINOv2** | Foundation Model | Vision Transformer (ViT-Base / Large) | Visual feature extraction & phenotyping | Superb semantic feature representations without supervision; enables clustering of leaf vs flower stages. |
| **DEIMv2 / RT-DETR** | Detection Transformer | DINOv3 ViT + Hybrid Encoder + Hungarian Matching | Complex scene detection | Eliminates NMS hand-tuning; superior handling of heavily occluded plants and dense leaf clusters. |

---

## 7. Multi-Stage Fine-Tuning Pipeline

Standard object detection models suffer from **"Yellow Flower Bias"**: models rely almost entirely on bright yellow blossoms and fail completely when encountering young rosettes before flowering.

To solve this, we designed a **specialized multi-stage fine-tuning methodology**:

### Phase A: Real-World Grounding (`src/fine_tuning/fine_tuning.py`)
1. **Stage 1 (Pure Annotated Data)**: Trains exclusively on high-quality verified images from `annotated_data` alongside hard negative pasture backgrounds using AdamW.
2. **Stage 2 (Weighted Ground-Truth Data)**: Integrates synthetic data regularizers while maintaining a 12–15x gradient dominance for `annotated_data`.

### Phase B: Leaf Geometry Specialization (`src/fine_tuning/fine_tuning_leaves.py`)
3. **Stage 3 (High Resolution - 1024px)**: Increases input resolution to capture fine serrations along leaf margins.
4. **Stage 4 (Leaf-Focused Augmentations)**: Dynamically masks or randomly erases flower heads, forcing attention heads toward stem and leaf morphology.
5. **Stage 5 (No-Yellow Training)**: Desaturates yellow color spectrums to simulate non-flowering or overcast conditions, ensuring the model identifies ragwort purely by plant geometry.

---

## 8. Docker Setup & Deployment

Docker eliminates CUDA/PyTorch version mismatches. Both containers feature full NVIDIA GPU passthrough and shared host IPC.

### Prerequisites
- [Docker Engine 24+](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose v2+

### 1. Build and Run YOLO Environment
```bash
# Build container image
docker compose build yolo

# Launch interactive GPU-enabled container
docker compose run --rm yolo bash
```

### 2. Build and Run DEIM / DINO Environment
```bash
# Build container image
docker compose build deim

# Launch interactive GPU-enabled container
docker compose run --rm deim bash
```

Inside the containers, the project workspace is mounted at `/workspace`.

---

## 9. Local Installation (Alternative to Docker)

If running directly on a Linux host with an NVIDIA GPU and CUDA 12:

```bash
# 1. Clone repository
git clone https://github.com/szmekowy/RagwortDetection.git
cd RagwortDetection

# 2. Create and activate Python 3.12 virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# 4. Install YOLO dependencies
pip install -r requirements/requirementsYOLO.txt

# (Optional) Install DEIM / Transformer dependencies
pip install -r requirements/requirementsDEIM.txt
```

---

## 10. How to Run Scripts: Step-by-Step Guide

### Step 1: Prepare and Weight Dataset
Generate the weighted dataset manifests and COCO JSON files:
```bash
python src/scripts/weight_dataset.py --good-weight 15 --other-weight 1
```

### Step 2: Train & Evaluate YOLOv8
Train YOLOv8 on the weighted dataset with automatic best weights export and IoU=0.5 error analysis:
```bash
python src/train_evaluation/yolo_eval.py --epochs 50 --batch 16 --imgsz 640
```
To evaluate existing checkpoints without retraining:
```bash
python src/train_evaluation/yolo_eval.py --skip-train --weights outputs/models/ragwort_yolov8_best.pt
```

### Step 3: Train & Evaluate DEIMv2 + DINOv3
Train the Real-Time Detection Transformer with a DINOv3 backbone:
```bash
# Run quick verification test
python src/train_evaluation/DINO_DEIM_eval.py --smoke-test --batch-size 4

# Run full training pipeline
python src/train_evaluation/DINO_DEIM_eval.py --epochs 30 --batch-size 4 --device cuda

# Evaluate existing checkpoint
python src/train_evaluation/DINO_DEIM_eval.py --skip-train --resume outputs/checkpoints/deimv2_dinov3_ragwort/best_stg1.pth
```

### Step 4: Run Multi-Stage Leaf Fine-Tuning
Execute the two-stage real-world fine-tuning:
```bash
python src/fine_tuning/fine_tuning.py --model outputs/models/ragwort_yolov8_best.pt --stage1-epochs 10 --stage2-epochs 10
```
Run the leaf-specialized stages (Stages 3–5) to counter yellow-flower bias:
```bash
python src/fine_tuning/fine_tuning_leaves.py --model outputs/fine_labeling_results/stage2/best.pt
```

### Step 5: Visual Feature Embeddings & Phenotyping (DINO)
Extract deep visual feature vectors:
```bash
python src/embeddings/extract_embedings.py --model dinov2_vits14 --data-dir data/annotated_data
```
Generate interactive 2D UMAP / PCA visualization clusters:
```bash
python src/visualisation/visualize_embeddings.py --features outputs/features.npy
```

---

## 11. Evaluation Methodology & Metrics

Model performance is evaluated across both standardized COCO metrics and custom agricultural operational metrics:

1. **Standard COCO Metrics**:
   - $\text{mAP}_{50}$: Mean Average Precision at IoU threshold 0.50.
   - $\text{mAP}_{50-95}$: Primary metric averaged across IoU thresholds from 0.50 to 0.95 with step 0.05.
   - $\text{AP}_S, \text{AP}_M, \text{AP}_L$: Detection performance across small, medium, and large plant scales.
2. **IoU=0.5 Operational Analysis**:
   - **Precision**: Ratio of true detected ragwort plants to all detections (minimizing false alarms on harmless pasture weeds).
   - **Recall**: Ratio of detected ragwort plants to all ground truth plants (ensuring toxic plants are not missed).
   - **F1 Score**: Harmonic mean of Precision and Recall.
   - **Error Mining**: Automatic saving of False Positive (FP) and False Negative (FN) images for targeted active learning.

---

## 12. Related Repositories & Acknowledgments

- **Web Application & Dashboard**: [TU-BerlinB/Website](https://github.com/TU-BerlinB/Website) — Companion interactive platform and visualization interface.

This research and software repository was conducted as part of the **Erasmus+** academic partnership with **Technische Universität Berlin (TU Berlin)**. 

Special thanks to the open-source vision community and the creators of **Ultralytics YOLO**, **Meta AI Research (DINOv2/DINOv3)**, and **DEIMv2 / RT-DETR**.