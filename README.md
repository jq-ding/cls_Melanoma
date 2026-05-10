# Melanoma Classification with Ensemble Learning

> **Two-Stage Deep Neural Network via Ensemble Learning for Melanoma Classification**
> 
> Jiaqi Ding, Jie Song, Jiawei Li, Jijun Tang, Fei Guo
> 
> *Frontiers in Bioengineering and Biotechnology*, 2022

## Overview

Skin lesion analysis is one of the most clinically impactful applications of medical image analysis, but dermoscopy images are challenging because the lesion of interest occupies only part of the field of view, and clinically distinct lesions can look visually similar. This work proposes a **two-stage ensemble framework** for melanoma classification that combines lesion segmentation, region-focused classification, and ensemble fusion.

**Key ideas**:
- **Stage 1 — Segmentation**: Use U-Net to segment the lesion area, generating a lesion mask used to crop and focus subsequent analysis on the diagnostically relevant region
- **Stage 2 — Multi-model classification**: Train five strong classification backbones (ResNet-50, DenseNet-169, Inception-V3, Inception-ResNet-V2, Xception) on the lesion-focused images, with Squeeze-and-Excitation (SE) blocks for feature emphasis
- **Ensemble fusion**: A learnable ensemble network (locally-connected layers + softmax) integrates the five classification outputs into a final prediction

The method achieves **0.909 accuracy** on the ISIC 2017 challenge dataset.

## Repository Structure

```
cls_Melanoma/
├── datasets/ISIC2018/      # Data directory
├── misc_utils/             # Utility functions
├── models/                 # Network definitions
├── seg_train.py            # Train segmentation network
├── seg_predict.py          # Predict lesion masks
├── resize.py               # Crop and resize lesion regions
├── training.py             # Train classification networks
├── cls_predict.py          # Classification inference
├── vote.py                 # Train and apply the ensemble fusion network
├── callback.py             # Training callbacks
├── paths.py                # Path configuration
└── requirements.txt
```

## Dataset

ISIC 2017 / 2018 Challenge dataset: https://challenge.isic-archive.com/data/#2017

## Setup

```bash
conda create -n melanoma python=3.8
conda activate melanoma
pip install -r requirements.txt
```

## Pipeline

### Step 1: Segment lesion regions

```bash
# Train the segmentation network
python seg_train.py

# Apply to test images
python seg_predict.py
```

### Step 2: Resize and crop to lesion area

```bash
python resize.py
```

This uses the segmentation masks to crop the original images to the lesion region, removing irrelevant background.

### Step 3: Train classification backbones

```bash
python training.py
```

Set the desired backbone in the script. Supported options:
- `resnet50`
- `densenet169`
- `inception_v3`
- `inception_resnet_v2`
- `xception`

### Step 4: Predict per backbone

```bash
python cls_predict.py
```

Specify the network name to evaluate. Save predictions for ensemble fusion.

### Step 5: Train and apply ensemble fusion

```bash
python vote.py
```

The ensemble network uses locally-connected layers to aggregate predictions across the five backbones:

```
inputs (150 x 3 x 5)  →  LocallyConnected1D(20)  →  LocallyConnected1D(1)  →  Softmax  →  output
```

This yields the final classification.

## Results

The paper provides full ablations including:
- Effect of segmentation-based cropping vs. raw image classification
- Performance per individual backbone
- Per-class accuracy on the two-class (melanoma vs. benign) and three-class challenges
- Comparison against ISIC 2017 challenge leaderboard methods

See the paper for the complete tables and figures.

## Citation

```bibtex
@article{ding2022two,
  title     = {Two-stage deep neural network via ensemble learning for melanoma classification},
  author    = {Ding, Jiaqi and Song, Jie and Li, Jiawei and Tang, Jijun and Guo, Fei},
  journal   = {Frontiers in Bioengineering and Biotechnology},
  volume    = {9},
  pages     = {758495},
  year      = {2022},
  publisher = {Frontiers Media SA}
}
```

## Contact

Jiaqi Ding — `jiaqid@cs.unc.edu`
[Personal website](https://jq-ding.github.io/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=5h5qru8AAAAJ)
