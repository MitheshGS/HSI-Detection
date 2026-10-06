# HSI-Detection: Hyperspectral Object Detection with S2ADet

A practical toolkit built around **S2ADet** (Spectral-Spatial Aggregation Detector) on a **YOLOv5** backbone. It covers the full path from raw hyperspectral sensor files to a YOLOv5-ready dataset, plus tools to look inside the network and analyze how its attention is split between shape and material cues.

This repository accompanies the paper:

> **Object Detection in Hyperspectral Image via Unified Spectral-Spatial Feature Aggregation**
> Xiao He, Chang Tang, Xinwang Liu, Wei Zhang, Kun Sun, Jiangfeng Xu
> arXiv:2306.08370 · IEEE TGRS 2023 · Official code and dataset: <https://github.com/hexiao-cs/S2ADet>

## Table of Contents

- [What this repo adds](#what-this-repo-adds)
- [Installation](#installation)
- [1. Viewing raw HSI data (.hdr / .bil)](#1-viewing-raw-hsi-data-hdr--bil)
- [2. Compressing 16 spectral bands to 3 channels](#2-compressing-16-spectral-bands-to-3-channels)
- [3. Model training](#3-model-training)
- [4. SSA attention score analysis](#4-ssa-attention-score-analysis)
- [About the base paper (S2ADet)](#about-the-base-paper-s2adet)
- [Citation](#citation)

## What this repo adds

| Stage | What it does | Script |
|-------|--------------|--------|
| View | Load raw ENVI `.hdr` / `.bil` data and display a single spectral band | `viewhsi.py` |
| Compress | Extract the 16-band cube and reduce it to 3 channels with PCA | `compress.py`, `finalviewhsi.py` |
| Train | Train S2ADet on the compressed dataset | `train.py` |
| Analyze | Intercept Layer 26 and plot the SSA softmax attention split | `extract_attention.py` |

## Installation

```bash
pip install torch torchvision numpy matplotlib scipy h5py spectral scikit-learn
```

## 1. Viewing raw HSI data (.hdr / .bil)

Hyperspectral sensors save data as ENVI files: a `.hdr` header describing the layout and a `.bil` binary file holding the pixel values. `viewhsi.py` loads the cube and shows one band as a grayscale image so you can confirm the data is read correctly before any processing.

```bash
python viewhsi.py
```

Example run on a 168-band capture:

```text<img width="1458" height="1269" alt="Screenshot From 2026-10-06 11-24-07" src="https://github.com/user-attachments/assets/71de0a50-3104-4249-945d-0c3bf5e55c33" />

Successfully loaded HSI Data Cube. Shape: (400, 320, 168)
```

![Raw HSI band 100](view_raw_hsi_band100.png)

*Band 100 of the raw cube, shown in grayscale. Different materials reflect differently at each wavelength, which is exactly the signal an RGB camera throws away.*

## 2. Compressing 16 spectral bands to 3 channels

YOLOv5 expects 3-channel input, so a `(Height, Width, 16)` cube cannot be fed in directly. The pipeline uses **Principal Component Analysis (PCA)** to keep the directions of highest variance across the spectral bands and compress them into exactly 3 channels, giving a `(Height, Width, 3)` false-color image that works with the standard YOLOv5 dataloader.

```bash
# Read raw binary data, extract the 16-band cube, apply PCA, show before/after
python finalviewhsi.py

# Same PCA step on a generic 16-band cube
python compress.py
```

Example run on real sensor data:

```text
Loading raw binary data...
Extracted 16-band cube shape: (400, 320, 16)
Applying PCA Dimensionality Reduction (16 -> 3 channels)...
Final Compressed Image Shape for YOLOv5: (400, 320, 3)
```

![16 band to 3 channel PCA result](pca_16_to_3_400x320.png)

*Left: one of the 16 raw bands in grayscale. Right: the 3-channel PCA composite. Objects that look similar in a single band separate into distinct false colors.*

Quick shape check on a `(256, 256, 16)` test cube:

```text
Original Raw HSI Shape: (256, 256, 16)
Applying PCA Dimensionality Reduction (16 -> 3 channels)...
Final Compressed Image Shape for YOLOv5: (256, 256, 3)
```

![PCA shape check](<img width="1638" height="990" alt="attention_softmax_layer26" src="https://github.com/user-attachments/assets/3c02b1a3-c833-4538-a8fa-453131d29a8d" /><img width="939" height="1168" alt="view_raw_hsi_band100" src="https://github.com/user-attachments/assets/e698e0d4-06e5-4db6-a93d-dc533b60624e" />
<img width="1304" height="656" alt="pca_sanity_check_256x256" src="https://github.com/user-attachments/assets/c7878a9c-8d63-4160-8978-fd3b46c2e45f" />
<img width="1397" height="775" alt="pca_16_to_3_400x320" src="https://github.com/user-attachments/assets/9804956d-4ae2-4b35-b41d-f5dae80e1275" />
.png)

> **Note:** this script covers the spectral (PCA) half of the paper's Hyperspectral Information Decoupling module. The paper additionally builds a second, spatial input by selecting informative bands (optimal neighborhood reconstruction) and color mapping them. See [About the base paper](#about-the-base-paper-s2adet).

## 3. Model training

Train on the compressed 3-channel images. Point the dataset `.yaml` at the folder containing the false-color PNGs.

```bash
python train.py --img 640 --batch 4 --epochs 100 --data data/hyperspectral.yaml --weights s2adet_best.pt
```

> Batch size can be tuned in `opt.yaml` and `hyp.yaml` depending on your GPU VRAM.

## 4. SSA attention score analysis

S2ADet fuses spectral and spatial features through its **Spectral-Spatial Aggregation (SSA)** module. To see how the network distributes its attention, `extract_attention.py` intercepts the computational graph at **Layer 26** and plots the softmax weights, which sum to 1.0 for every feature channel, across three streams:

| Stream | Meaning |
|--------|---------|
| **Spatial** | Geometry, edges and structural shape |
| **Spectral** | Material reflectance, useful for telling apart objects that look alike but are made of different materials |
| **Joint** | Cross-fused consensus of shape and material |

```bash
python extract_attention.py --source test_image.png
```

![SSA softmax attention distribution](attention_softmax_layer26.png)

### Reading the plot

Five sampled deep-feature channels from Layer 26 (approximate values read from the chart):

| Channel | Dominant stream | Rough split (spatial / spectral / joint) |
|---------|-----------------|------------------------------------------|
| 34 | Spectral | 0.04 / 0.59 / 0.37 |
| 49 | Spatial | 0.95 / 0.00 / 0.05 |
| 80 | Spectral | 0.20 / 0.51 / 0.29 |
| 93 | Spatial and Joint | 0.50 / 0.01 / 0.49 |
| 636 | Spectral | 0.02 / 0.66 / 0.32 |

**Takeaway:** the network does not favor one stream globally. Some channels specialize in shape (channel 49), others in material signature (channels 34 and 636), and some lean on the joint fusion (channel 93). This is the dynamic routing behavior the two-stream design is meant to produce.

## About the base paper (S2ADet)

Most hyperspectral detectors use either spectral or spatial information. S2ADet uses both and cuts redundancy between neighboring bands.

### Architecture

1. **HID (Hyperspectral Information Decoupling):** splits the cube into two inputs.
   - **SE (spectral) information:** PCA over the spectral dimension.
   - **SA (spatial) information:** band selection via optimal neighborhood reconstruction (ONR), then color mapping.
2. **Two-stream backbone:** DarkNet-FPN, five stages, one stream per input.
3. **SSA (Spectral-Spatial Aggregation):** inserted in the last three stages (outputs S3, S4, S5). It concatenates both streams, runs spatial-reduction attention so the two streams exchange context, and adds a channel/spatial attention module (SAM, CBAM style). Aggregated features are added back into each stream.
4. **One-stage detection head** with loss `L = L_cls + L_box`.

### HOD3K dataset

Introduced by the paper, captured with a XIMEA snapshot VIS camera.

| Property | Value |
|----------|-------|
| Images | 3,242 |
| Resolution | 512 × 256 |
| Bands | 16 (470 nm to 620 nm) |
| Scenes | Urban roads, campuses, residential areas |
| Classes | People (12,144), Bike (2,188), Car (817) |
| Total boxes | 15,149 |
| Split | 7 : 1 : 2 (train / val / test) |

### Results reported in the paper

**HOD3K (512 × 256)**

| Model | Input | mAP50 | mAP |
|-------|-------|-------|-----|
| YOLOv5 baseline | SA | 88.1 | 54.4 |
| YOLOv5 (SA + SE) | SA + SE | 91.7 | 56.3 |
| Faster R-CNN | SA | 89.4 | 56.9 |
| **S2ADet** | SA + SE | **93.4** | **59.8** |

Per class mAP50 for S2ADet: people 87.2, bike 97.7, car 95.3.

**HOD-1:** S2ADet reaches 86.6 mAP versus 83.5 for the original HOD-1 method, while using an input one-sixteenth the size.

**SSA ablation (HOD3K mAP):** baseline 54.4, then 56.3 with HID only, 57.1 adding SAM, 58.9 adding SSA, and 59.8 with both.

### Training setup used in the paper

DarkNet-FPN backbone with DarkNet-50 pretrained initialization, 50 epochs, mosaic augmentation, SGD with learning rate 0.01 and a poly schedule (power 0.9), NMS IoU 0.6, single NVIDIA RTX 3090.

### Known limitations

The paper reports that S2ADet struggles with overlapping targets and small objects, and that fine-grained classification still needs work (for example, the "pen screen" class on HOD-1 reaches only 0.31 accuracy).

## Citation

```bibtex
@article{he2023s2adet,
  title   = {Object Detection in Hyperspectral Image via Unified Spectral-Spatial Feature Aggregation},
  author  = {He, Xiao and Tang, Chang and Liu, Xinwang and Zhang, Wei and Sun, Kun and Xu, Jiangfeng},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2023},
  note    = {arXiv:2306.08370}
}
```
