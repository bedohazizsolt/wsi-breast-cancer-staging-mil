# 01 — Patch Extraction from BRACS Whole-Slide Images

This folder contains the patch extraction pipeline for the **BRACS** (BReAst Carcinoma Subtyping) dataset,
used to prepare 224×224 pixel patches at **2.5× magnification** (~4.0 µm/pixel) for downstream backbone fine-tuning.


### BRACS-Specific Notebooks

| File | Description |
|------|-------------|
| **`bracs_save_level4equivalent.ipynb`** | Reads BRACS `.svs` files at OpenSlide level 2 (≡ 2.5× magnification with downsample ≈16) and saves them as `.npy` arrays. These arrays are later used for ROI-based patch extraction with physician annotations.
| **`bracs_extract_patches_from_roi_with_mask_level4.ipynb`** | Loads `.npy` arrays from the previous step and BRACS geojson annotation polygons, creates a 224×224 patch grid per slide, intersects patches with annotation polygons (using R-tree spatial indexing) to assign ROI class labels (7 classes: Normal, PB, UDH, FEA, ADH, DCIS, IC). Splits into train/val/test following BRACS official splits (with patient-level leak fix). Also computes the Macenko reference image from training patches and applies stain normalization. Output: labeled `.npy` arrays used directly by `02_backbone_finetuning/`.
| **`color_normalization_for_patches.ipynb`** | Computes the **median reference image** for Macenko normalization by analyzing color statistics (mean/std per BGR channel) across all extracted BRACS patches. Selects the patch closest to the median and saves it as `reference_image_patches_level4_224_224_3_bracs.npy`.
---