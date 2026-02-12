## This folder contains codes to generate coords of patches of size 224x224x3 from the original WSIs and extract+save 224x224x3 images.  


### 1. Use color space masking to generate coords of non-background patches:

`generate_patches_fp_improved.py`

Set the following variables prior running:

DATA_ROOT: point to raw data location

DATA_WRITE: point to location to write masks of WSIs


### 2. Extract patches with Macenko stain normalization:

`generate_patches_224_stainnorm.py`


The reference image `reference_image_patches_level4_224_224_3_bracs.npy` was calculated locally on the bracs dataset where the UNI-finetuned backbone were created and was used alongside this script to extract patches for embedding with the UNI-finetuned backbone.
