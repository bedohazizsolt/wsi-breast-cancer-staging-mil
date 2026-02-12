## This folder contains codes to extract patches of size 224x224x3 from the original WSIs.


### 1.) Use color space masking to get non-background patches:

`generate_patches_fp.py`

run with: 

`python generate_patches_fp.py --num_thread 0 --maxnum_threads 1 --config_level_0  classic_mil_on_embeddings_bag  --config_level_1  tcga_brca_224_224_patches`


### 2.) Extract patches with Macenko stain normalization:

`generate_patches_224_stainnorm.py`

run with:

`python generate_patches_224_stainnorm.py  --num_thread 0 --maxnum_threads 1 --config_level_0  classic_mil_on_embeddings_bag  --config_level_1 bracs_224_224_patches`

