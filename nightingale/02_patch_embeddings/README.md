## This folder contains the scripts to generate the embeddings of 224x224x3 patches.



### 1. Generate embeddings with UNI-finetuned feature extractor: 

For this, the extracted patches has to be set to the ones extracted with Maceno using: `reference_image_patches_level4_224_224_3_bracs.npy`

`python generate_finetuned_uni_embeddings.py --config_level_0  classic_mil_on_embeddings_bag --config_level_1 nightingale_224_224_patches --maxnum_threads 8 --num 2 --norm macenko`


### 2. Collect embeddings for patients:

`create_biopsy_bags_from_embeddings_patients.ipynb`
