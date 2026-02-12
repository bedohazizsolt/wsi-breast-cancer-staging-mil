"""
Copyright 2026 Zsolt Bedőházi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os 
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import pickle
import sys
import glob

# LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from openslide import OpenSlide
import argparse

#from tiatoolbox import utils
#from tiatoolbox.wsicore import wsireader
#from tiatoolbox import data
from tiatoolbox.tools import stainnorm

from omegaconf import OmegaConf



def load_h5_file(filename):
    with h5py.File(filename, "r") as f:
        coords = f['coords'][()]
        patch_level = f['coords'].attrs['patch_level']
        patch_size = f['coords'].attrs['patch_size']
        return coords, patch_level, patch_size
        
def save_h5_file(filename, imgs, coords):
    with h5py.File(filename, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("imgs", data=imgs)
        
def create_all_224_224_3_patches_for_a_slide(h5_filepath, stain_normalizer):

    wsi_file_path = SOURCE_TIF + os.path.basename(h5_filepath).replace('.h5', FILE_EXTENSION)
    coords, patch_level, patch_size = load_h5_file(SOURCE_H5+h5_filepath)

    file_name_w_ext = os.path.basename(h5_filepath)
    filename = DEST_PATCH_DIR + file_name_w_ext
    
    if not os.path.exists(DEST_PATCH_DIR + file_name_w_ext):
    
        img_all = np.zeros( (len(coords), 224, 224, 3), dtype=np.uint8 )
        img_all.fill(0)
        
        wsi = OpenSlide(wsi_file_path)
        for c in range(len(coords)):
            img_224 = np.array(wsi.read_region(coords[c], patch_level, (patch_size, patch_size)).convert('RGB'))
            img_all[c] = img_224
            
        ## HERE TRANSFORM STAIN
        img_all_normed = np.zeros(img_all.shape, dtype=np.uint8)
        for c in range(img_all_normed.shape[0]):
            img_all_normed[c] = stain_normalizer.transform(img_all[c].copy())

        save_h5_file(filename, img_all_normed, coords)
    else:
        print('Already done!')



parser = argparse.ArgumentParser()
parser.add_argument("--num_thread", type=int, required=False, default=0)
parser.add_argument("--maxnum_threads", type=int, required=False, default=1)
parser.add_argument("--config_level_0", type=str, required=False, default='classic_mil_on_embeddings_bag')
parser.add_argument("--config_level_1", type=str, required=False, default='nightingale_224_224_patches')

args = parser.parse_args()
num = args.num_thread
maxnum = args.maxnum_threads
config_level_0 = args.config_level_0
config_level_1 = args.config_level_1        

# Load config
preproc_conf = OmegaConf.load("../conf/preproc.yaml")
preproc_conf = preproc_conf[config_level_0][config_level_1]

# HARCODED PATHS:
SOURCE_H5 = preproc_conf.mask_dir_improved_v2
SOURCE_TIF = preproc_conf.data_root_dir
DEST_PATCH_DIR = preproc_conf.patch_dir_macenko_improved_v2

os.makedirs(DEST_PATCH_DIR, exist_ok=True)


files_fp = np.array( sorted( [ k for k in os.listdir( SOURCE_H5 ) if '.h5' in k ] ) )

idx_to_run = np.append(np.arange( 0, len(files_fp), len(files_fp)/maxnum  ).astype(int), len(files_fp))
idx_to_run_all = np.vstack( (idx_to_run[:-1], idx_to_run[1:]) ).T
idx_to_run_now = idx_to_run_all[num]

print( 'start:', idx_to_run_now[0], 'end:', idx_to_run_now[1], files_fp[ idx_to_run_now[0]:idx_to_run_now[1] ].shape ) 
stats_array_all = []

## ADD MACENKO NORMALIZER
reference_image = np.load(preproc_conf.reference_image_macenko)
stain_normalizer = stainnorm.MacenkoNormalizer()
stain_normalizer.fit(reference_image)

# DO PROCESS HERE:
for o in tqdm(range( idx_to_run_now[0],idx_to_run_now[1] )):
    create_all_224_224_3_patches_for_a_slide(files_fp[o], stain_normalizer)
