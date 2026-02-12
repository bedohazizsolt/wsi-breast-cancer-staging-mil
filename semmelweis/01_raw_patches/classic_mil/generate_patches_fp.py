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
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys
import glob
from tqdm import tqdm
import argparse

sys.path.append('..')
from maskWSI_fp import MaskWSI_fp
from omegaconf import OmegaConf
from openslide import OpenSlide


def filter_patches_by_black_pixel_count(wsi_path, coords, level=6, patch_size=(112, 112), black_pixel_threshold=50):
    wsi = OpenSlide(wsi_path)
    filtered_coords = []

    for coord in coords:
        region = wsi.read_region(coord, level, patch_size).convert("RGB")
        array = np.array(region)
        
        black_pixels = np.sum(np.all(array == [0, 0, 0], axis=2))
        
        if black_pixels <= black_pixel_threshold:
            filtered_coords.append(coord)

    return np.array(filtered_coords)



def main(idx_to_run_now):
    ## extract patches with own method
    for s in tqdm( range(idx_to_run_now[0], idx_to_run_now[1]) ):
        
        if not os.path.exists( PATCH_SAVE_DIR+os.path.basename(slides[s]).replace(FILE_EXTENSION, '.h5') ):
            
            print('Processing: ', slides[s])
            patch_level = LEVEL # preset value
            patch_size = PATCH_SIZE # preset value

            mask_wsi_fp = MaskWSI_fp( slide_path=slides[s], patch_level=patch_level, predefined_mask_dir=None, macenko=False )

            mask, w, h = mask_wsi_fp.get_mask(patch_size, mask_extrapolate=False, load_mask=False, ADAPT=MASKING_ADAPT, verbose=0)  # original -1
            
            # mask comes in shape: (N,2) and is in order (y, x) -> needs to be flipped -> vstack().T does the job

            # transform info to save into h5 files to enable on-the-fly patching (at inference)
            output_path = PATCH_SAVE_DIR + os.path.basename(slides[s]).replace(FILE_EXTENSION, '.h5')
            coords = np.vstack( (mask[:,1], mask[:,0]) ).T*patch_size*2**patch_level #  coords should be given on level0 !

            # filter artifacts
            coords = filter_patches_by_black_pixel_count(slides[s], coords);
            
            asset_dict = {'coords': coords}
            attr_dict = {'patch_size': patch_size, 'patch_level': patch_level}
            attr_dict = {'coords' : attr_dict}
            mask_wsi_fp.save_hdf5(output_path, asset_dict, attr_dict) # open in write mode (all data is here)

            thumbnail_output_path = MASK_THUMBNAIL_DIR + os.path.basename(slides[s]).replace(FILE_EXTENSION, '.jpg')

            print('Savng thumbnail: ', slides[s] )
            mask_wsi_fp.save_thumbnail_with_patches(slide_name=slides[s], coords=coords, patch_size=patch_size, level=patch_level, output_path=thumbnail_output_path)
        
        
        else:
            print("Already done: ", slides[s] )


        #break

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_thread", type=int, required=False, default=0)
    parser.add_argument("--maxnum_threads", type=int, required=False, default=8)
    parser.add_argument("--config_level_0", type=str, required=False, default='HIPT')

    args = parser.parse_args()
    num = args.num_thread
    maxnum = args.maxnum_threads
    config_level_0 = args.config_level_0


    # Load config
    conf_preproc = OmegaConf.load("../conf/preproc.yaml")
    conf_preproc = conf_preproc[config_level_0]
    
    DATA_ROOT = conf_preproc.data_root_dir
    FILE_EXTENSION = conf_preproc.wsi_extension
    LEVEL = conf_preproc.patch_level
    PATCH_SIZE = conf_preproc.patch_size
    MASKING_ADAPT = conf_preproc.masking_adapt
    PATCH_SAVE_DIR = conf_preproc.mask_save_dir
    MASK_THUMBNAIL_DIR = conf_preproc.mask_thumbnail_save_dir
    os.makedirs(PATCH_SAVE_DIR, exist_ok=True)
    os.makedirs(MASK_THUMBNAIL_DIR, exist_ok=True)

    # Load slide names
    slides = np.sort( np.array( glob.glob( os.path.join(DATA_ROOT, f"*{conf_preproc.wsi_extension}") ) ) )
    print('\nNr. of slides found for inference: ', len(slides))
    
    # Get index splits
    idx_to_run = np.append(np.arange( 0, len(slides), len(slides)/maxnum  ).astype(int), len(slides))
    idx_to_run_all = np.vstack( (idx_to_run[:-1], idx_to_run[1:]) ).T
    idx_to_run_now = idx_to_run_all[num]

    print( 'start:', idx_to_run_now[0], 'end:', idx_to_run_now[1], slides[ idx_to_run_now[0]:idx_to_run_now[1] ].shape ) 
    
    main(idx_to_run_now)