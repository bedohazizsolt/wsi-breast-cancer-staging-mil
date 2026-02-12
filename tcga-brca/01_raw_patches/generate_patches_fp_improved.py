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
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('..')
from maskWSI_fp_improved import MaskWSI_fp
from omegaconf import OmegaConf
from openslide import OpenSlide


    
def filter_patches_by_tissue_presence(wsi_path, coords, level=4, patch_size=(224, 224), save_patches=False, save_dir='patches'):
    """
    224: wsi_path, coords, level=2, patch_size=(224, 224), save_patches=False, save_dir=None
    """
    wsi = OpenSlide(wsi_path)
    filtered_coords = []

    if save_patches:
        os.makedirs(save_dir, exist_ok=True)
    
    for i, coord in enumerate(coords):
        patch = wsi.read_region(coord, level, patch_size).convert("RGB")
        patch = np.array(patch)
        
        #if save_patches:
        #    save_path = os.path.join(save_dir, f'patch_{i}.png')
        #    save_patch(patch, save_path)
    
        # Convert the patch to the grayscale
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 1)  # 11,2
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8) #5,5
        morph_patch = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        if save_patches:
            save_intermediate_results(patch, gray_patch, adaptive_thresh, morph_patch, save_dir, i)
        
        # Calculate the percentage of the patch that is considered tissue
        tissue_ratio = np.sum(morph_patch) / (morph_patch.shape[0] * morph_patch.shape[1] * 255)

        # Use Canny edge detection to find edges in the patch
        edges = cv2.Canny(gray_patch, 50, 150, 3)  # Lower thresholds for Canny edge detection 50,150,3

        # Calculate the edge density
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)

        # Determine if the patch is a tissue patch
        if tissue_ratio > 0.075 and edge_density > 0.040:
            filtered_coords.append(coord)
        
    return np.array(filtered_coords)


def filter_patches_by_tissue_presence_improved(wsi_path, coords, level=4, patch_size=(512, 512), save_patches=False, save_dir='patches'):
    wsi = OpenSlide(wsi_path)
    filtered_coords = []

    if save_patches:
        os.makedirs(save_dir, exist_ok=True)
    
    for i, coord in enumerate(coords):
        patch = wsi.read_region(coord, level, patch_size).convert("RGB")
        patch = np.array(patch)
        
        # Convert the patch to the grayscale
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # Convert the patch to HSV color space to detect black regions
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        
        # Mask black regions (low V in HSV)
        black_mask = cv2.inRange(hsv_patch, (0, 0, 0), (180, 255, 40))  # Detect dark areas
        
        # Apply adaptive thresholding for tissue detection
        adaptive_thresh = cv2.adaptiveThreshold(gray_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 1)
        
        # Apply morphological operations to close gaps and remove small artifacts
        kernel = np.ones((3, 3), np.uint8)
        morph_patch = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Detect vertical lines with morphological operation
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))  # Tall and narrow for vertical lines
        vertical_lines = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, vertical_kernel)

        # Detect horizontal lines with a separate morphological operation
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))  # Wide and flat for horizontal lines
        horizontal_lines = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, horizontal_kernel)

        # Combine vertical and horizontal line detections
        combined_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
        
        if save_patches:
            save_intermediate_results_improved(patch, gray_patch, adaptive_thresh, morph_patch, black_mask, vertical_lines, horizontal_lines, combined_lines, save_dir, i)
        
        # Calculate the percentage of the patch that is considered tissue
        tissue_ratio = np.sum(morph_patch) / (morph_patch.shape[0] * morph_patch.shape[1] * 255)
        
        # Calculate the percentage of black areas and line artifacts
        black_area_ratio = np.sum(black_mask) / (black_mask.shape[0] * black_mask.shape[1] * 255)
        line_artifact_ratio = np.sum(combined_lines) / (combined_lines.shape[0] * combined_lines.shape[1] * 255)

        # Use Canny edge detection to find edges in the patch
        edges = cv2.Canny(gray_patch, 50, 150, 3)  # Lower thresholds for Canny edge detection

        # Calculate the edge density
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)

        # Filter out patches with too much black area or line artifacts
        if tissue_ratio > 0.075 and edge_density > 0.040 and black_area_ratio < 0.1 and line_artifact_ratio < 0.1:
            filtered_coords.append(coord)
        
    return np.array(filtered_coords)


def main(idx_to_run_now, num):
    ## extract patches with own method
    for s in tqdm( range(idx_to_run_now[0], idx_to_run_now[1]) ):
        
        
        if not os.path.exists( PATCH_SAVE_DIR+os.path.basename(slides[s]).replace(FILE_EXTENSION, '.h5') ):
            
            
            print('\n')
            print('\nProcessing: ', slides[s])
            patch_level = LEVEL # preset value
            patch_size = PATCH_SIZE # preset value

            mask_wsi_fp = MaskWSI_fp( slide_path=slides[s], patch_level=patch_level, predefined_mask_dir=None, macenko=False )

            mask, w, h, _, _ = mask_wsi_fp.get_mask(patch_size, mask_extrapolate=False, load_mask=False, ADAPT=MASKING_ADAPT, verbose=0)  # original -1
            
            # mask comes in shape: (N,2) and is in order (y, x) -> needs to be flipped -> vstack().T does the job

            # transform info to save into h5 files to enable on-the-fly patching (at inference)
            output_path = PATCH_SAVE_DIR + os.path.basename(slides[s]).replace(FILE_EXTENSION, '.h5')

            wsi = OpenSlide(slides[s])
            #coords = np.vstack( (mask[:,1], mask[:,0]) ).T*patch_size*2**patch_level #  coords should be given on level0 !  BUGGY
            coords = np.vstack( (mask[:,1], mask[:,0]) ).T*PATCH_SIZE*int(wsi.level_downsamples[ patch_level ])     #  coords should be given on level0 !
            

            
            coords_v2 = filter_patches_by_tissue_presence_improved(slides[s], coords, level=2, patch_size=(224, 224), save_patches=False, save_dir=None)
            
            #print("DIFFERENCE: ", coords_v1.shape[0] - coords_v2.shape[0])
            if coords_v2.shape[0] != 0:
                coords = coords_v2
            
            asset_dict = {'coords': coords}
            attr_dict = {'patch_size': patch_size, 'patch_level': patch_level}
            attr_dict = {'coords' : attr_dict}
            mask_wsi_fp.save_hdf5(output_path, asset_dict, attr_dict) # open in write mode (all data is here)
  
        
        else:
            print("Already done: ", slides[s] )

        #break




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_thread", type=int, required=False, default=0)
    parser.add_argument("--maxnum_threads", type=int, required=False, default=8)
    parser.add_argument("--config_level_0", type=str, required=False, default='classic_mil_on_embeddings_bag')
    parser.add_argument("--config_level_1", type=str, required=False, default='tcga_brca_224_224_patches_jmcs')

    args = parser.parse_args()
    num = args.num_thread
    maxnum = args.maxnum_threads
    config_level_0 = args.config_level_0
    config_level_1 = args.config_level_1
    
    # Load config
    conf_preproc = OmegaConf.load("../conf/preproc.yaml")
    conf_preproc = conf_preproc[config_level_0][config_level_1]
    
    DATA_ROOT = conf_preproc.slide_root_dir
    FILE_EXTENSION = conf_preproc.wsi_extension
    LEVEL = conf_preproc.patch_level
    PATCH_SIZE = conf_preproc.patch_size
    MASKING_ADAPT = conf_preproc.masking_adapt
    
    PATCH_SAVE_DIR = conf_preproc.mask_dir
    print('LEVEL:', LEVEL, '\n', 'PATCH_SIZE:', PATCH_SIZE)
    os.makedirs(PATCH_SAVE_DIR, exist_ok=True)

    
    # Load all slide on disk
    #slides = np.sort( np.array( glob.glob( os.path.join(DATA_ROOT, f"*{conf_preproc.wsi_extension}") ) ) )
    #print('\nNr. of slides found for inference: ', len(slides))    



    # load only slides in cohort
    labels_df = pd.read_csv("../02_patch_embeddings/labels_df.csv") # (731, 3)
    
    slides = np.array([conf_preproc.slide_root_dir + s + ".svs" for s in labels_df.slide_id.values])
    print('\nNr. of slides found for inference: ', len(slides))
    
    # process only that are not already processed
    slides_done = np.sort(np.array( glob.glob( os.path.join(PATCH_SAVE_DIR, "*.h5")) ))
    slides_names = np.array([s.split('/')[-1].split('.')[0] for s in slides])
    slides_done_names = np.array([s.split('/')[-1].split('.')[0] for s in slides_done])
    slides_not_done = slides[np.isin(slides_names, slides_done_names, invert=True)]
    print('\nNr. of remaining slides found for inference: ', len(slides_not_done))
    slides = slides_not_done



    
    
    # Get index splits
    idx_to_run = np.append(np.arange( 0, len(slides), len(slides)/maxnum  ).astype(int), len(slides))
    idx_to_run_all = np.vstack( (idx_to_run[:-1], idx_to_run[1:]) ).T
    idx_to_run_now = idx_to_run_all[num]

    print( 'start:', idx_to_run_now[0], 'end:', idx_to_run_now[1], slides[ idx_to_run_now[0]:idx_to_run_now[1] ].shape ) 
    
    main(idx_to_run_now, num)