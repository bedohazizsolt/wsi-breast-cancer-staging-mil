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


import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from openslide import OpenSlide
import h5py
#import staintools
#from tiatoolbox.tools import stainnorm
import warnings
# Convert RuntimeWarning to an exception
warnings.filterwarnings("error", category=RuntimeWarning)
from matplotlib import patches
from shapely.plotting import plot_polygon


class MaskWSI_fp:

    def __init__(self, slide_path, patch_level, predefined_mask_dir, macenko=False) -> None:
        self.slide_path = slide_path
        self.slide_dir_path = os.path.dirname(os.path.realpath(self.slide_path))
        self.slide_name = os.path.splitext(os.path.basename(self.slide_path))[0]
        self.patch_level = patch_level
        self.macenko = macenko
        self.predefined_mask_dir = predefined_mask_dir
    
    def calc_mask_improved(self, patch_size, mask_extrapolate=False, ADAPT=-1, verbose=0):

        slide_openslide = OpenSlide(self.slide_path)
        #mask_level = 7 if slide_openslide.level_count-1 >= 7 else slide_openslide.level_count-1
        mask_level = 6 if slide_openslide.level_count-1 >= 6 else slide_openslide.level_count-3
        img_openslide = slide_openslide.read_region((0, 0), mask_level, slide_openslide.level_dimensions[mask_level])
        img_openslide_RGB = img_openslide.convert("RGB")
        img_openslide_np = np.array(img_openslide_RGB)

        if self.macenko:
            #target_image = np.load('reference_image_level2.npy')
            target_image = np.load('reference_image_level7.npy')
            #stain_normalizer = staintools.StainNormalizer(method='macenko')
            stain_normalizer = stainnorm.MacenkoNormalizer()
            try:
                stain_normalizer.fit(target_image)
                img_openslide_np_macenko = stain_normalizer.transform(img_openslide_np.copy())
                img_openslide_np = img_openslide_np_macenko
            except RuntimeWarning as e:
                print(f"Caught a RuntimeWarning during mask normalization, skipping it: {e}")


        img_openslide_np_flattened = img_openslide_np.flatten()
        white_filt = np.logical_and( img_openslide_np.flatten() > 200, img_openslide_np_flattened < 248 ) 
        hist_vals, bin_vals = np.histogram( img_openslide_np_flattened[white_filt], bins=46 )
        
        
        # old pproach
        #target_white = bin_vals[ np.argmax( hist_vals ) ] #- 3
        
        # new adaptive
        hist_vals_max_idx = (np.nanpercentile(hist_vals, 80) < hist_vals)
        target_white = np.min( bin_vals[:-1][hist_vals_max_idx] )
        
        
        if verbose:
            print('Resolution of image to extract:', slide_openslide.level_dimensions[self.patch_level])
            print('Current mask level:', mask_level )
            print('Mask image shape:', img_openslide_np.shape)
            plt.figure()
            plt.imshow(img_openslide_np)
            print( 'Target white on current image:', target_white )

        #if self.patch_level != 0:
        scale_factor = int( 2 ** self.patch_level * (patch_size / 2**(mask_level)) )
        
        if verbose:
            print('Scale factor:', scale_factor)

        # check which dim is small and modify if needed 
        width_new = scale_factor if img_openslide_np.shape[1] < scale_factor else img_openslide_np.shape[1] #if img_openslide_np.shape[0] < scale_factor else width
        height_new = scale_factor if img_openslide_np.shape[0] < scale_factor else img_openslide_np.shape[0] #if img_openslide_np.shape[1] < scale_factor else height
        
        #if width_new != width or height_new != height:
        img_openslide_np = cv2.resize(
                img_openslide_np, (width_new, height_new), interpolation=cv2.INTER_AREA
            )
        
        if patch_size % 2**mask_level != 0: # and not float.is_integer(scale_factor):
            img_openslide_np = img_openslide_np[
                : int(np.floor((img_openslide_np.shape[0] // scale_factor) * scale_factor)),
                : int(np.floor((img_openslide_np.shape[1] // scale_factor) * scale_factor)),
                :,
            ]
        else:
            img_openslide_np = img_openslide_np[
                : int((img_openslide_np.shape[0] // scale_factor) * scale_factor),
                : int((img_openslide_np.shape[1] // scale_factor) * scale_factor),
                :,
            ]
            
        width, height = (
            int(img_openslide_np.shape[1] // scale_factor),
            int(img_openslide_np.shape[0] // scale_factor),
        )

        if mask_extrapolate:
            width, height = width + 1, height +1
        
        img_openslide_np = cv2.resize(
            img_openslide_np, (width, height), interpolation=cv2.INTER_AREA   # HERE +1
        )
        
        if verbose:
            print('scale factor:', scale_factor, 'width:', width, 'height:', height)
        
        current_image = img_openslide_np

        white_start=target_white
        if verbose:
            print('Mask downsample image shape:', current_image.shape, current_image.mean())
            plt.figure()
            plt.imshow(current_image)

        # which color channel to choose
        percentiles = np.zeros((3,99))
        perc_limits = np.ones(3, dtype=int)*-1


        for ch in range(percentiles.shape[0]):
            color_hist = np.cumsum(np.histogram(current_image[:,:,ch].flatten(), bins=np.arange(0,257), density=True)[0])

            for p in range(percentiles.shape[1]):

                perc_threshold = np.arange(256)[color_hist < p*0.01 + 0.01]
                if len(perc_threshold) > 0:
                    percentiles[ch, p] = np.max(perc_threshold)


            perc_filter = percentiles[ch, :] < white_start

            if np.sum(perc_filter) >0:
                perc_limits[ch] = np.argmax( percentiles[ch, np.arange(percentiles.shape[1])[ percentiles[ch, :] < white_start ] ] )+ADAPT 


        logical_filters = []

        mask_img = np.zeros((current_image.shape[0], current_image.shape[1]))
       
        if mask_img.flatten().shape[0] < 5: # give back whole image because it is small
            return np.argwhere( np.isclose(np.ones((current_image.shape[0], current_image.shape[1])), 1.) ) , width, height
            
        if (perc_limits != -1).sum():

            for i in np.arange(3)[perc_limits != -1]:
                logical_filters.append(np.logical_and( current_image[:,:,i], current_image[:, :, i] < percentiles[i, perc_limits[i]] ))

            if len(logical_filters) == 1:
                mask = np.array(np.where(logical_filters[0])).T
                mask_img[mask[:,0], mask[:,1]] = 1

            if len(logical_filters) == 2:
                mask = np.array(np.where(np.logical_or(logical_filters[0], logical_filters[1]))).T
                mask_img[mask[:,0], mask[:,1]] = 1

            if len(logical_filters) == 3:
                mask = np.array(np.where(np.logical_or(np.logical_or(logical_filters[0], logical_filters[1]) , logical_filters[2] ))).T
                mask_img[mask[:,0], mask[:,1]] = 1
            
            if mask.shape[0] < 1: # empty based on the filtering process
                mask_img_sum_sorted = np.sort( current_image.sum(2).flatten() )
                threshold_percentile = np.percentile( mask_img_sum_sorted, 30 )
                mask = np.argwhere( current_image.sum(2) < threshold_percentile )

            return mask, width, height

        else:
            return np.argwhere( np.isclose(np.ones((current_image.shape[0], current_image.shape[1])), 1.) ) , width, height


    def load_mask_from_disk(self, patch_size, mask_extrapolate=False):

        pass


    
    def get_mask(self, patch_size, mask_extrapolate=False, load_mask=False, ADAPT=-1, verbose=0):
        if load_mask:
            raise Exception("No predefined masks available for this project")
        else:
            print("Calculating mask...")
            return self.calc_mask_improved(patch_size=patch_size, mask_extrapolate=mask_extrapolate, ADAPT=ADAPT, verbose=verbose)


    def save_thumbnail_with_patches(self, slide_name, coords, patch_size, level, output_path):
        patchlev = 0 # as patch coords readed in are level 0 coords
        vislev = 6

        PATCH_SIZE = patch_size
        LEVEL = level

        wsi = OpenSlide(slide_name)
        img_to_vis = np.array(wsi.read_region((0,0), vislev, wsi.level_dimensions[vislev] ).convert("RGB"))
        
        #fig = plt.figure(figsize=(6, 6), dpi=200)
        fig = plt.figure(figsize=(wsi.level_dimensions[vislev][0]/200, wsi.level_dimensions[vislev][1]/200), dpi=200)
        ax = fig.add_subplot(111)
        
        coords_viz =  coords // 2**(vislev-patchlev) # bring coords to vislev
        
        for idx, c in enumerate(coords_viz):
        
            rect = patches.Rectangle((c[0], c[1]), ( (PATCH_SIZE*2**LEVEL) //2**(vislev-patchlev)),
                                      ( (PATCH_SIZE*2**LEVEL) //2**(vislev-patchlev)),
                                      facecolor='none', edgecolor='black', linestyle='--', linewidth=0.75)
            ax.add_patch(rect)
            
            # Add patch number as text in the top left corner of the rectangle
            ax.text(c[0], c[1]+(PATCH_SIZE//2**(vislev-patchlev)), str(idx), fontsize=5, color='black', weight='bold')
            
            # Add coords
            # commented out because plot is too crowded
            #ax.text(c[0], c[1], f'({c[0]},{c[1]})', fontsize=8, color='blue')
            
        ax.imshow(img_to_vis)
        ax.axis('off')
    
        plt.savefig(output_path)
        plt.close(fig)
        
        
    def save_hdf5(self, output_path, asset_dict, attr_dict=None, mode='w'):
        file = h5py.File(output_path, mode)
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():   
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
        file.close()
        return output_path