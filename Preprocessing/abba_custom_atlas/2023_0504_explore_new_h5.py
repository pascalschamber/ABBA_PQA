import numpy as np
import os
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
from skimage.io import imread, imsave
import napari
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from time import time
import h5py
import json
from pprint import pprint
import scipy
from skimage.transform import rescale, resize, downscale_local_mean

def print_array_info(array):
    print(array.shape, array.min(), array.max(), array.dtype)

def print_atlas_info(atlas_h5, ch):
    # ch can be one of 's03'
    for res in ['0', '1', '2', '3']:
        print_array_info(np.array(atlas_h5['t00000'][ch][res]['cells']))

'''
####################################################################################################
# NOTES - 5/4/23
    on 5/1/23 they released a new version of ABBBA that replaced my previous ccf2017-mod65000-border-centered-mm-bc.h5
    # this neccesitated this script to see what changes if any occured
'''



img_dir_atlas = r'F:\ABBA\myAtlases'
img_name_atlas = 'ccf2017-mod65000-border-centered-mm-bc.h5'
new_img_name_atlas = 'thierNew0504_ccf2017-mod65000-border-centered-mm-bc.h5'
img_path_atlas = os.path.join(img_dir_atlas, img_name_atlas)
new_img_path_atlas = os.path.join(img_dir_atlas, new_img_name_atlas)

atlas_h5 = h5py.File(img_path_atlas)
new_atlas_h5 = h5py.File(new_img_path_atlas)

for ch in ['s00', 's01', 's02', 's03']:
    for atl in [atlas_h5, new_atlas_h5]:
        print_atlas_info(atl, ch=ch)


for attr in ['__DATA_TYPES__', 's00', 's01', 's02', 's03', 't00000']:
    for atl in [atlas_h5, new_atlas_h5]:
        keys = list(atl.get(attr).keys())
        for k in keys:
            print(atl.get(attr).get(k))
    print()


# img_labels_path = os.path.join(img_dir_atlas, 'ccf2017_labels_bla-subdivisions_20230302.tif')


# with h5py.File(img_path_atlas,'r+') as atlas_h5:
#     print_atlas_info(atlas_h5, ch='s03')
# mylabels = imread(img_labels_path)
# mylabels = np.moveaxis(mylabels, [0, 2], [2,0]).astype('int16')
# print_array_info(mylabels)
# mylabels2x = rescale(mylabels, 1/2, anti_aliasing=False, preserve_range=True).astype('int16')
# print_array_info(mylabels2x)
# mylabels4x = rescale(mylabels, 1/4, anti_aliasing=False, preserve_range=True).astype('int16')
# print_array_info(mylabels4x)
# mylabels8x = rescale(mylabels, 1/8, anti_aliasing=False, preserve_range=True).astype('int16')
# print_array_info(mylabels8x)

# # overwrite the data at a given resolution
# with h5py.File(img_path_atlas,'r+') as atlas_h5:
#     atlas_h5['t00000']['s03']['0']['cells'][...] = mylabels
#     atlas_h5['t00000']['s03']['1']['cells'][...] = mylabels2x
#     atlas_h5['t00000']['s03']['2']['cells'][...] = mylabels4x
#     atlas_h5['t00000']['s03']['3']['cells'][...] = mylabels8x

# atlas_h5 = h5py.File(img_path_atlas)
# new_h5_labels = np.array(atlas_h5['t00000']['s03']['0']['cells'])
# print_array_info(new_h5_labels)



# # (1140, 800, 1320) -23066 26977 int16
# # appears to be downscaled by a factor of 2
# # can't use antialiasing b/c it removes px values

# print(prev_labels_array.shape, prev_labels_array.min(), prev_labels_array.max(), prev_labels_array.dtype)
# downsampled2x = rescale(prev_labels_array, 1/8, anti_aliasing=False, preserve_range=True).astype('int16')
# print(downsampled2x.shape, downsampled2x.min(), downsampled2x.max(), downsampled2x.dtype)


# for res in ['0', '1', '2', '3']:
#     print_array_info(np.array(atlas_h5['t00000']['s03'][res]['cells']))
# # (1140, 800, 1320) -23066 26977 int16
# # (570, 400, 660) -32736 32693 int16
# # (285, 200, 330) -32767 32767 int16
# # (142, 100, 165) -32766 32766 int16

# for rf in [1/2, 1/4, 1/8][-1:]:
#     downsampled2x = rescale(prev_labels_array, 1/8, anti_aliasing=False, preserve_range=True).astype('int16')
#     print_array_info(downsampled2x)



# # channels = ['s00', 's01', 's02', 's03']
# # # cells = [atlas_h5['t00000'][ch][resolution]['cells'] for ch in channels]

# # print(atlas_h5.keys()) # <KeysViewHDF5 ['__DATA_TYPES__', 's00', 's01', 's02', 's03', 't00000']>

# # for el in ['s00', 's01', 's02', 's03']:
# #     print(el)
# #     for ind in ['0', '1', '2', '3']:
# #         print(atlas_h5['t00000'][el][ind]['cells'])
# #     print()



# # creating new groups ontop of existing ones in the .h5 file
# # g2 = hf.create_group('group2/subfolder')
# # i would be adding new groups at this level
# # atlas_h5['t00000']['s00']['0']
# # <HDF5 group "/t00000/s00/0" (1 members)>
# # where that 1 member is 'cells'
# # which is of type h5py._hl.dataset.Dataset




 



