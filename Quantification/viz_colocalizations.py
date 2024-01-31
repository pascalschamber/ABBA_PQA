import sys
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as patches
import pandas as pd
import time
import geojson
import timeit
from tifffile import imread
from skimage.measure import regionprops_table
import numba as nb
from numba import jit, types
from numba.typed import List
import shutil
import pyclesperanto_prototype as cle
from timeit import default_timer as dt
import ast
import matplotlib.patches as patches

import concurrent
import concurrent.futures
import multiprocessing
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
import utilities.utils_image_processing as uip
import utilities.utils_plotting as up
from utilities.utils_general import verify_outputdir
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import core_regionPoly as rp

def show_channels(img, chs=[0,1,2], clip_max=None, draw_box=None, show_lbl_is_only=None, extent=None, axs=None, interpolation = None):
    assert img.ndim == 3

    # normalize, or apply label color map
    if img.dtype == np.int32: # labels
        cmaps = [up.get_label_colormap() for _ in range(img.shape[-1])]
        interpolation = 'nearest'
    elif img.dtype == np.uint16: # images
        if clip_max is None: clip_max = img.max() # equates to min/max normalization
        img = uip.convert_16bit_image(img, NORM=True, CLIP=(0,clip_max))

    if img.dtype == np.uint8:
        rgb = ['red', 'green', 'blue']
        cmaps = [up.generate_custom_colormap(rgb[c], img) for c in chs]

    
    for i in range(img.shape[-1]):
        if axs is None:
            fig, ax = plt.subplots()
        else:
            ax=axs[i]
        if show_lbl_is_only is not None:
            img[...,i] = np.where(img[...,i]==show_lbl_is_only[i], 1, 0)
        
        ax.imshow(img[...,i], cmap=cmaps[i], extent=extent, interpolation=interpolation)

        if draw_box is not None:
            rect = patches.Rectangle(
                (draw_box[0], draw_box[1]), draw_box[3]-draw_box[1], draw_box[2]-draw_box[0],
                linewidth=1, edgecolor='w', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

def plot_colocalization(rpdf_view, centroid_i=63222, nuc_view_pad=500):
    rpdf_view = rpdf_view.loc[centroid_i, :].to_dict()
    possible_cols = ['ch0_intersecting_label', 'label', 'ch2_intersecting_label']
    show_lbl_is = [int(rpdf_view[col]) for col in possible_cols if ((col in rpdf_view) and (not pd.isnull(rpdf_view[col])))]

    img_bbox = ast.literal_eval(rpdf_view['bbox'])
    view_bbox = np.array(img_bbox) + np.array([-nuc_view_pad, -nuc_view_pad, nuc_view_pad, nuc_view_pad])
    nuc_bbox = np.array(img_bbox) - np.array([view_bbox[0], view_bbox[1], view_bbox[0], view_bbox[1]])
    x1, y1, x2, y2 = view_bbox

    fig, axs = plt.subplots(3,2, figsize=(10,15))
    axs = axs.flatten()
    show_channels(fs_img[x1:x2, y1:y2,  :], axs=[axs[0],axs[2],axs[4]], extent=None, draw_box=nuc_bbox)
    show_channels(nuc_img[x1:x2, y1:y2,  :], axs=[axs[1],axs[3],axs[5]], extent=None, draw_box=nuc_bbox)#, show_lbl_is_only=show_lbl_is)
    plt.show()



ac = AnimalsContainer()
ac.init_animals()
animals = [ac.get_animals('TEL15')]
an = animals[0]
datum = an.d[0]
rpdf = pd.read_csv(datum.rpdf_paths)
region_df = pd.read_csv(datum.region_df_paths)


read_img_kwargs = {'flip_gr_ch':lambda an_id: True if (an_id > 29 and an_id < 50) else False} 
d_read_img_kwargs = {k:v if not callable(v) else v(an.animal_id_to_int(an.animal_id)) for k,v in read_img_kwargs.items()} if read_img_kwargs else {}
fs_img = uip.read_img(datum.fullsize_paths, **d_read_img_kwargs)
nuc_img = imread(datum.quant_dir_paths)

## show fullsize images
if bool(0):
    plt.imshow(np.stack([
        uip.convert_16bit_image(fs_img[...,i], NORM=True, CLIP=(0,fs_img[...,i].max())) for i in range(fs_img.shape[-1])], -1)
        );plt.show()
    plt.imshow(nuc_img[...,0], cmap=up.lbl_cmap(), interpolation='nearest');plt.show()

clc_id=3
rpdf_clc = rpdf[rpdf['colocal_id']==clc_id]
valid_centroid_is = rpdf_clc['centroid_i'].values
print(f"num valid_centroid_is: {len(valid_centroid_is)}")
for ci in range(15): 
    plot_colocalization(rpdf_clc, centroid_i=valid_centroid_is[ci], nuc_view_pad=100)

