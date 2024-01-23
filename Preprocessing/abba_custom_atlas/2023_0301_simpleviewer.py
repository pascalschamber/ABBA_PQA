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

def cell_to_array(cell_obj, rot=True):
    ''' 
    convert the h5 cell object to a numpy array on the fly to preserve memory 
    rotate the array so coronal sections are presented 
    '''

    return np.rot90(np.array(cell_obj), k=1, axes=(2,0)) if rot else np.array(cell_obj)

def load_atlas_cells(resolution='0'):
    ''' 
    extract the precursor np arrays (cell object) from the h5 file 
        resolution = '0' is highest '3' is lowest
    '''
    img_dir_atlas = r'C:\Users\pasca\Downloads\fiji-win64_1\abba_atlases'
    img_name_atlas = 'ccf2017-mod65000-border-centered-mm-bc.h5'
    img_path_atlas = os.path.join(img_dir_atlas, img_name_atlas)

    atlas_h5 = h5py.File(img_path_atlas)
    channels = ['s00', 's01', 's02', 's03']
    cells = [atlas_h5['t00000'][ch][resolution]['cells'] for ch in channels]
    return cells

def save_atlas_as_npy():
    ''' save extracted cell objects as a 4d np array '''
    out_path = os.path.join(r'C:\Users\pasca\Desktop\ABA2017', 'ABA2017_maxRes.npy')
    cells = load_atlas_cells()
    # convert to single np array
    arrays = np.stack([cell_to_array(cells[i]) for i in range(len(cells))], axis=-1)
    print(arrays.shape)
    np.save(out_path, arrays)




def load_json_data():
    img_dir_labels = r'C:\Users\pasca\Downloads\fiji-win64_1\abba_atlases'
    img_name_labels = '1.json'
    img_path_labels = os.path.join(img_dir_labels, img_name_labels)
    with open(img_path_labels, 'r') as f: 
        img_labels_json = json.load(f)
    basic_groups = img_labels_json['msg'][0]['children'][0]['children']
    return basic_groups

def iter_children(obj, result=None):
    if result is None: 
        result = {}
    if isinstance(obj, list):
        if obj == []:
            pass
        for el in obj:
            iter_children(el, result=result)
    elif isinstance(obj, dict):
        result[obj['atlas_id']] = obj['name']
        iter_children(obj['children'], result=result)
    return result

def iterate_dicts(dicts, result=None, counter=0):
    if result==None:
        result = {}

    for d in dicts:
        # print(d)
        
        children = d.get('children')
        result[d['id']] = d['name']

        if children:
            counter+=1
            iterate_dicts(children, result=result, counter=counter)
    return result


'''
####################################################################################
# Working with region labels
    
    Notes
    ##########################################################################
    Regions of interest that already exist
        131 - Lateral amygdalar nucleus
        303 - Basolateral amygdalar nucleus, anterior part
        311 - Basolateral amygdalar nucleus, posterior part

    I want to subdivide 303 into 3 parts
        304 - BLA.am
        305 - BLA.al
        306 - BLA.ac - am becomes ac at 729

        
    Align ABBA atlas to ARA (ABBA res='0' z-ind == ARA #)
    ##########################################################################
    642 == 66
    706 == 72
    761 == 78


    Align ABBA to Hintiryan
    #########################################################################
    all 303 before z642 == BLA.am
    from 642->729 303 is divided into 
    all 303 after z729 == BLA.ac

    z771 BLA.ac dissapears, BLAp takes over 


    view major groups under "basic celltypes and regions"
    #########################################################################
    import pprint
    basic_groups = load_json_data()
    ids = iterate_dicts(basic_groups)
    pprint.pprint(basic_groups)



####################################################################################
# Working with h5 files

    Notes
    ##########################################################################
    0 - NISSL
    1 - Labels_border
    2 - ARA
    3 - labels_MOD_65000
    ##########################################################################

    explore contents
    ``````````````````````````````````````````````````````````````````````````````````````````````````
    print(atlas_h5.keys()) # <KeysViewHDF5 ['__DATA_TYPES__', 's00', 's01', 's02', 's03', 't00000']>

    for el in ['s00', 's01', 's02', 's03']:
    print(el)
    for ind in ['0', '1', '2', '3']:
        print(atlas_h5['t00000'][el][ind]['cells'])
    print()

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output: 
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        s00
        <HDF5 dataset "cells": shape (1140, 800, 1320), type "<i2">
        <HDF5 dataset "cells": shape (570, 400, 660), type "<i2">
        <HDF5 dataset "cells": shape (285, 200, 330), type "<i2">
        <HDF5 dataset "cells": shape (142, 100, 165), type "<i2">

        s01
        <HDF5 dataset "cells": shape (1140, 800, 1320), type "<i2">
        <HDF5 dataset "cells": shape (570, 400, 660), type "<i2">
        <HDF5 dataset "cells": shape (285, 200, 330), type "<i2">
        <HDF5 dataset "cells": shape (142, 100, 165), type "<i2">

        s02
        <HDF5 dataset "cells": shape (1140, 800, 1320), type "<i2">
        <HDF5 dataset "cells": shape (570, 400, 660), type "<i2">
        <HDF5 dataset "cells": shape (285, 200, 330), type "<i2">
        <HDF5 dataset "cells": shape (142, 100, 165), type "<i2">

        s03
        <HDF5 dataset "cells": shape (1140, 800, 1320), type "<i2">
        <HDF5 dataset "cells": shape (570, 400, 660), type "<i2">
        <HDF5 dataset "cells": shape (285, 200, 330), type "<i2">
        <HDF5 dataset "cells": shape (142, 100, 165), type "<i2">
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        
    rotating the h5 array
    ``````````````````````````````````````````````````````````````````````````````````````````````````
    I confirmed that using np.rot90 imposes no alteration to morphology to horizontal oritentation initially provided
        i used rot90 so orientation is coronal
'''






def init_viewer(resolution='0', load_npy=False):
    # load array
    if load_npy:
        arrays = np.load(os.path.join(r'C:\Users\pasca\Desktop\ABA2017', 'ABA2017_maxRes.npy'))
        cells = None
        print(arrays.shape)
    else:
        cells = load_atlas_cells(resolution=resolution)
        arrays = None
        print(cells)
    viewer = napari.Viewer() # create napari viewer
    iter_ind = len(cells) if cells else arrays.shape[-1]
    iter_obj = cells if cells else arrays
    img_ind = None
    iter_slice = img_ind if cells else (None, None, None, img_i)
    # slicing is not made to work with load np array, need to change to [..., img_i] instead of [img_i]
    for img_i in range(iter_ind):
        # print(iter_obj[..., img_i].shape, arrays[..., img_i].dtype)
        # add arrays to viewer
        print(img_i)
        if img_i == 3:
            viewer.add_image(cell_to_array(iter_obj[img_i]), name=f'aba{img_i}', rgb=False)
        else:
            viewer.add_image(cell_to_array(iter_obj[img_i]), name=f'aba{img_i}', rgb=False)
    return viewer

viewer = init_viewer(resolution='0')

basic_groups = load_json_data()
ids = iterate_dicts(basic_groups)





