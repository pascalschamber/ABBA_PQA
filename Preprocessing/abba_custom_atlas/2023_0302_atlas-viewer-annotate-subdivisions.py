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
        print(img_i)
        viewer.add_image(cell_to_array(iter_obj[img_i]), name=f'aba{img_i}', rgb=False) # add arrays to viewer

    return viewer

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
    return 0





'''
####################################################################################
 JSON LABEL FUNCTIONS

####################################################################################
'''


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
    ''' chatgpt's implementation with edits '''
    if result==None:
        result = {}

    for d in dicts:     
        children = d.get('children')
        result[d['id']] = d['name']

        if children:
            counter+=1
            iterate_dicts(children, result=result, counter=counter)
    return result












viewer = init_viewer(resolution='0')

basic_groups = load_json_data()
ids = iterate_dicts(basic_groups)

# add annotated labels
inprogress_anno = imread(r'C:\Users\pasca\Downloads\fiji-win64_1\myAtlas\ccf2017_labels_bla-subdivisions_20230302.tif')
viewer.add_image(inprogress_anno, name='final_atlas_labels')

# add outline image
outline_image = imread(r'C:\Users\pasca\Downloads\fiji-win64_1\myAtlas\ccf2017_outlines_bla_subdivisions_only.tif')
viewer.add_image(outline_image, name='outline')











'''
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
    
    Issue: ABBA/QuickNii atlas doesn't match ARA
        no rotation in Quicknii provides a close alignment to ARA
        so this will be an estimation
        * ARA appears to have some rotation of sectioning index, hpc landmarks doesn't match bottom landmarks

    view major groups under "basic celltypes and regions"
    #########################################################################
    import pprint
    basic_groups = load_json_data()
    ids = iterate_dicts(basic_groups)
    pprint.pprint(basic_groups)
        
    
    Align ABBA atlas to ARA (ABBA res='0' z-ind == ARA #)
    ##########################################################################
    642 == 66 --> BLA.al appear
    685 == 72 --> BLAp appears, BLA.al dissapears
    706 == 74 --> BLA.am becomes BLA.ac
    771 == 78 --> BLA.ac dissapears, BLAp takes over
    
    # create interpolation between ARA and ABBA
        interp = np.array([[66,72,74,78],[642, 685, 706, 771]])
        xs = np.arange(66, 79)
        xp, fp = interp[0], interp[1]
        interpolation = np.interp(xs, xp, fp)
        dict(zip(xs, [int(xx) for xx in interpolation])) # print

        66: 642, --> BLA.al appears
        67: 649,
        68: 656,
        69: 663,
        70: 670,
        71: 677, 
        72: 685, --> BLAp appears, BLA.al dissapears
        73: 695,
        74: 706, --> BLA.am becomes BLA.ac
        75: 722,
        76: 738,
        77: 754,
        78: 771}


    Align ABBA to Hintiryan
    #########################################################################
    all 303 before z642 == BLA.am
    from 642->729 303 is divided into 
    all 303 after z729 == BLA.ac
    z685 BLAp appears
    z771 BLA.ac dissapears, BLAp takes over 

    notes on filling
    #########################################################################
    I am filling all of BLA.am with label #1, should have just filled BLA.al
    label#2 == BLA.al


    notes on interpolation of labels between slices
    #########################################################################
    region slicing
        before 66
            # labels slice 
            am_slice = aba3[:642]
            am_label = np.where(am_slice==303, 1, 0)
        all ac from 706 -> end
            ac_slice = aba3[706:]
            ac_label = np.where(ac_slice==303, 3, 0)

        interpolate labels from 66->72
            interp_slice = aba3[642:685]
            partial_volume = viewer.layers['aba1 copy'].data[642:685]
            # add in label 1 and 2
            interp_slice = np.where(partial_volume==1, 1, interp_slice)
            interp_slice = np.where(partial_volume==2, 2, interp_slice)
            # find incomplete_indicies
            incomplete_indices = np.argwhere(interp_slice == 303)
            # Define interpolation function
            interp_func = RegularGridInterpolator(
                points=[interp_slice.shape[0], interp_slice.shape[1], interp_slice.shape[2]], 
                values=interp_slice)
            # Interpolate incomplete label regions using linear interpolation
            interpolated_values = interp_func(incomplete_indices)
            # Replace incomplete label regions with interpolated values
            volume = interp_slice.copy()
            volume[incomplete_indices[:, 0], incomplete_indices[:, 1], incomplete_indices[:, 2]] = interpolated_values
        
        Join the three sections
            segmented_BLA = np.concatenate(am_label, volume, ac_label)
    
    Joining regions from inprogress bla subregion segmentations
    #########################################################################
        region slicing
            wierd 2 label at 642 578 878 on horizontal

        # before 66
        # labels slice 
        aba3 = viewer.layers['aba3'].data
        am_slice = aba3[:642]
        am_label = np.where(am_slice==303, 1, 0)
        aba1_am = np.concatenate((am_label, viewer.layers['aba1 copy'].data[642:]), axis=0)

        # all ac from 706 -> end
        ac_slice = aba3[706:]
        ac_label = np.where(ac_slice==303, 3, 0)
        aba1_ac = np.concatenate((viewer.layers['aba1_am'].data[:706], ac_label), axis=0)
        viewer.add_image(aba1_ac)

        # inbetween 
        # interpolate labels from 66->72
        interp_slice = np.where(aba3[642:685]==303, 1, 0)
        current_interp_og = viewer.layers['aba1_ac'].data[642:685]
        current_interp = np.where((np.where(aba3[642:685]==303, 1, 0))==1, 1, current_interp_og)
        current_interp = np.where(current_interp_og == 2, 2, current_interp)
        current_interp = np.concatenate(
            (viewer.layers['aba1_ac'].data[:642], current_interp,viewer.layers['aba1_ac'].data[685:]), axis=0)
        # clear outlines
        current_interp = np.where(current_interp==255, 0, current_interp)
        viewer.add_image(current_interp)

        # np.save(os.path.join(r'C:\Users\pasca\Downloads\fiji-win64_1\myAtlas', 'bla_subdivisions_pre-final.npy'), current_interp)

        # set all to label 1
        # fill in existing 
        interp_slice = np.where( != 1, 2, 
        # zero left hemisphere BLA
        interp_slice[:, :, :interp_slice.shape[2]//2] = 0
        partial_volume = viewer.layers['aba1 copy'].data[642:685]
        # add in label 1 and 2
        interp_slice = np.where(partial_volume==1, 100, interp_slice)
        interp_slice = np.where(partial_volume==2, 200, interp_slice)
        viewer.add_image(interp_slice)

        # Join the three sections
        segmented_BLA = np.concatenate((am_label, volume, ac_label), axis=0)

        
    Final processing of added annotations
    ##########################################################################
        # remove any pixels outside of the bla that were accidentaly added during anno
        cleaned = np.where(viewer.layers['aba3'].data == 303, viewer.layers['inprogress_anno'].data, 0)
        viewer.add_image(cleaned)

        # assign a temporary id to the added subregions that doesn't conflict with existing labels
        # ideally these would be children of 303, thus 304,305,306
        current_max = viewer.layers['aba3'].data.max()
        theoretical_max = 2**16

        for current_label, new_label in zip([1,2,3], [30000, 30001, 30002]):
            cleaned = np.where(cleaned == current_label, new_label, cleaned)
        viewer.add_image(cleaned, name='cleaned2')
    
        
    Creating the final atlas labels with bla subdivisions added
    ##########################################################################
        # forgot to add in the section of BLA.ac after BLA.al ends, adding it here
        viewer.layers['cleaned2'].data[685:706] = np.where(viewer.layers['aba3'].data[685:706] == 303, 30000, viewer.layers['cleaned2'].data[685:706])

        # mirror the annotation created on the right side so left has a complement
        right_half = viewer.layers['cleaned2'].data[:, :, viewer.layers['cleaned2'].data.shape[2]//2:]
        left_half = np.stack([np.fliplr(right_half[i]) for i in range(right_half.shape[0])], axis=0)
        concat = np.concatenate((left_half, right_half), axis = 2)
        viewer.add_image(concat, name='mirrored')

        # add the final annotation to the existing label atlas
        final_atlas_labels = np.where(viewer.layers['mirrored'].data > 0, viewer.layers['mirrored'].data,viewer.layers['aba3'].data)
        viewer.add_image(final_atlas_labels)
    
        
####################################################################################
# get outlines of new annotations

    # iter through vertical and horizontal dims looking for areas where px values change
    ##########################################################################

        def get_indicies_where_px_val_changes(rav):
            # VERY SLOW, consider ravel on 3D and use numba
            # rav is a raveled array, # C = rows

            delta = []
            for i, el in enumerate(rav):
                if i != 0:
                    if el != previous:
                        delta.append(1)
                    else:
                        delta.append(0)
                else:
                    delta.append(0)
                previous = el
            return np.array(delta)

        def get_outline_2d(mir1):
            # get the outline of a 2d array 
            mir1shape = mir1.shape
            indR = get_indicies_where_px_val_changes(np.ravel(mir1, order='C'))
            indC = get_indicies_where_px_val_changes(np.ravel(np.rot90(mir1,k=1), order='C')) # rotate horizontally
            indR = np.reshape(indR, mir1shape)
            indC = np.rot90(np.reshape(indC, (mir1shape[1], mir1shape[0])), k=3)
            combined = indR + indC
            combined = np.where(combined > 0, 1, 0) # constrain to binary array
            return combined

        outline_array = np.stack(
            [get_outline_2d(viewer.layers['mirrored'].data[i]) for i in range(viewer.layers['mirrored'].data.shape[0])],
            axis=0)

        print(outline_array.shape)

        viewer.add_image(outline_array)
        # may need to move the array up 1 px before merging, should try this
        # also need to try removing corners it seems 
        # or try to get the mesh of the annotations

        # merge with exisisting outline
        merged_outline = np.where(outline_array>0, 255, viewer.layers['aba1'].data)
        viewer.add_image(merged_outline)


'''
