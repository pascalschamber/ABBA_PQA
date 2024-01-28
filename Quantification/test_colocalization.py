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

import concurrent
import concurrent.futures
import multiprocessing
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_image_processing import read_img, print_array_info, convert_16bit_image
from utilities.utils_general import verify_outputdir
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import utilities.utils_plotting as up
import core_regionPoly as rp



import time

def colocalize(colocalization_params, rpdf, quant_img, prt_str=''):
    for clc_props in colocalization_params:
        coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
        rpdf, prt_str = get_colocalization(self, rpdf, quant_img, intersection_threshold=0.00, 
                                            coIds=coIds, coChs=coChs, assign_colocal_id=assign_colocal_id, prt_str=prt_str)
    return rpdf, prt_str


def get_colocalization(disp, rpdf, quant_img, intersection_threshold=0.00, coIds=(1,2), coChs=(0,1), assign_colocal_id=3, prt_str=''):
    ''' #TODO bring info about what to colocalize out into dispatcher
    colocalization method, very fast (<0.2 s)
    grab bboxes for ch0 and ch1 
    for each ch1 nuclei get signal in other channels at that location 
    also gets the label of ch0 nuclei colocal with ch1
    ARGS
        - rpdf (pd.Dataframe) --> region props as a dataframe (e.g. by using regionprops_table)
        - quant_img (np.array[int]) --> label image containing detected nuclei
        - intersection_threshold (float) --> intersection is valid between nuclei if greater than this percentage (range 0,1)
        - coIds (tup[int,int]) --> colocal ids in rpdf to perform colocalization, colocal nuclei are constructed from second value
        - coChs (tup[int,int]) --> channels in quant_img corresponding to coIDs
        - assign_colocal_id (int) --> colocal id to assign resulting colocalization in rpdf
    '''

    # for testing intersection images
    # self, intersection_threshold = disps[0], 0.0
    # rpdf, quant_img = get_region_props(self)
    # plot_count = 0
    st_time, len_init_prt_str = dt(), len(prt_str)
    # extract bboxes for zif and gfp channels
    ch0_bboxes, ch0_df_indicies, ch1_bboxes, ch1_df_indicies = (rpdf[rpdf['colocal_id'] == coIds[0]]['bbox'].to_list(), rpdf[rpdf['colocal_id'] == coIds[0]].index, 
                                                                rpdf[rpdf['colocal_id'] == coIds[1]]['bbox'].to_list(), rpdf[rpdf['colocal_id'] == coIds[1]].index)
    print('starting colocalization')
    # colocalization
    colocal_instances_df = []
    ch_intersecting_label_col = f'ch{coChs[0]}_intersecting_label'
    # Initialize time accumulators for each section
    time_extract_bbox = 0
    time_intersection = 0
    time_check_intersection = 0
    time_find_largest = 0
    time_create_df_row = 0

    for cl1i, ch1_bbox in zip(ch1_df_indicies, ch1_bboxes):

        ###########################################################
        start_time = time.time()
        # extract bbox coords from nuclei image
        minx, miny, maxx, maxy = ch1_bbox
        ch0_nucleus_img, ch1_nucleus_img = (quant_img[...,i][minx:maxx, miny:maxy] for i in coChs)
        if ch0_nucleus_img.max() <= 0: # if no signal in ch0 skip
            continue 

        end_time = time.time()
        time_extract_bbox += end_time - start_time

        ###########################################################
        # get intersection percentage if any signal in ch0
        start_time = time.time()
        intersecting_coords = ch0_nucleus_img[ch1_nucleus_img.nonzero()]
        intersection_percent = np.sum(intersecting_coords>0)/intersecting_coords.shape[0]

        end_time = time.time()
        time_intersection += end_time - start_time


        ###########################################################
        start_time = time.time()
        # check intersection percentage is above threshold
        if intersection_percent <= intersection_threshold: continue

        end_time = time.time()
        time_check_intersection += end_time - start_time

        ###########################################################
        start_time = time.time()
        # if some intersecting label get the largest, i.e. get largest non zero label in zif channel
        inter_labels, inter_counts = np.unique(ch0_nucleus_img[ch1_nucleus_img.nonzero()], return_counts=True)
        nonzero_labels = np.nonzero(inter_labels)[0]
        nonzero_counts = inter_counts[np.nonzero(inter_labels)]
        largest_count_index = np.argmax(nonzero_counts)
        largest_intersecting_label_ch0 = inter_labels[nonzero_labels[largest_count_index]]

        end_time = time.time()
        time_find_largest += end_time - start_time

        ###########################################################
        start_time = time.time()
        # create a df row for this colocalization instance
        colocal4_row = dict(rpdf.loc[cl1i, :])
        colocal4_row['colocal_id'] = assign_colocal_id
        colocal4_row['intersection_p'] = intersection_percent
        colocal4_row[ch_intersecting_label_col] = int(largest_intersecting_label_ch0)
        colocal_instances_df.append(colocal4_row)

        end_time = time.time()
        time_create_df_row += end_time - start_time
        

    # After the loop, print the accumulated times
    print("Time taken for extracting bbox coordinates:", time_extract_bbox)
    print("Time taken for calculating intersection:", time_intersection)
    print("Time taken for checking intersection threshold:", time_check_intersection)
    print("Time taken for finding largest intersecting label:", time_find_largest)
    print("Time taken for creating DF row:", time_create_df_row)

    # convert rows to df
    colocal_instances_df = pd.DataFrame(colocal_instances_df)
    print('ended colocalization')
    # add new columns to rpdf, if they don't exist already e.g. if multiple colocalizations are being done
    rpdf_colocal = rpdf.copy(deep=True)
    if 'intersection_p' not in rpdf_colocal.columns:
        rpdf_colocal['intersection_p'] = np.nan
    if ch_intersecting_label_col not in rpdf_colocal.columns:
        rpdf_colocal[ch_intersecting_label_col] = np.nan
    rpdf_final = pd.concat([rpdf_colocal, colocal_instances_df], ignore_index=True)
    
    prt_str+=(f"{rpdf_final.value_counts('colocal_id')}\n")
    prt_str+=(f'colocalization complete took {timeit.default_timer() - st_time}\n')

    if len_init_prt_str==0: # i.e. was called here and not from a func that didn't expect this output
        print(prt_str, flush=True)
        return rpdf_final
    print('returning colocalization')
    return rpdf_final, prt_str


pp_st = time.time()
ac = AnimalsContainer()
ac.init_animals()

###################################################################################################
# PARAMS
# COLOCALID_CH_MAP = {0:1, 1:2, 2:0} # dict mapping channels in intensity image to colocal id (e.g dapi (ch2) is colocal_id 0, zif (ch0) is 1, and GFP (ch1) is 2)
TEST = bool(0) # process centroid subset
WRITE_OUTPUT = bool(1) # write rpdf and region df to disk
CLEAN = bool(0) # delete previous counts
MULTIPROCESS = bool(0) # use Processpool for parallel processing, else run serially
BACKGROUND_SUBRACTION = bool(1) # use tophat algorithm to correct for uneven illumination
MAX_WORKERS = 12 # num of cores, if running multithreded, for me 12
READ_IMG_KWARGS = {'flip_gr_ch':lambda an_id: True if (an_id > 29 and an_id < 50) else False} 

animals = ac.get_animals(['cohort2', 'cohort3', 'cohort4'])[:1]
start_i_disps = 0 # start from this dispatcher, useful if run was interupted 
end_i_disps = 1
COLOCALID_CH_MAP = ac.ImgDB.get_colocalid_ch_map()
COLOCALIZATION_PARAMS = ac.ImgDB.colocalizations



###################################################################################################
# MAIN
if CLEAN: ac.clean_animal_dir(animals, 'counts') 

from img2df import get_dispatchers, get_region_props
# get dispatchers
disps = get_dispatchers(
    animals, TEST=TEST, BACKGROUND_SUBRACTION=BACKGROUND_SUBRACTION, WRITE_OUTPUT=WRITE_OUTPUT, 
    COLOCALID_CH_MAP=COLOCALID_CH_MAP, read_img_kwargs=READ_IMG_KWARGS, colocalization_params=COLOCALIZATION_PARAMS,
) [start_i_disps:end_i_disps]
print(f'processing num dispatchers {len(disps)}', flush=True)


self = disps[0]
rpdf, quant_img, self.prt_str = get_region_props(self, ch_colocal_id=self.ch_colocal_id, prt_str=self.prt_str)
print('get_region_props finished')

fs_img = read_img(self.datum.fullsize_paths, **self.read_img_kwargs)
# rpdf_coloc, self.prt_str = colocalize(self.colocalization_params, rpdf, quant_img, prt_str=self.prt_str)
# # rpdf_coloc, self.prt_str = get_colocalization(self, rpdf, quant_img, prt_str=self.prt_str)
# print(self.prt_str)


###################################################################################################
# get_colocalization(disp, rpdf, quant_img, intersection_threshold=0.00, coIds=(1,2), coChs=(0,1), assign_colocal_id=3, prt_str='')
import utilities.utils_image_processing as uip
import utilities.utils_plotting as up
from skimage import measure

def overlay_nuc_outlines(img_list, def_title=None, colors=['blue', 'red', 'green'], ax=None):
    SHOW = True if ax is None else False
    if ax is None:
        fig,ax = plt.subplots()
    if def_title is not None: ax.set_title(def_title)
    for i in range(len(img_list)):
        outlines = measure.find_contours(uip.to_binary(img_list[i]))
        for contour in outlines:
            ax.plot(contour[:, 1], contour[:, 0], color=colors[i])
    ax.invert_yaxis()
    if SHOW:
        plt.show()
    return ax
    
def plot_nuc_crops(img_list, def_title=None):
    # add list of binary images together, highlighting overlapping masks 
    plt_img = np.sum([uip.to_binary(img_list[i])*(i+1) for i in range(len(img_list))], axis=0)
    fig,axs= plt.subplots(2,1, figsize=(8,12))
    axs[0].imshow(plt_img, interpolation='nearest', cmap=up.lbl_cmap())
    if def_title is not None: axs[0].set_title(def_title)
    overlay_nuc_outlines(img_list, ax=axs[1])
    
    plt.show()



def calculate_iou(masks):
    """
    Calculate the Intersection over Union (IoU) for multiple masks.

    :param masks: A 3D NumPy array of shape (height, width, n_masks) containing binary masks.
    :return: IoU value.
    """
    intersection = np.logical_and.reduce(masks, axis=2).sum()
    union = np.logical_or.reduce(masks, axis=2).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_pairwise_overlap(masks):
    """
    Calculate the pairwise overlap between each pair of masks.

    :param masks: A 3D NumPy array of shape (height, width, n_masks) containing binary masks.
    :return: A 2D array where each element [i, j] is the overlap between mask i and mask j.
    """
    n_masks = masks.shape[2]
    overlap_matrix = np.zeros((n_masks, n_masks))

    for i in range(n_masks):
        for j in range(i, n_masks):
            intersection = np.logical_and(masks[:, :, i], masks[:, :, j]).sum()
            union = np.logical_or(masks[:, :, i], masks[:, :, j]).sum()
            overlap = intersection / union if union != 0 else 0
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap  # Symmetric matrix

    return overlap_matrix

def calculate_total_overlap(masks):
    """
    Calculate the total overlap of all masks.

    :param masks: A 3D NumPy array of shape (height, width, n_masks) containing binary masks.
    :return: Proportion of total overlap area to the image area.
    """
    total_overlap = np.logical_and.reduce(masks, axis=2).sum()
    # total_area = masks.shape[0] * masks.shape[1]
    return total_overlap / np.sum(masks>0)
# ARGS
#######################################
clc_props = self.colocalization_params[0]
# coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
coChs, coIds, assign_colocal_id = [0, 1], [1, 2], 3
# coChs, coIds, assign_colocal_id = [2, 0, 1], [0, 1, 2], 3
intersection_threshold=0.01
prt_str=''

def colocalize(colocalization_params, rpdf, quant_img, fs_img, prt_str='', intersection_threshold=0.01):
    prt_str += f"base_coloc_counts: {rpdf['colocal_id'].value_counts().to_dict()}"
    for clc_props in colocalization_params:
        coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
        rpdf, prt_str = get_colocalization(rpdf, quant_img, fs_img, intersection_threshold=intersection_threshold, 
                                            coIds=coIds, coChs=coChs, assign_colocal_id=assign_colocal_id, prt_str=prt_str)
    return rpdf, prt_str
def get_colocalization(rpdf, quant_img, fs_img, intersection_threshold=0.00, coIds=(1,2), coChs=(0,1), assign_colocal_id=3, prt_str=''):
    # clc_props = self.colocalization_params[0]
    # coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
    # coChs, coIds, assign_colocal_id = [0, 1], [1, 2], 3
    # coChs, coIds, assign_colocal_id = [2, 0, 1], [0, 1, 2], 3
    intersection_threshold=0.01
    prt_str=''


    # func
    #######################################
    st_time, len_init_prt_str = dt(), len(prt_str)
    # extract bboxes for zif and gfp channels, using last provided colocal_id as base for comparison
    n_clc_channels = len(coIds)
    base_coId, base_coCh = coIds[-1], coChs[-1]
    other_coIds, other_coChs = coIds[:-1], coChs[:-1]
    len_other_coChs = len(other_coChs)
    iter_other_coChs = np.arange(len_other_coChs)

    # extract indicies in rpdf for base colocal id and use these to get other input values
    ch_df_indicies = rpdf[rpdf['colocal_id'] == base_coId].index.values
    ch_bboxes = np.array([el for el in rpdf.loc[ch_df_indicies, 'bbox'].values])
    ch_nuc_lbls = rpdf.loc[ch_df_indicies, 'label'].values
    assert ch_bboxes.shape[0] == ch_df_indicies.shape[0] == ch_nuc_lbls.shape[0]
    print('starting colocalization')

    # colocalization
    
    # result cols = cli, assign_colocal_id, intersection_percent, *(largest_label for each other ch)
    result_default_cols = ['cli', 'intersection_percent']
    ch_intersecting_label_cols = [f'ch{coCh}_intersecting_label' for coCh in other_coChs]
    len_result_default_cols = len(result_default_cols) 
    results_arr = np.full((ch_bboxes.shape[0], len_result_default_cols + len(ch_intersecting_label_cols)), fill_value=-1.0)

    SHOW_FAILED = bool(0)
    SHOW_SUCCESS = bool(0)
    for row_i in np.arange(ch_bboxes.shape[0])[:]:
        CONTINUE_FLAG = False
        cli, ch_bbox, base_nuc_lbl = ch_df_indicies[row_i], ch_bboxes[row_i], ch_nuc_lbls[row_i]
        minx,miny,maxx,maxy = ch_bbox

        # extract bbox coords from nuclei image
        quant_crop = quant_img[minx:maxx, miny:maxy,:]
        fs_crop = fs_img[minx:maxx, miny:maxy,:]
        
        ch_nucleus_img_base = quant_img[minx:maxx, miny:maxy, base_coCh] # extract bbox around this label
        base_non_zero = (ch_nucleus_img_base==base_nuc_lbl).nonzero() # get nonzero coords for current label in base ch (remove other labels in this channel if present)

        ch_nucleus_img_others = quant_img[minx:maxx, miny:maxy, other_coChs]
        others_non_zero = ch_nucleus_img_others[base_non_zero[0], base_non_zero[1], :]
        others_masked = np.zeros_like(ch_nucleus_img_others)
        for oCh_i in range(len(other_coChs)):
            o_masked = ch_nucleus_img_others[base_non_zero[0], base_non_zero[1], oCh_i]
            if o_masked.max() == 0: 
                CONTINUE_FLAG = True
                if SHOW_FAILED:
                    fig,axs = plt.subplots(2,3)
                    up.show_channels(fs_crop, axs=axs.flatten()[:3])
                    up.show_channels(quant_crop, axs=axs.flatten()[3:])
                    fig.suptitle('failed b/c no label in one channel')
                    plt.show()
                break
            inter_labels, inter_counts = np.unique(o_masked.ravel()[o_masked.nonzero()[0]], return_counts=True) # flatten, take non-zero values, count unique
            largest_intersecting_label = inter_labels[np.argmax(inter_counts)] # take label w highest count
            others_masked[base_non_zero[0], base_non_zero[1], oCh_i] = np.where(o_masked==largest_intersecting_label, 1, 0)
            results_arr[row_i, len_result_default_cols+oCh_i] = largest_intersecting_label

        if CONTINUE_FLAG: continue # if any of the other channels had no label overlapping with base, skip

        # get intersection percentage
        all_stacked = np.concatenate((others_masked, np.where(ch_nucleus_img_base[:, :, np.newaxis]==base_nuc_lbl, 1, 0)), axis=2)
        iou = calculate_iou(all_stacked)
        if iou < intersection_threshold:
            if SHOW_FAILED:
                fig,axs = plt.subplots(2,3)
                up.show_channels(fs_crop, axs=axs.flatten()[:3])
                up.show_channels(quant_crop, axs=axs.flatten()[3:])
                fig.suptitle('failed b/c < intersection threshold')
                plt.show()
            continue
        results_arr[row_i, 0:len_result_default_cols] = cli, iou
        
        if SHOW_SUCCESS:
            # calculate_pairwise_overlap(all_stacked)
            tot_overlap = calculate_total_overlap(all_stacked)

            fig,axs = plt.subplots(4,3, figsize=(9,12))
            axs=axs.flatten()
            up.show_channels(fs_crop, axs=axs[:3])
            up.show_channels(quant_crop, axs=axs[3:6])
            up.show(ch_nucleus_img_base, def_title='ch_nucleus_img_base', ax=axs[7])
            other_ax = [7+1, 7-1]
            [up.show(ch_nucleus_img_others[...,i], def_title=f'{other_coChs[i]} ch_nucleus_img_others', ax=axs[other_ax[i]]) for i in iter_other_coChs]
            [up.show(others_masked[...,i], def_title=f'{other_coChs[i]} others_masked', ax=axs[[10+1, 10-1][i]]) for i in iter_other_coChs]
            axs[10].imshow(np.logical_and.reduce(all_stacked, axis=2), cmap='gray')
            vals = [round(v,2) for v in [iou, tot_overlap]]
            axs[10].set_title(f"iou:{vals[0]}, ovr:{vals[1]}")
            plt.show()


    # convert successful results to dataframe and merge
    results = results_arr[(results_arr[:, 0]>-1), :]
    print(results.shape)
    df_results = pd.DataFrame(results, columns=result_default_cols + ch_intersecting_label_cols).set_index('cli')
    merge_on_df = rpdf.iloc[df_results.index.values, :]
    override_cols = [c for c in ch_intersecting_label_cols+['intersection_percent'] if c in merge_on_df.columns]
    coloc_df = (pd.merge(
        merge_on_df.drop(columns=override_cols), df_results, how='left', left_index=True, right_index=True)
        .assign(colocal_id = assign_colocal_id, ))
    for col in coloc_df.columns.to_list():
        if col not in rpdf:
            rpdf[col] = np.nan
    rpdf_coloc = pd.concat([rpdf, coloc_df], ignore_index=True)
    prt_str += f"colocal_id_counts: {rpdf_coloc['colocal_id'].value_counts().to_dict()}"
    return rpdf_coloc, prt_str

t0=dt()
self.colocalization_params.append({'coChs': [2, 0, 1],
  'coIds': [0, 1, 2],
  'assign_colocal_id': 5,
  'intersecting_label_column': 'ch0_intersecting_label',
  'intersecting_colocal_id': 2,
  'other_intensity_name': 'Zif_intensity'})
rpdf_colocal, self.prt_str = colocalize(self.colocalization_params, rpdf, quant_img, fs_img, prt_str=self.prt_str, intersection_threshold=0.01)
print(f"finished in {dt()-t0}.")
print(self.prt_str)