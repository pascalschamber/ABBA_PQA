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



# def get_colocalization(disp, rpdf, quant_img, intersection_threshold=0.00, coIds=(1,2), coChs=(0,1), assign_colocal_id=3, prt_str=''):
#     ''' #TODO bring info about what to colocalize out into dispatcher
#     colocalization method, very fast (<0.2 s)
#     grab bboxes for ch0 and ch1 
#     for each ch1 nuclei get signal in other channels at that location 
#     also gets the label of ch0 nuclei colocal with ch1
#     ARGS
#         - rpdf (pd.Dataframe) --> region props as a dataframe (e.g. by using regionprops_table)
#         - quant_img (np.array[int]) --> label image containing detected nuclei
#         - intersection_threshold (float) --> intersection is valid between nuclei if greater than this percentage (range 0,1)
#         - coIds (tup[int,int]) --> colocal ids in rpdf to perform colocalization, colocal nuclei are constructed from second value
#         - coChs (tup[int,int]) --> channels in quant_img corresponding to coIDs
#         - assign_colocal_id (int) --> colocal id to assign resulting colocalization in rpdf
#     '''

#     # for testing intersection images
#     # self, intersection_threshold = disps[0], 0.0
#     # rpdf, quant_img = get_region_props(self)
#     # plot_count = 0
#     st_time, len_init_prt_str = dt(), len(prt_str)
#     # extract bboxes for zif and gfp channels
#     ch0_bboxes, ch0_df_indicies, ch1_bboxes, ch1_df_indicies = (rpdf[rpdf['colocal_id'] == coIds[0]]['bbox'].to_list(), rpdf[rpdf['colocal_id'] == coIds[0]].index, 
#                                                                 rpdf[rpdf['colocal_id'] == coIds[1]]['bbox'].to_list(), rpdf[rpdf['colocal_id'] == coIds[1]].index)
#     print('starting colocalization')
#     # colocalization
#     colocal_instances_df = []
#     ch_intersecting_label_col = f'ch{coChs[0]}_intersecting_label'
#     for cl1i, ch1_bbox in zip(ch1_df_indicies, ch1_bboxes):
        
#         # extract bbox coords from nuclei image
#         minx,miny,maxx,maxy = ch1_bbox
#         ch0_nucleus_img, ch1_nucleus_img = (quant_img[...,i][minx:maxx, miny:maxy] for i in coChs) # ch1_nucleus_img = quant_img[...,coChs[1]][minx:maxx, miny:maxy]
#         if ch0_nucleus_img.max() <= 0: # if no signal in ch0 skip
#             continue 
        
#         # get intersection percentage if any signal in ch0
#         intersecting_coords = ch0_nucleus_img[ch1_nucleus_img.nonzero()]
#         intersection_percent = np.sum(intersecting_coords>0)/intersecting_coords.shape[0]
        
        
#         # check intersection percentage is above threshold
#         if intersection_percent <= intersection_threshold: continue


#         # if some intersecting label get the largest, i.e. get largest non zero label in zif channel
#         inter_labels, inter_counts = np.unique(ch0_nucleus_img[ch1_nucleus_img.nonzero()], return_counts=True)
#         nonzero_labels = np.nonzero(inter_labels)[0]
#         nonzero_counts = inter_counts[np.nonzero(inter_labels)]
#         largest_count_index = np.argmax(nonzero_counts)
#         largest_intersecting_label_ch0 = inter_labels[nonzero_labels[largest_count_index]]


#         # create a df row for this colocalization instance
#         colocal4_row = dict(rpdf.loc[cl1i, :])
#         colocal4_row['colocal_id'] = assign_colocal_id
#         colocal4_row['intersection_p'] = intersection_percent
#         colocal4_row[ch_intersecting_label_col] = int(largest_intersecting_label_ch0)
#         #TODO: add colocal intensity for zif channel here? e.g. rpdf[(rpdf['colocal_id'==coIds[0]) & (rpdf['label']==largest_intersecting_label_ch0)]['intensity_mean'].values[0]
#         colocal_instances_df.append(colocal4_row)
            

#     # convert rows to df
#     colocal_instances_df = pd.DataFrame(colocal_instances_df)
#     print('ended colocalization')
#     # add new columns to rpdf, if they don't exist already e.g. if multiple colocalizations are being done
#     rpdf_colocal = rpdf.copy(deep=True)
#     if 'intersection_p' not in rpdf_colocal.columns:
#         rpdf_colocal['intersection_p'] = np.nan
#     if ch_intersecting_label_col not in rpdf_colocal.columns:
#         rpdf_colocal[ch_intersecting_label_col] = np.nan
#     rpdf_final = pd.concat([rpdf_colocal, colocal_instances_df], ignore_index=True)
    
#     prt_str+=(f"{rpdf_final.value_counts('colocal_id')}\n")
#     prt_str+=(f'colocalization complete took {timeit.default_timer() - st_time}\n')

#     if len_init_prt_str==0: # i.e. was called here and not from a func that didn't expect this output
#         print(prt_str, flush=True)
#         return rpdf_final
#     print('returning colocalization')
#     return rpdf_final, prt_str

# def colocalize(self, colocalization_params, rpdf, quant_img, prt_str=''):
#     for clc_props in colocalization_params:
#         coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
#         rpdf, prt_str = get_colocalization(self, rpdf, quant_img, intersection_threshold=0.00, 
#                                             coIds=coIds, coChs=coChs, assign_colocal_id=assign_colocal_id, prt_str=prt_str)
#     return rpdf, prt_str



"""
######################################################################################################
PROGRAM DESCRIPTION (version: 2024_0121)
~~~~~~~~~~~~~~~~~~~
    Env: stardist
    Input: stardist predictions and geojson file
    Output: two dataframes for each image
        1) rpdf (region props) - containing properties for all nuclei (after area thresholding), where each nuclei belongs to one of 4 colocalization classes
        2) region_counts_df - nuclei counts per region (sides are separated) 
    Process:
        1) stardist prediction image is loaded, and region props are generated for each channel
        2) colocalization is performed, merging the region props into a single df
        3) nuclei counts are derived for lowest structural level only using region polygons extracted from qupath's geojson file
            - this avoids having to store the region ids as a list in a dataframe, which is slow when reading later
            - these counts are propogated to all parent regions during compiling
        4) resulting counts per region df is saved
    Timings NEW (max) (TEL15_s023_2-1)
        - single
            - load images 18s
            - get_region_props 77s
            - colocalization 0.5s
            - get_nuclei_counts_by_region 33s (num region polys: 445, 309731/318433 (8702 unassigned))
                polys took 5.152107999999998
                load took 5.506371900000005
                localize took 21.85091699999998
                    separate_polytypes took 0.04850439999998457
                    nb_process_polygons took 21.484798499999982
                    map_df took 0.3176204000000098
            - TOTAL: 166s
        - ANIMAL (TEL15, 46 imgs) processPool finished in: 407s
    Timings OLD (max)
        - load images 18s
        - get_region_props 75s
        - colocalization 1.4s
        - get_nuclei_counts_by_region 63s
        - TOTAL: ~150s

Notes 01/21/24
~~~~~~~~~~~
    updated region poly processing to include all polygons
    new module called core_regionprocessing that handles all region polygon operations
    changed how regions are assigned to centroids, not a list of all regions but rather lowest structural level found in

Notes 06/02 
~~~~~~~~~~~
    - changing function of this file to just ouput all rpdfs for each animal, region counts are then generated during compiling 
    - purpose is to apply thresholding to nuclei counts during compiling, so this just generates raw data.
        - thresholding by area will be moved to compiling
    - main ouput is now just region prop dfs (rpdf) for each image
    - also will output a df that contains a summary of the atlas regions for each image that includes the area

NOTES 05/13
~~~~~~~~~~~
    Replaces "load qupath Exports" and "image_to_df (colocalization)" files
    This is a new method for extracting the regions a nuclei belongs to and performing colocalization that is much faster and doesn't rely on generating atlas masks.
    This implements some changes too
        - colocalization is performed between zif and gfp only (no dapi)
        - region counts are generated for all structural levels present
        - region localization is performed from geojson polygons that were exported in qupath
    
    TESTing notes
        - for TEL30 s016 should be about 15 colocal in id30000 L/R sides

######################################################################################################
"""

def write_csv(fp, df):
    with open(fp, 'wb') as f:
        df.to_csv(f)

def load_nuc_intensity_imgs(disp, prt_str=''):
    # get images 
    st_time, len_init_prt_str = dt(), len(prt_str)
    prt_str += (f'fullsize, quant filenames: {Path(disp.datum.fullsize_paths).stem}, {Path(disp.datum.quant_dir_paths).stem}\n')
    quant_img = imread(disp.datum.quant_dir_paths)
    fullsize_img = read_img(disp.datum.fullsize_paths, **disp.read_img_kwargs)
    prt_str += (f'images loaded in {dt() - st_time}.\n')
    if len_init_prt_str==0: # i.e. was called here and not from a func that didn't expect this output
        print(prt_str, flush=True)
        return quant_img, fullsize_img
    return quant_img, fullsize_img, prt_str

def bg_subtract_tophat(img, tophat_radius=8):
    assert img.ndim==3
    bg_sub = np.zeros_like(img)
    for i in range(3):
        bg_sub[...,i] = cle.top_hat_box(img[...,i], radius_x=tophat_radius, radius_y=tophat_radius)
    return bg_sub

def get_region_props(disp, ch_colocal_id={0:1, 1:2, 2:0}, prt_str=''):
    ''' load imgs then create the region props table and make modifications to simply the columns and add the colocal id '''
    t0 = dt()
    quant_img, fullsize_img, prt_str = load_nuc_intensity_imgs(disp, prt_str=prt_str)
    fullsize_img = bg_subtract_tophat(fullsize_img, tophat_radius=8) if disp.BACKGROUND_SUBRACTION else fullsize_img
    rpdf, prt_str = get_rp_table(disp, quant_img, fullsize_img, ch_colocal_id=ch_colocal_id, prt_str=prt_str)
    prt_str += f"get_region_props took: {dt()-t0}"
    return rpdf, quant_img, prt_str


def get_rp_table(disp, quant_img, fullsize_img, channels=3, ch_colocal_id={0:1, 1:2, 2:0}, prt_str=''):
    ''' 
    use label image and intensity image to extract a dataframe containing region props for each channel specified (imgs must have ch last)
        most time is spent in regionprops_table() (~80s, 4s is added through formatting bbox/centroid columns
        
        ARGS
        - quant_img (np.array[int]) --> label image containing detected nuclei
        - fullsize_img (np.array) --> intensity image
        - channels (int) --> number of channels in image
        - ch_colocal_id (dict) --> dictionary assigning colocal id to img channels by map the image channels to their respective colocal ids, 
            e.g. 0=dapi, 1=zif, 2=gfp, 3=zif+gfp
    '''
    rp_st_time, len_init_prt_str = dt(), len(prt_str)
    # define region props to extract
    rps_to_get = [
        'label', 'area', 'bbox', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max',  'axis_major_length', 'axis_minor_length'
    ]
     
    # create the region props table and make modifications to simply the bbox/centroid columns and add the colocal id
    rpdfs = []
    for ch_i in range(channels):
        rpdfs.append(pd.DataFrame(regionprops_table(quant_img[..., ch_i], fullsize_img[..., ch_i], properties=rps_to_get))
                .assign(cohort=disp.cohort,
                        img_name=disp.img_name,
                        colocal_id=ch_colocal_id[ch_i],
                        bbox=lambda x: x[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].apply(lambda row: row.tolist(), axis=1),
                        centroid=lambda x: x[['centroid-0', 'centroid-1']].apply(lambda row: row.tolist(), axis=1))
                .drop(columns=['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'centroid-0', 'centroid-1'])
        )
    # region_props = [pd.DataFrame(regionprops_table(quant_img[..., i], fullsize_img[..., i], properties=rps_to_get)).assign(colocal_id=i, in_place=True) for i in range(3)]
    rpdf = pd.concat(rpdfs, ignore_index=True)
    
    prt_str+=(f"initial rpdf value counts {rpdf.value_counts(['colocal_id'])}\n")
    prt_str+=(f"rps extracted in {timeit.default_timer() - rp_st_time}.\n")
    
    if len_init_prt_str==0: # i.e. was called here and not from a func that didn't expect this output
        print(prt_str, flush=True)
        return rpdf
    return rpdf, prt_str
    

def colocalize(
        colocalization_params, rpdf, quant_img,
        prt_str='', intersection_threshold=0.001, intersection_metric=None
    ):
    prt_str += f"base_coloc_counts: {rpdf['colocal_id'].value_counts().to_dict()}"
    for clc_props in colocalization_params:
        coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
        rpdf, prt_str = get_colocalization(rpdf, quant_img, intersection_threshold=intersection_threshold, 
                                            coIds=coIds, coChs=coChs, assign_colocal_id=assign_colocal_id, 
                                            prt_str=prt_str, intersection_metric=intersection_metric)
    return rpdf, prt_str

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

def get_colocalization(
        rpdf, quant_img,
        intersection_threshold=0.00, coIds=(1,2), coChs=(0,1), assign_colocal_id=3, 
        prt_str='', intersection_metric=None,
    ):
    # parse args and setup for efficiency
    #######################################
    # support arbitrary functions that take as argument 3 dim array and return a float
    intersection_metric = calculate_iou if intersection_metric is None else intersection_metric
    st_time, len_init_prt_str = dt(), len(prt_str)

    # using last provided colocal_id as base for comparison
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
    
    # setup array to store results - cli, intersection_percent, *(largest_label for each other ch)
    result_default_cols = ['cli', 'intersection_percent']
    ch_intersecting_label_cols = [f'ch{coCh}_intersecting_label' for coCh in other_coChs]
    len_result_default_cols = len(result_default_cols) 
    results_arr = np.full((ch_bboxes.shape[0], len_result_default_cols + len(ch_intersecting_label_cols)), fill_value=-1.0)

    # colocalization
    #######################################
    for row_i in np.arange(ch_bboxes.shape[0]):
        CONTINUE_FLAG = False
        cli, ch_bbox, base_nuc_lbl = ch_df_indicies[row_i], ch_bboxes[row_i], ch_nuc_lbls[row_i]
        minx,miny,maxx,maxy = ch_bbox

        # extract bbox coords from nuclei image
        ch_nucleus_img_base = quant_img[minx:maxx, miny:maxy, base_coCh] # extract bbox around this label
        base_non_zero = (ch_nucleus_img_base==base_nuc_lbl).nonzero() # get nonzero coords for current label in base ch (remove other labels in this channel if present)

        # generate a mask for the other colocalization channels from the base channel
        ch_nucleus_img_others = quant_img[minx:maxx, miny:maxy, other_coChs]
        others_masked = np.zeros_like(ch_nucleus_img_others)
        for oCh_i in range(len(other_coChs)):
            o_masked = ch_nucleus_img_others[base_non_zero[0], base_non_zero[1], oCh_i]
            if o_masked.max() == 0: 
                CONTINUE_FLAG = True
                break
            inter_labels, inter_counts = np.unique(o_masked.ravel()[o_masked.nonzero()[0]], return_counts=True) # flatten, take non-zero values, count unique
            largest_intersecting_label = inter_labels[np.argmax(inter_counts)] # take label w highest count
            others_masked[base_non_zero[0], base_non_zero[1], oCh_i] = np.where(o_masked==largest_intersecting_label, 1, 0)
            results_arr[row_i, len_result_default_cols+oCh_i] = largest_intersecting_label

        if CONTINUE_FLAG: 
            continue # if any of the other channels had no label overlapping with base, skip

        # get intersection percentage
        all_stacked = np.concatenate((others_masked, np.where(ch_nucleus_img_base[:, :, np.newaxis]==base_nuc_lbl, 1, 0)), axis=2)
        intersection = intersection_metric(all_stacked)
        if intersection < intersection_threshold:
            continue
        # write to results
        results_arr[row_i, 0:len_result_default_cols] = cli, intersection
        

    # write results to input dataframe
    #######################################
    # convert successful results to dataframe and merge
    results = results_arr[(results_arr[:, 0]>-1), :]
    print(results.shape)
    df_results = pd.DataFrame(results, columns=result_default_cols + ch_intersecting_label_cols).set_index('cli')
    merge_on_df = rpdf.iloc[df_results.index.values, :]

    # remove the cols we are writing the results to if they already exist
    override_cols = [c for c in ch_intersecting_label_cols+['intersection_percent'] if c in merge_on_df.columns]
    coloc_df = (pd.merge(
        merge_on_df.drop(columns=override_cols), df_results, how='left', left_index=True, right_index=True)
        .assign(colocal_id = assign_colocal_id, ))

    # make cols if they don't exist in input rpdf
    for col in coloc_df.columns.to_list():
        if col not in rpdf:
            rpdf[col] = np.nan
    
    # add to input df
    rpdf_coloc = pd.concat([rpdf, coloc_df], ignore_index=True)
    prt_str += f"colocal_id_counts: {rpdf_coloc['colocal_id'].value_counts().to_dict()}\n"
    prt_str += f"colocalization completed in {dt()-st_time}.\n"
    return rpdf_coloc, prt_str



def numba_get_nuclei_counts_by_region(geojson_path, rpdf_coloc, ont, TEST=False, prt_str=''):
    load_t0 = dt()
    # parse qupath's geojson file containing region polygons
    geojson_objs = rp.load_geojson_objects(geojson_path)
    regionPoly_list = rp.extract_polyregions(geojson_objs, ont) 
    region_df = pd.DataFrame([rp.to_region_df_row() for rp in regionPoly_list])
    prt_str += f"num region polys: {len(regionPoly_list)}\n"
    polys_t1 = dt()
    
    # extract centroids from rpdf
    centroids =  np.array([c[::-1] for c in rpdf_coloc['centroid'].to_list()])
    # centroids = np.array([ast.literal_eval(c)[::-1] for c in rpdf_coloc['centroid'].to_list()]) # get centroids
    if TEST: centroids = centroids[:1000]
    load_t1 = dt()
    
    # localize nuclei to lowest structural level 
    localize_t0 = dt()
    nb_singles, nb_multis, infos = rp.separate_polytypes(regionPoly_list)
    separate_polytypes_t1 = dt()
    pp_result = rp.nb_process_polygons(nb_singles, nb_multis, centroids)
    prt_str +=(f"{len(np.where(pp_result > -1)[0])}/{pp_result.shape[0]} ({len(np.where(pp_result == -1)[0])} unassigned)")
    nb_process_polygons_t1 = dt()
    # map poly_index to poly info 
    rpdf_final = (
        pd.DataFrame(list(np.array(infos + [{k:np.nan for k in infos[0]}])[pp_result])) # add an empty dict for centroids unassigned to a polygon
        .assign(centroid_i = np.arange(centroids.shape[0]))
        .merge(rpdf_coloc, left_on='centroid_i', right_index=True, how='left') # rpdf_input.set_index('Unnamed: 0', drop=True)
    )
    
    prt_str += '\nTIMES:\n'
    prt_str +=(f'\tpolys took {polys_t1-load_t0}\n')
    prt_str +=(f'\tload took {load_t1-load_t0}\n')
    prt_str +=(f'\tlocalize took {dt()-localize_t0}\n')
    prt_str +=(f'\t  separate_polytypes took {separate_polytypes_t1-localize_t0}\n')
    prt_str +=(f'\t  nb_process_polygons took {nb_process_polygons_t1-separate_polytypes_t1}\n')
    prt_str +=(f'\t  map_df took {dt()-nb_process_polygons_t1}\n')
    prt_str +=(f'nuclei_counts_by_region complete took {dt()-load_t0}.')
    
    if TEST: rpdf_final = rpdf_final.loc[:1000, :]
    return rpdf_final, region_df, prt_str



class Dispatcher:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.prt_str = f"{'`'*75}\n{self.disp_i} --> {self.img_name}\n" # for display of info colllected during processing
    
    
    

    def run(self):
        print(f'\nstarting {self.disp_i} --> {self.img_name}', flush=True)
        datum_st_time = dt()
        exit_code = 0
        try:
            # get region props from quant image and perform colocalization
            rpdf, quant_img, self.prt_str = get_region_props(self, ch_colocal_id=self.ch_colocal_id, prt_str=self.prt_str)
            print('get_region_props finished')
            
            rpdf_coloc, self.prt_str = colocalize(self.colocalization_params, rpdf, quant_img, prt_str=self.prt_str)
            # rpdf_coloc, self.prt_str = get_colocalization(self, rpdf, quant_img, prt_str=self.prt_str)
            
            rpdf_final, region_df, self.prt_str = numba_get_nuclei_counts_by_region(self.datum.geojson_regions_dir_paths, rpdf_coloc, self.ont, TEST=self.TEST, prt_str=self.prt_str)
            # add eccentricity to rpdf
            rpdf_final = rpdf_final.assign(eccentricity = rpdf_final['axis_major_length']/rpdf_final['axis_minor_length'])
            # rpdf_final, region_df, prt_str = numba_get_nuclei_counts_by_region(self, rpdf_coloc) #get_nuclei_counts_by_region(self, rpdf_coloc)
            
            # save rpdfs and region_df
            if (not self.TEST) or (not self.WRITE_OUTPUT):
                write_csv(os.path.join(self.counts_dir, f'{self.img_name}_rpdf.csv'), rpdf_final) 
                write_csv(os.path.join(self.counts_dir, f'{self.img_name}_region_df.csv'), region_df) 
                # TODO: instead of writing rpdfs and region dfs can I make it so counts are stored in h5py and region_dfs are no longer needed?
                    # for counts need to be able to write to h5py concurrently merging data for the same animal... probably not feasible 
                    # for region dfs need to extract the info needed from it here
                        # which would work, but area is still needed, so maybe I can make another h5py or some database that stores region areas

            self.prt_str+=(f'{self.disp_i} --> {self.img_name} completed in {dt()-datum_st_time}\n')
            
            if self.TEST: return rpdf_final, region_df
            
        except Exception as e:
            self.prt_str += (f"An error occurred: {e}")
            exit_code = 1
        finally:
            print(self.prt_str, flush=True)
            return exit_code

    
            


def get_dispatchers(animals, TEST=False, BACKGROUND_SUBRACTION=False, WRITE_OUTPUT=True, 
                    COLOCALID_CH_MAP={0:1, 1:2, 2:0}, read_img_kwargs={}, colocalization_params=[],
    ):
    dispatchers, disp_count = [], 0
    for an in animals:
        # create counts directory for each animal
        counts_dir = verify_outputdir(os.path.join(an.base_dir, 'counts'))
        # read img kwargs 
        an_id = an.animal_id_to_int(an.animal_id)
        d_read_img_kwargs = {k:v if not callable(v) else v(an_id) for k,v in read_img_kwargs.items()} if read_img_kwargs else {}
        
        for d in an.get_valid_datums(['fullsize_paths', 'quant_dir_paths', 'geojson_regions_dir_paths']):
            dispatchers.append(Dispatcher(
                    disp_i=disp_count,
                    datum=d,
                    an_base_dir=an.base_dir,
                    counts_dir=counts_dir,
                    TEST=TEST,
                    BACKGROUND_SUBRACTION=BACKGROUND_SUBRACTION,
                    WRITE_OUTPUT=WRITE_OUTPUT,
                    read_img_kwargs = d_read_img_kwargs,
                    colocalization_params = deepcopy(colocalization_params),
                    cohort = an.cohort['cohort_name'],
                    img_name = Path(d.fullsize_paths).stem[:-4],
                    ont = arhfs.Ontology(),
                    ch_colocal_id=COLOCALID_CH_MAP,
                ))
            disp_count+=1
    print(f'num dispatchers {disp_count}', flush=True)
    return dispatchers




    


# TODO need way to interupt terminal


if __name__ == '__main__':
    # '''
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONDA CODE
    # conda activate stardist
    # cd "C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\ABBA_PQA\Quantification"
    # python "C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\ABBA_PQA\Quantification\img2df.py"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # '''
    
    # PARAMS
    ###################################################################################################
    TEST = bool(0) # process centroid subset
    WRITE_OUTPUT = bool(1) # write rpdf and region df to disk
    CLEAN = bool(0) # delete previous counts
    BACKGROUND_SUBRACTION = bool(1) # use tophat algorithm to correct for uneven illumination
    MULTIPROCESS = bool(1) # use Processpool for parallel processing, else run serially
    MAX_WORKERS = 12 # num of cores if running multithreded, for me 12
    READ_IMG_KWARGS = {'flip_gr_ch':lambda an_id: True if (an_id > 29 and an_id < 50) else False} 
    GET_ANIMALS = ['cohort2', 'cohort3', 'cohort4']
    start_i_disps, end_i_disps = 0, 1 #useful if run was interupted 
    

    # MAIN
    ###################################################################################################
    # initializations
    pp_st = time.time()
    ac = AnimalsContainer()
    ac.init_animals()
    animals = ac.get_animals(GET_ANIMALS)[:1]
    COLOCALID_CH_MAP = ac.ImgDB.get_colocalid_ch_map()
    COLOCALIZATION_PARAMS = ac.ImgDB.colocalizations
    if CLEAN: ac.clean_animal_dir(animals, 'counts') 
        
    # get dispatchers
    disps = get_dispatchers(
        animals, TEST=TEST, BACKGROUND_SUBRACTION=BACKGROUND_SUBRACTION, WRITE_OUTPUT=WRITE_OUTPUT, 
        read_img_kwargs=READ_IMG_KWARGS,
        COLOCALID_CH_MAP=COLOCALID_CH_MAP,  
        colocalization_params=COLOCALIZATION_PARAMS,
    ) [start_i_disps:end_i_disps]
    print(f'processing num dispatchers {len(disps)}', flush=True)
    
    # run
    if TEST:
        rpdf_final, region_df = disps[0].run()
        print('test finished in:', time.time() - pp_st, flush=True)
    
    elif MULTIPROCESS: # run multithreaded    
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = list(executor.map(Dispatcher.run, disps))
        print('processPool finished in:', time.time() - pp_st, flush=True)
    
    else: # single threaded
        for disp in disps:
            disp.run()
        print('serial processing finished in:', time.time() - pp_st, flush=True)


        


    
    





    
        
        

        




    


    

    

    

    
    



    
    
    
    
    
    
    










        







