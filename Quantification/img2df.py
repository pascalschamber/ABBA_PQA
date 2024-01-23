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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_image_processing import read_img, print_array_info, convert_16bit_image
from utilities.utils_general import verify_outputdir
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import utilities.utils_plotting as up
import core_regionPoly as rp



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
        - ANIMAL (TEL15, 46 imgs) processPool finished in: 418s
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

######################################################################################################
# tests
def test_visualize_polygon_overlay(datums):
    # visualize brain region boundaries overlayed ontop of fullsize image
    # takes a list of Datum as input and plots the extracted polygons
    for datum in datums:
        geojson_path = datum.geojson_regions_dir_paths
        fs_path = datum.fullsize_paths
        quant_path = datum.quant_dir_paths

        region_df = rp.load_geometries(geojson_path)
        fs_img = imread(fs_path)
        print_array_info(fs_img)
        
        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        disp_img = convert_16bit_image(np.moveaxis(fs_img, 0, -1))
        ax.imshow(disp_img)

        # Add the patch to the Axes
        for i in range(len(region_df)):
            coordinates = region_df.iloc[i,0]
            polygon = patches.Polygon(coordinates, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)
        plt.show()


def test_visualize_predictions():
    # visualize fullsize image and stardist nuclei detections
    animals = ac.get_animals('cohort4')
    an = animals[15]
    datums = an.get_valid_datums(['fullsize_paths', 'geojson_regions_dir_paths', 'quant_dir_paths'], warn=True)
    for datum in datums[:2]:
        fs_img = imread(datum.fullsize_paths)
        disp_img = convert_16bit_image(np.moveaxis(fs_img, 0, -1))
        quant_img = imread(datum.quant_dir_paths)
        
        print_array_info(fs_img)
        print_array_info(disp_img)
        print_array_info(quant_img)

        # Create figure and axes
        fig,axs = plt.subplots(2,3, figsize=(15,10))
        for i in range(3):
            j,k = i%3, i//3
            axs[k,j].imshow(quant_img[3000:4000, 4000:5000, i], cmap=up.lbl_cmap(), interpolation='nearest')
            axs[k+1,j].imshow(disp_img[3000:4000, 4000:5000, i])
        plt.show()


######################################################################################################
# main functions
# def load_objects(geojson_path):
#     with open(geojson_path) as f:
#         allobjects = geojson.load(f)
#     allfeatures = allobjects['features']
#     all_objs = [obj for obj in allfeatures if 'measurements' in obj['properties'].keys()]
#     all_obj_region_ids = [obj['properties']['measurements']['ID'] for obj in all_objs]
#     all_obj_region_sides = [obj['properties']['classification']['names'][0] for obj in all_objs]
#     all_exteriors = [geomertry_to_exteriors(obj) for obj in all_objs]
#     return all_objs

# def load_geometries(geojson_path):
#     with open(geojson_path) as f:
#         allobjects = geojson.load(f)
#     allfeatures = allobjects['features']
    
#     # get all objs except the root, which doesn't have an atlas id
#     all_objs = [obj for obj in allfeatures if 'measurements' in obj['properties'].keys()] # keeping for backwards compat.
#     all_objs = [obj for obj in all_objs if 'ID' in obj['properties']['measurements']] # NEW 2023_0808 replaced above line
#     all_obj_region_ids = [obj['properties']['measurements']['ID'] for obj in all_objs]
#     all_obj_region_sides = [obj['properties']['classification']['names'][0] for obj in all_objs]
#     all_obj_atlas_coords = extract_atlas_coords(all_objs) # NEW 2023_0808
    
#     all_exteriors = [geomertry_to_exteriors(obj) for obj in all_objs]
#     all_areas = [polygon_area(ext) for ext in all_exteriors]
#     region_df = get_region_df(all_exteriors, all_obj_region_ids, all_obj_region_sides, all_areas, all_obj_atlas_coords)
#     return region_df

# def get_region_df(all_exteriors, all_obj_region_ids, all_obj_region_sides, all_areas, all_obj_atlas_coords):
#     ''' convert the extracted region attributes to a dataframe '''
#     all_objx, all_objy, all_objz = zip(*all_obj_atlas_coords)
#     region_var_lens = [len(var) for var in [all_exteriors, all_obj_region_ids, all_obj_region_sides, all_areas, all_objx,all_objy,all_objz]]
#     assert len(set(region_var_lens)) == 1, f'region variable lens do not match {region_var_lens}'

#     region_df = pd.DataFrame([
#         {
#             'region_polygons':all_exteriors[i], 'region_ids':all_obj_region_ids[i], 'region_sides':all_obj_region_sides[i], 'region_areas':all_areas[i],
#             'atlas_x':all_objx[i], 'atlas_y':all_objy[i], 'atlas_z':all_objz[i]
#         }
#         for i in range(len(all_exteriors))])
#     return region_df


# def extract_atlas_coords(objs):
#     ''' NEW 2023_0808
#         extracts the atlas coords for each region from geojson file if present
#     '''
#     atlas_coords_not_found = 0 # store number of coords not found
#     atlas_measurements_keys = ['Atlas_X', 'Atlas_Y', 'Atlas_Z']
#     output = []
#     for obj in objs:
#         obj_output = []
#         measurements = obj['properties']['measurements']
#         for coord_key in atlas_measurements_keys:
#             if coord_key not in measurements: # check atlas coords exist
#                 atlas_coords_not_found += 1
#                 obj_output.append(None)
#             else:
#                 obj_output.append(measurements[coord_key])
#         output.append(obj_output)
#     if atlas_coords_not_found > 0: print(f'WARN --> atlas coords not extracted (num: {atlas_coords_not_found})', flush=True)
#     return output










        



#############################################################################################################################
# new code 5/28/23
def write_csv(fp, df):
    with open(fp, 'wb') as f:
        df.to_csv(f)

def load_nuc_intensity_imgs(disp, prt_str=''):
    # get images 
    st_time, len_init_prt_str = dt(), len(prt_str)
    prt_str += (f'fullsize, quant filenames: {Path(disp.datum.fullsize_paths).stem}, {Path(disp.datum.quant_dir_paths).stem}\n')
    quant_img = imread(disp.datum.quant_dir_paths)
    fullsize_img, _ = read_img(disp.datum.fullsize_paths)
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
    


def get_colocalization(disp, rpdf, quant_img, intersection_threshold=0.00, coIds=(1,2), coChs=(0,1), assign_colocal_id=3, prt_str=''):
    '''
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

    # colocalization
    colocal_instances_df = []
    for cl1i, ch1_bbox in zip(ch1_df_indicies, ch1_bboxes):
        
        # extract bbox coords from nuclei image
        minx,miny,maxx,maxy = ch1_bbox
        ch0_nucleus_img, ch1_nucleus_img = (quant_img[...,coChs[i]][minx:maxx, miny:maxy] for i in coChs) # ch1_nucleus_img = quant_img[...,coChs[1]][minx:maxx, miny:maxy]
        if ch0_nucleus_img.max() <= 0: # if no signal in ch0 skip
            continue 
        
        # get intersection percentage if any signal in ch0
        intersecting_coords = ch0_nucleus_img[ch1_nucleus_img.nonzero()]
        intersection_percent = np.sum(intersecting_coords>0)/intersecting_coords.shape[0]
        
        
        # check intersection percentage is above threshold
        if intersection_percent <= intersection_threshold: continue


        # if some intersecting label get the largest, i.e. get largest non zero label in zif channel
        inter_labels, inter_counts = np.unique(ch0_nucleus_img[ch1_nucleus_img.nonzero()], return_counts=True)
        nonzero_labels = np.nonzero(inter_labels)[0]
        nonzero_counts = inter_counts[np.nonzero(inter_labels)]
        largest_count_index = np.argmax(nonzero_counts)
        largest_intersecting_label_ch0 = inter_labels[nonzero_labels[largest_count_index]]


        # create a df row for this colocalization instance
        colocal4_row = dict(rpdf.loc[cl1i, :])
        colocal4_row['colocal_id'] = assign_colocal_id
        colocal4_row['intersection_p'] = intersection_percent
        colocal4_row['ch0_intersecting_label'] = int(largest_intersecting_label_ch0)
        #TODO: add colocal intensity for zif channel here? e.g. rpdf[(rpdf['colocal_id'==coIds[0]) & (rpdf['label']==largest_intersecting_label_ch0)]['intensity_mean'].values[0]
        colocal_instances_df.append(colocal4_row)
            

    # convert rows to df
    colocal_instances_df = pd.DataFrame(colocal_instances_df)
    
    # add new columns to rpdf, if they don't exist already e.g. if multiple colocalizations are being done
    rpdf_colocal = rpdf.copy(deep=True)
    if 'intersection_p' not in rpdf_colocal.columns:
        rpdf_colocal['intersection_p'] = np.nan
    if f'ch{coChs[0]}_intersecting_label' not in rpdf_colocal.columns:
        rpdf_colocal[f'ch{coChs[0]}_intersecting_label'] = np.nan
    rpdf_final = pd.concat([rpdf_colocal, colocal_instances_df], ignore_index=True)
    
    prt_str+=(f"{rpdf_final.value_counts('colocal_id')}\n")
    prt_str+=(f'colocalization complete took {timeit.default_timer() - st_time}\n')

    if len_init_prt_str==0: # i.e. was called here and not from a func that didn't expect this output
        print(prt_str, flush=True)
        return rpdf_final
    
    return rpdf_final, prt_str






# @nb.jit(nopython=True)
# def ray_casting(x, y, centroid_x, centroid_y):
#     num_vertices = len(x)
#     j = num_vertices - 1
#     odd_nodes = False
#     for i in range(num_vertices):
#         if ((y[i] < centroid_y and y[j] >= centroid_y) or (y[j] < centroid_y and y[i] >= centroid_y)) and (x[i] <= centroid_x or x[j] <= centroid_x):
#             odd_nodes ^= (x[i] + (centroid_y - y[i]) / (y[j] - y[i]) * (x[j] - x[i]) < centroid_x) # The ^ operator does a binary xor. a ^ b will return a value with only the bits set in a or in b but not both
#         j = i
#     return odd_nodes

# @nb.jit(nopython=True) #, parallel=True)
# def point_in_polygon(polygons_x, polygons_y, centroids):
#     num_polygons = len(polygons_x)
#     num_centroids = len(centroids)
#     results = np.zeros((num_centroids, num_polygons), dtype=np.bool_)

#     for i in nb.prange(num_centroids):
#         centroid_x, centroid_y = centroids[i]
#         for j in nb.prange(num_polygons):
#             if ray_casting(polygons_x[j], polygons_y[j], centroid_x, centroid_y):
#                 results[i, j] = True

#     return results

# def init_nb_points_in_polygon():
#     polygons_x, polygons_y, = nb.typed.List([np.asarray([i for i in range(100)]) for i in range(10)]), nb.typed.List([np.asarray([i for i in range(100)]) for i in range(10)])
#     centroids = np.array([[i for i in range(1000)], [i for i in range(1000)]]).T
#     res = point_in_polygon(polygons_x, polygons_y, centroids)




# def numba_get_nuclei_counts_by_region(self, rpdf_final):
#     regions_t0 = timeit.default_timer()
#     region_df = load_geometries(self.datum.geojson_regions_dir_paths)
#     polygons = [poly for poly in region_df.region_polygons.to_list()] 
#     polygons_x, polygons_y = nb.typed.List(np.asarray(poly).T[0] for poly in polygons), nb.typed.List(np.asarray(poly).T[1] for poly in polygons)

#     centroids =  np.array([c[::-1] for c in rpdf_final['centroid'].to_list()])
#     if self.TEST: centroids = centroids[:1000]
#     print(f'num regions: {len(region_df.region_polygons)}, num nuclei: {len(centroids)}', flush=True)

#     # Perform the calculation
#     results = point_in_polygon(polygons_x, polygons_y, centroids)
#     centroid_regions = {i:list(np.where(results[i])[0]) for i in range(len(centroids))} # store dict of regions for each nuclei

    
#     # append to rpdf_final a column that stores a list of each region a centroid is found in
#     if self.TEST: centroid_regions = dict(zip([i for i in range(len(rpdf_final))],list(centroid_regions.values()) + [np.nan for i in range(len(rpdf_final)-len(centroids))]))
#     rpdf_final['region_ids_list'] = list(centroid_regions.values())
#     # TODO filter so only parent, lowest st levels, and side?
    
#     print(f'get_nuclei_counts_by_region took: {timeit.default_timer() - regions_t0}', flush=True)
#     return rpdf_final, region_df


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
        try:
            # get region props from quant image and perform colocalization
            rpdf, quant_img, self.prt_str = get_region_props(self, ch_colocal_id=self.ch_colocal_id, prt_str=self.prt_str)
            rpdf_coloc, self.prt_str = get_colocalization(self, rpdf, quant_img, prt_str=self.prt_str)
            # rpdf_coloc = rpdf_coloc.assign(eccentricity = rpdf['axis_major_length']/rpdf['axis_minor_length']) # TODO: # add eccentricity to rpdf? 
            
            rpdf_final, region_df, self.prt_str = numba_get_nuclei_counts_by_region(self.datum.geojson_regions_dir_paths, rpdf_coloc, self.ont, TEST=self.TEST, prt_str=self.prt_str)
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
            return 0
        except Exception as e:
            self.prt_str += (f"An error occurred: {e}")
            return 1
        finally:
            print(self.prt_str, flush=True)


def get_dispatchers(animals, TEST=False, BACKGROUND_SUBRACTION=False, WRITE_OUTPUT=True, COLOCALID_CH_MAP={0:1, 1:2, 2:0}):
    dispatchers, disp_count = [], 0
    for an in animals:
        # create counts directory for each animal
        counts_dir = verify_outputdir(os.path.join(an.base_dir, 'counts'))
        
        for d in an.get_valid_datums(['fullsize_paths', 'quant_dir_paths', 'geojson_regions_dir_paths']):
            dispatchers.append(Dispatcher(
                    disp_i=disp_count,
                    datum=d,
                    an_base_dir=an.base_dir,
                    counts_dir=counts_dir,
                    TEST=TEST,
                    BACKGROUND_SUBRACTION=BACKGROUND_SUBRACTION,
                    WRITE_OUTPUT=WRITE_OUTPUT,
                    cohort = an.cohort['cohort_name'],
                    img_name = Path(d.fullsize_paths).stem[:-4],
                    ont = arhfs.Ontology(),
                    ch_colocal_id=COLOCALID_CH_MAP,
                ))
            disp_count+=1
    return dispatchers





if __name__ == '__main__':
    # '''
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONDA CODE
    # conda activate stardist
    # cd "C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification"
    # python 'C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\img2df.py'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # '''
    ###################################################################################################
    pp_st = time.time()
    ac = AnimalsContainer()
    ac.init_animals()

    ###################################################################################################
    # PARAMS
    COLOCALID_CH_MAP = {0:1, 1:2, 2:0} # dict mapping channels in intensity image to colocal id (e.g dapi (ch2) is colocal_id 0, zif (ch0) is 1, and GFP (ch1) is 2)
    TEST = bool(0) # process centroid subset
    WRITE_OUTPUT = bool(1) # write rpdf and region df to disk
    CLEAN = bool(0) # delete previous counts
    MULTIPROCESS = bool(1) # use Processpool for parallel processing, else run serially
    BACKGROUND_SUBRACTION = bool(1) # use tophat algorithm to correct for uneven illumination
    MAX_WORKERS = 12 # num of cores, if running multithreded, for me 12

    animals = ac.get_animals(['cohort2', 'cohort3', 'cohort4'])[:1]
    start_i_disps = 0 # start from this dispatcher, useful if run was interupted 


    ###################################################################################################
    # MAIN
    if CLEAN: ac.clean_animal_dir(animals, 'counts') 
        
    # get dispatchers
    dispatchers = get_dispatchers(animals, TEST=TEST, BACKGROUND_SUBRACTION=BACKGROUND_SUBRACTION, WRITE_OUTPUT=WRITE_OUTPUT, COLOCALID_CH_MAP=COLOCALID_CH_MAP)
    disps = dispatchers[start_i_disps:]
    print(f'num dispatchers {len(disps)}', flush=True)
    
    # init numba functions
    # init_nb_points_in_polygon() 

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


        


    
    





    
        
        

        




    


    

    

    

    
    



    
    
    
    
    
    
    










        







