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
import matplotlib.patches as mpatches
from pathlib import Path
from timeit import default_timer as dt
import pickle
import ast

# plotting imports
import seaborn as sns        

import concurrent
import concurrent.futures
import multiprocessing

# package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_image_processing import read_img, print_array_info, convert_16bit_image
import utilities.utils_general as ug
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import utilities.utils_plotting as up

import img2df_2023_0602 as img2df
import core_regionPoly as rp
from core_regionPoly import plot_compare_region_counts, plot_coordinates_inside_polygon
import compile_data_2023_0602 as compile


###################################################

DEBUG = bool(1)
# get animals data (rpdf, regions)
ont = arhfs.Ontology()
names_dict = dict(zip([d['name'] for d in ont.ont_ids.values()], ont.ont_ids.keys()))

ac = AnimalsContainer()
ac.init_animals()
# animals = ac.get_animals('cohort4')
animals = ac.get_animals(['cohort2', 'cohort3', 'cohort4'])
for an in animals[-1:]:
    an_t0 = dt()
    datums = an.get_valid_datums(['fullsize_paths', 'geojson_regions_dir_paths', 'quant_dir_paths'], warn=True)
    
    for datum in datums[:1]:
        # TODO it is potentially possible to remove the numpy warning when processing geojson objects with obj specificication
        # would need to fix how I initially separate polys 
        # all_objs = rp.load_geojson_objects(datum.geojson_regions_dir_paths)
        # for fi, aFeature in enumerate(all_objs[:]):
        #     print(fi)
        #     coord_arrs, num_poly_coords = rp.get_coordinates(aFeature)

        #     polyType = aFeature['geometry']['type']
        #     if polyType == 'MultiPolygon':
        #         coord_arrs2 = [np.array(el) for el in aFeature['geometry']['coordinates']]
        #     elif polyType == 'Polygon':
        #         print(f"\t{len(aFeature['geometry']['coordinates'])}")
        #         coord_arrs2 = [np.array(aFeature['geometry']['coordinates'])]
        #     else: 
        #         raise ValueError(f"{polyType} is not handled")





        
        datum_t0 = dt()
        prt_str = '' # add to this string for display of debugging
        geojson_path = datum.geojson_regions_dir_paths
        fs_path = datum.fullsize_paths
        quant_path = datum.quant_dir_paths
        prt_str += f"geojson_path: {geojson_path}\n"

        # prepare for first pass through and generate "GT" to compare results to 
        load_t0, polys_t0 = dt(), dt()
        # parse qupath's geojson file containing region polygons
        geojson_objs = rp.load_geojson_objects(geojson_path)
        regionPoly_list = rp.extract_polyregions(geojson_objs, ont) 
        prt_str += f"num region polys: {len(regionPoly_list)}\n"
        region_df = pd.DataFrame([rp.to_region_df_row() for rp in regionPoly_list])
        polys_t1 = dt()
        
        # generate counts to compare to
        rpdf_input = pd.read_csv(datum.rpdf_paths)
        centroids = np.array([ast.literal_eval(c)[::-1] for c in rpdf_input['centroid'].to_list()]) # get centroids
        region_df_GT = pd.read_csv(datum.region_df_paths)
        region_count_df_GT = compile.get_nuclei_per_region_df(rpdf_input, region_df_GT, an.base_dir)
        load_t1 = dt()
        
        # localize nuclei to lowest structural level 
        localize_t0 = dt()
        nb_singles, nb_multis, infos = rp.separate_polytypes(regionPoly_list)
        separate_polytypes_t1 = dt()
        pp_result = rp.nb_process_polygons(nb_singles, nb_multis, centroids)
        prt_str +=(f"{len(np.where(pp_result > -1)[0])}/{pp_result.shape[0]} ({len(np.where(pp_result == -1)[0])} unassigned)")
        nb_process_polygons_t1 = dt()
        # map poly_index to poly info 
        rpdf = (
            pd.DataFrame(list(np.array(infos + [{k:np.nan for k in infos[0]}])[pp_result])) # add an empty dict for centroids unassigned to a polygon
            .assign(centroid_i = np.arange(centroids.shape[0]))
            .merge(rpdf_input.set_index('Unnamed: 0', drop=True), left_on='centroid_i', right_index=True, how='left')
        )
        localize_t1 = dt()

        def pool_centroids_by_region(
                rpdf, 
                stain_col_names=['nDapi', 'nZif', 'nGFP', 'nBoth'], 
                info_cols={'reg_id': int, 'st_level': int, 'region_name': str, 'reg_side': str, 'acronym': str, 'region_area': float}
            ):
            """ convert to summary of nuclei by colocal_id per region
            # Group by 'region_id' and 'colocal_id', and count the occurrences
            # Pivot the table to get 'colocal_id' as separate columns for each 'region_id'"""
            pivot_table = (rpdf
               .groupby(['poly_index', 'colocal_id'])
               .size()
               .reset_index(name='count')
               .pivot_table(index='poly_index', columns='colocal_id', values='count', fill_value=0)
               .assign(poly_index=lambda df: df.index.values)
               .reset_index(drop=True)
               .rename(columns=dict(zip(range(len(stain_col_names)), stain_col_names))) # rename colocal id cols to name of stain
               .assign(**{col: pd.Series(dtype=dt) for col, dt in info_cols.items()})
            )
            for row_i, row in pivot_table.iterrows(): # add info
                pivot_table.loc[row_i, info_cols.keys()] = rpdf[rpdf['poly_index']==row['poly_index']][info_cols.keys()].values[0]

            return pivot_table
        def get_parent_poly_ids(region_df, parent_ids, side):
            # map region ids to poly_index
            out = []
            for pid in parent_ids:
                reg_df_row = region_df[(region_df['region_ids']==pid) & (region_df['region_sides']==side)]
                assert len(reg_df_row) == 1
                out.append(int(reg_df_row['poly_index'].values[0]))
            return out
        def get_attrs_ont(d):
            return {k:v for k,v in d.items() if k != 'children'}
        def add_to_count_dict(fpi, all_region_counts, base_counts):
            if fpi not in all_region_counts:
                all_region_counts[fpi] = base_counts()
            else:
                bc = base_counts()
                for k,v in all_region_counts[fpi].items():
                    all_region_counts[fpi][k] = v + bc[k]
            return all_region_counts
        def propogate_region_counts(regionPoly_list, pivot_table, region_df, ont):
            # for compiling need to explode the region heirarchy based on the assigned lowest level region ids so all parent regions are quantified
            all_region_counts = {}
            all_regPoly_reg_ids = [p.obj_id for p in regionPoly_list]
            
            for rowi, row in pivot_table.iterrows():
                base_reg_id, base_side, base_poly_index = row['reg_id'], row['reg_side'], int(row['poly_index'])
                base_counts = lambda: row[stain_col_names].to_dict() # need to create this every time or it overinflates the values 
                all_region_counts = add_to_count_dict(base_poly_index, all_region_counts, base_counts)
                
                # filter_parent_ids by those that have polys
                filtered_pids = [el for el in arhfs.get_all_parents(ont, base_reg_id) if el in all_regPoly_reg_ids]
                filtered_polyis = get_parent_poly_ids(region_df, filtered_pids, base_side) # use parent reg_id to get poly index
                
                for fpi in filtered_polyis: # add the counts to the parent
                    all_region_counts = add_to_count_dict(fpi, all_region_counts, base_counts)
            # sort by polyi
            all_region_counts = {outer_key: all_region_counts[outer_key] for outer_key in sorted(all_region_counts)}
            # convert to dataframe
            return pd.DataFrame.from_dict(all_region_counts, orient='index')
        
        propogate_counts_t0 = dt()
        stain_col_names = ['nDapi',	'nZif',	'nGFP',	'nBoth']
        region_count_df = pool_centroids_by_region(rpdf, stain_col_names=stain_col_names)
        region_count_df = propogate_region_counts(regionPoly_list, region_count_df, region_df, ont)
        toadd = dict(zip(['region_name', 'acronym', 'st_level'], ['name', 'acronym', 'st_level'])) # TODO?
        region_count_df = pd.merge(region_count_df, region_df, left_index=True, right_index=True)
        region_count_df = region_count_df.rename(columns={'region_ids':'reg_id'})
        
        for col, ont_attr in toadd.items():
            region_count_df[col] = arhfs.get_attributes_for_list_of_ids(ont.ont_ids, region_count_df['reg_id'].to_list(), ont_attr)
        cpropogate_counts_t1 = dt()
        
        prt_str += 'TIMES:\n'
        prt_str +=(f'\tpolys took {polys_t1-polys_t0}\n')
        prt_str +=(f'\tload took {load_t1-load_t0}\n')
        prt_str +=(f'\tlocalize took {localize_t1-localize_t0}\n')
        prt_str +=(f'\t  separate_polytypes took {separate_polytypes_t1-localize_t0}\n')
        prt_str +=(f'\t  nb_process_polygons took {nb_process_polygons_t1-separate_polytypes_t1}\n')
        prt_str +=(f'\t  map_df took {localize_t1-nb_process_polygons_t1}\n')
        prt_str +=(f'\tpropogate_counts took {cpropogate_counts_t1-propogate_counts_t0}\n')
        prt_str +=(f'\tdatum took: {dt()-datum_t0}.\n')
        print(prt_str, flush=True)

        if DEBUG:
            plot_compare_region_counts(region_count_df_GT, region_count_df, stain_col_names, st_lvl_max=10)
            plot_coordinates_inside_polygon(
                rpdf, nb_singles, infos, regionPoly_list, centroids, get_polyi=72, get_colocal_id=0)
        
        

        

       






        
        

        