import sys
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
import multiprocessing as mp
import timeit
from timeit import default_timer as dt
from collections import Counter
import ast
import seaborn as sns

import concurrent.futures
from datetime import datetime
import json
from tabulate import tabulate
import scipy
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import utilities.utils_general as ug
import core_regionPoly as rp



'''
##################################################################################################
    Notes
        - 0602 
            - changing function of this file to load all rpdfs for each animal and generate region counts here (opposed to in img2df.py) 
            - purpose is to apply thresholding to nuclei counts here
            - main ouput is still a single df containing region counts for all animals
    TODO
        - 0827
            - is nuc per region affected by filtering process? 
##################################################################################################
'''

def filter_nuclei(
        threshold_dict, rpdf_coloc, 
        group_labels_col='img_name', return_labels=False, 
        clc_nucs_info={3:{'intersecting_label_column':'ch0_intersecting_label', 'intersecting_colocal_id':1}}
    ):
    ''' 
        DESCRIPTION
            filter dataframe of region props 
        ARGS
        - threshold_dict (dict) --> dictionary mapping colocal_ids to dict of region_props (col in df) and min/max vals
        - rpdf_coloc (pd.DataFrame) --> df containing region props
        - group_labels_cols (str, None) --> col in rpdf_coloc used to separate duplicate nuclei labels when combining rpdfs (e.g. from different images)
        - return_labels (bool) --> whether to return valid/invalid labels
        - clc_nucs_info (dict) --> dictionary mapping colocal_ids that are result of colocalization to dict of:
            - intersecting_label_column (str) --> name of column containing intersecting labels
            - intersecting_colocal_id (int) --> colocal_id value of intersecting nuclei
    '''
    # prepare ouputs
    thresholded_rpdfs, valid_labels, invalid_labels, filtered_counts = {}, {}, {}, {'init':get_colocal_id_counts(rpdf_coloc)}
    # get present colocal_ids
    all_colocal_ids = sorted(rpdf_coloc['colocal_id'].unique())

    # assign cols to df to track labels by img_name
    rpdf_coloc, new_cols, label_col, clc_nucs_info = prepare_group_label_columns(rpdf_coloc, group_labels_col, clc_nucs_info)

    for colocal_id in all_colocal_ids:
        cdf = rpdf_coloc.loc[rpdf_coloc['colocal_id'] == colocal_id]
        filtered_counts[colocal_id] = {}
        all_labels = set(cdf[label_col].to_list())


        # threshold regionprops
        if colocal_id in threshold_dict:
            cdf, rp_filtercounts = region_prop_table_filter(cdf, threshold_dict[colocal_id]) 
            filtered_counts[colocal_id]['byFilter']: rp_filtercounts
        filtered_counts[colocal_id]['post_filters']: get_colocal_id_counts(cdf)


        # cdf = clean_up_colocal_nuclei(cdf, colocal_id, clc_nucs_info, valid_labels, group_labels_col)
        if (clc_nucs_info is not None) and (colocal_id in clc_nucs_info):
            # get col name with intersecting label and intersecting colocal id
            intersecting_label_column, intersecting_colocal_id = \
                clc_nucs_info[colocal_id]['intersecting_label_column'], clc_nucs_info[colocal_id]['intersecting_colocal_id']
            drp_prev_col = \
                intersecting_label_column if not 'grouped_col' in clc_nucs_info[colocal_id] else clc_nucs_info[colocal_id]['grouped_col']
            
            # remove  nuclei that are colocal with same object
            cdf = drop_duplicate_colocalizations(cdf, intersecting_label_column, group_by_col=group_labels_col)
            filtered_counts[colocal_id]['post_drop_duplicates'] = get_colocal_id_counts(cdf)
            # remove any colocal nuclei where overlapping zif nuc was removed previously (e.g. area/axis length)
            cdf = drop_previously_removed(cdf, drp_prev_col, valid_labels[intersecting_colocal_id])
            filtered_counts[colocal_id]['post_drop_previously_removed'] = get_colocal_id_counts(cdf)

        valid_labels[colocal_id] = cdf[label_col].to_list()
        invalid_labels[colocal_id] = list(all_labels - set(valid_labels[colocal_id]))
        thresholded_rpdfs[colocal_id] = cdf

    thresholded_rpdf = pd.concat(thresholded_rpdfs.values())
    filtered_counts['final'] = get_colocal_id_counts(thresholded_rpdf)
    if return_labels:
        return thresholded_rpdf, filtered_counts, valid_labels, invalid_labels
    return thresholded_rpdf, filtered_counts

def region_prop_table_filter(cdf, t_dict, this_clc_id):
    ''' filter region props data frame of a single colocal_id using dictionary of min,max vals '''
    filter_counts_post_thresh = {'pre_filters':get_colocal_id_counts(cdf, [this_clc_id])[this_clc_id]}
    for prop_name, (min_value, max_value) in t_dict.items():
        min_limit = float('-inf') if min_value is None else min_value
        max_limit = float('inf') if max_value is None else max_value
        cdf = cdf.loc[(cdf[prop_name] > min_limit) & (cdf[prop_name] < max_limit)]
        filter_counts_post_thresh[f'post_{prop_name}'] = get_colocal_id_counts(cdf, [this_clc_id])[this_clc_id]
    return cdf, filter_counts_post_thresh

# def clean_up_colocal_nuclei(cdf, colocal_id, clc_nucs_info, valid_labels, group_labels_col):
#     ''' clean up colocal detections, removing nuclei colocal w/same intersecting label, and intersecting labels that were removed previously '''
#     if (clc_nucs_info is not None) and (colocal_id in clc_nucs_info):
#         # get col name with intersecting label and intersecting colocal id
#         intersecting_label_column, intersecting_colocal_id = \
#             clc_nucs_info[colocal_id]['intersecting_label_column'], clc_nucs_info[colocal_id]['intersecting_colocal_id']
#         drp_prev_col = \
#             intersecting_label_column if not 'grouped_col' in clc_nucs_info[colocal_id] else clc_nucs_info[colocal_id]['grouped_col']
        
#         # remove  nuclei that are colocal with same object
#         cdf = drop_duplicate_colocalizations(cdf, intersecting_label_column, group_by_col=group_labels_col)
#         # remove any colocal nuclei where overlapping zif nuc was removed previously (e.g. area/axis length)
#         cdf = drop_previously_removed(cdf, drp_prev_col, valid_labels[intersecting_colocal_id])

#     return cdf

def drop_duplicate_colocalizations(rpdf_colocal, intersecting_label_column, group_by_col='img_name'):
    ''' drop colocal nuclei that are colocal with same object, keeping highest intersection '''
    if group_by_col is None:
        dropped_dupes = (rpdf_colocal
         .sort_values([intersecting_label_column, 'intersection_percent'], ascending=[True, False])
         .drop_duplicates(subset=[intersecting_label_column], keep='first'))
    else:
        dropped_dupes = (rpdf_colocal
                .groupby(group_by_col, as_index=False)
                .apply(lambda x: x.sort_values(by=[intersecting_label_column, 'intersection_percent'], ascending=[True, False])
                            .drop_duplicates(subset=[intersecting_label_column], keep='first'))
                )
    try:
        dropped_dupes = dropped_dupes.droplevel(level=0) if group_by_col is not None else dropped_dupes
    except ValueError:
        pass
    return dropped_dupes

    
def prepare_group_label_columns(rpdf_coloc, group_labels_cols, clc_nucs_info):
    # assign col to df to track labels by img_name
    og_cols = set(rpdf_coloc.columns.to_list())
    if group_labels_cols is not None:
        rpdf_coloc = rpdf_coloc.assign(grouped_labels=rpdf_coloc[group_labels_cols].astype('str') + '###' + rpdf_coloc['label'].astype('str'))
        if clc_nucs_info is not None:
            for k, v in clc_nucs_info.items():
                grp_col_name = f"grouped_labels_{v['intersecting_label_column']}"
                clc_nucs_info[k]['grouped_col'] = grp_col_name # add new col name to clc_nuc_info
                # rpdf_coloc[grp_col_name] = np.nan
                rpdf_coloc.loc[~rpdf_coloc[v['intersecting_label_column']].isna(), grp_col_name] = \
                    rpdf_coloc[group_labels_cols].astype('str') + '###' + rpdf_coloc[v['intersecting_label_column']].fillna(-1).astype('int').astype('str')
    else:
        # if group_labels_col not set, check if need to group labels, count duplicate labels for each colocal_id
        assert rpdf_coloc.groupby('colocal_id').apply(lambda group: group['label'].duplicated().sum()).sum() == 0, \
            'there are duplicate labels for each colocal_id, set group_labels_col to separate labels by image they came from'
    new_cols = list(set(rpdf_coloc.columns.to_list()) - og_cols) 
    label_col = 'label' if len(new_cols) == 0 else new_cols[0]
    return rpdf_coloc, new_cols, label_col, clc_nucs_info

def drop_previously_removed(rpdf_colocal, intersecting_label_column, keep_labels_list):
    '''remove any colocal nuclei where overlapping zif nuc was removed previously (e.g. area/axis length)'''
    return rpdf_colocal[rpdf_colocal[intersecting_label_column].isin(keep_labels_list)]


def get_colocal_id_counts(rpdf, all_colocal_ids):
    # all_colocal_ids is a sorted list of every colocal id found in ImgDB
    vc = rpdf.value_counts('colocal_id').to_dict()
    return {i:0 if i not in vc else vc[i] for i in all_colocal_ids}




class Dispatcher:
    def __init__(self, **kwargs):
        # set/init attributes
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.filter_counts = {}
        self.aidf = None
        if self.TEST: 
            self.rpdfs_og = []
            self.rpdfs_filtered = []
        self.init_animal_info()

    def init_animal_info(self):
        # prep animal_info_df
        if self.aidf_path is not None:
            self.aidf = pd.read_excel(self.aidf_path)
            # rename mapping cols and ensure animal_id col exists
            self.aidf = self.aidf.rename(columns=self.animal_info_df_map_cols)
            assert 'animal_id' in self.aidf.columns, f"animal_id col not found in animal info df\n{self.aidf.columns.to_list()}"
        

    def write_csv(self, outpath, df):
        with open(outpath, 'wb') as f:
            df.to_csv(f, index=False)
        return 0
    
    def map_animal_info(self, counts_df, animal_info_row, datum):
        # parse colummns and values to map onto counts_df
        default_map_colvals = {'cohort':self.cohort, 'animal_id':self.an.animal_id, 'img_name':self.get_image_name(datum)}
        if (self.aidf is not None) and (self.animal_info_df_map_cols is not None):
            for _, mapped_col_name in self.animal_info_df_map_cols.items():
                if mapped_col_name not in default_map_colvals: # skip mapping animal_id again
                    default_map_colvals[mapped_col_name] = animal_info_row[mapped_col_name].values[0]
        return counts_df.assign(**default_map_colvals)
        
    def get_image_name(self, datum):
        return Path(datum.fullsize_paths).stem[:-4]
    
    
    def run(self):
        # get row in aidf that corresponds to this animal
        animal_info_row = self.aidf[self.aidf['animal_id']==self.an.animal_id] if self.aidf is not None else None
        datums = self.an.get_valid_datums(['rpdf_paths', 'region_df_paths', 'geojson_regions_dir_paths'])
        print(len(datums))
        ont = arhfs.Ontology()

        collected_dfs = []
        for datum in datums:
            img_name = self.get_image_name(datum)
            rpdf = pd.read_csv(datum.rpdf_paths)
            region_df = pd.read_csv(datum.region_df_paths)
            
            # check extant regionprops vs filters, or maybe move this out to init disps if possible?
            # TODO, create class called FatalError
            verify_threshold_params(self.threshold_dict, rpdf)

            # for colocalizations, map colocal (intersecting) labels to their intensities
            for clc_params in self.colocalization_params:
                for clc_i in range(len(clc_params['coChs'])-1):
                    clc_ch = clc_params['coChs'][clc_i]
                    clc_id = clc_params['coIds'][clc_i]
                    intensity_col = f"ch{clc_ch}_intensity"
                    if intensity_col in rpdf.columns: continue # if already did this col 
                    itx_col = f"ch{clc_ch}_intersecting_label"
                    rpdf[intensity_col] = rpdf[itx_col].map(
                        rpdf.loc[rpdf['colocal_id'] == clc_id].set_index('label')['intensity_mean'])


            # rpdf['zif_intensity'] = rpdf['ch0_intersecting_label'].map(
            #     rpdf.loc[rpdf['colocal_id'] == 1].set_index('label')['intensity_mean'])

            # filter nuclei
            if self.TEST: self.rpdfs_og.append(rpdf)
            rpdf_filtered, fcs = filter_nuclei(self.threshold_dict, rpdf, group_labels_col=None, return_labels=False, 
                clc_nucs_info=self.clc_nucs_info)
            
            self.filter_counts[img_name] = fcs



class Dispatcher:
    def __init__(self, **kwargs):
        # set/init attributes
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.filter_counts = {}
        self.aidf = None
        if self.TEST: 
            self.rpdfs_og = []
            self.rpdfs_filtered = []
        self.init_animal_info()

    def init_animal_info(self):
        # prep animal_info_df
        if self.aidf_path is not None:
            self.aidf = pd.read_excel(self.aidf_path)
            # rename mapping cols and ensure animal_id col exists
            self.aidf = self.aidf.rename(columns=self.animal_info_df_map_cols)
            assert 'animal_id' in self.aidf.columns, f"animal_id col not found in animal info df\n{self.aidf.columns.to_list()}"
        
    def write_csv(self, outpath, df):
        with open(outpath, 'wb') as f:
            df.to_csv(f, index=False)
        return 0
    
    def map_animal_info(self, counts_df, animal_info_row, datum):
        # parse colummns and values to map onto counts_df
        default_map_colvals = {'cohort':self.cohort, 'animal_id':self.an.animal_id, 'img_name':self.get_image_name(datum)}
        if (self.aidf is not None) and (self.animal_info_df_map_cols is not None):
            for _, mapped_col_name in self.animal_info_df_map_cols.items():
                if mapped_col_name not in default_map_colvals: # skip mapping animal_id again
                    default_map_colvals[mapped_col_name] = animal_info_row[mapped_col_name].values[0]
        return counts_df.assign(**default_map_colvals)
        
    def get_image_name(self, datum):
        return Path(datum.fullsize_paths).stem[:-4]
    
    def run(self):
        # get row in aidf that corresponds to this animal
        animal_info_row = self.aidf[self.aidf['animal_id']==self.an.animal_id] if self.aidf is not None else None
        datums = self.an.get_valid_datums(['rpdf_paths', 'region_df_paths', 'geojson_regions_dir_paths'])
        print(len(datums))
        ont = arhfs.Ontology()

        collected_dfs = []
        for datum in datums:
            img_name = self.get_image_name(datum)
            rpdf = pd.read_csv(datum.rpdf_paths)
            region_df = pd.read_csv(datum.region_df_paths)
            
            # check extant regionprops vs filters, or maybe move this out to init disps if possible?
            # TODO, create class called FatalError
            verify_threshold_params(self.threshold_dict, rpdf)

            # for colocalizations, map colocal (intersecting) labels to their intensities
            for clc_params in self.colocalization_params:
                for clc_i in range(len(clc_params['coChs'])-1):
                    clc_ch = clc_params['coChs'][clc_i]
                    clc_id = clc_params['coIds'][clc_i]
                    intensity_col = f"ch{clc_ch}_intensity"
                    if intensity_col in rpdf.columns: continue # if already did this col 
                    itx_col = f"ch{clc_ch}_intersecting_label"
                    rpdf[intensity_col] = rpdf[itx_col].map(
                        rpdf.loc[rpdf['colocal_id'] == clc_id].set_index('label')['intensity_mean'])


            # rpdf['zif_intensity'] = rpdf['ch0_intersecting_label'].map(
            #     rpdf.loc[rpdf['colocal_id'] == 1].set_index('label')['intensity_mean'])

            # filter nuclei
            if self.TEST: self.rpdfs_og.append(rpdf)
            rpdf_filtered, fcs = filter_nuclei(self.threshold_dict, rpdf, group_labels_col=None, return_labels=False, 
                clc_nucs_info=self.clc_nucs_info)
            
            self.filter_counts[img_name] = fcs

def get_dispatchers(
        animal_info_df_path, animal_info_df_map_cols, animals, 
        count_channel_names, clc_nucs_info, 
        THRESHOLD_DICTS, outdir, THE_DATE, MODE, 
        colocalization_params=[], TEST=False
    ):
    dispatchers = []
    for an in animals:
        ds = an.get_valid_datums(['rpdf_paths', 'region_df_paths'], warn=True)
        cohort = an.cohort['cohort_name']
        
        if len(ds) == 0: # check that there are valid paths
            print(f'skipping {an.animal_id} --> no valid datums')
            continue

        dispatchers.append(
            Dispatcher(
                an = an, 
                cohort = cohort,
                aidf_path = animal_info_df_path,
                animal_info_df_map_cols = animal_info_df_map_cols,
                count_channel_names = count_channel_names,
                clc_nucs_info = clc_nucs_info,
                colocalization_params = deepcopy(colocalization_params),
                threshold_dict = THRESHOLD_DICTS[cohort].copy(),
                TEST=TEST,  
        ))

    # save threshold dicts to json
    ug.verify_outputdir(outdir)
    with open(os.path.join(outdir, f"{THE_DATE}_ThresholdsByCohort.json"), 'w') as f:
        json.dump(THRESHOLD_DICTS, f)

    print(f'num dispatchers: {len(dispatchers)}\nmode: {MODE} (test={TEST}).\nThresholds:{THRESHOLD_DICTS}\n\n')
    return dispatchers


if __name__ == '__main__':
    """
    Description
        Reduce dfs with all nuclei for each image and animal to counts per region per animal
    Params and Arguments
        animal_info_df_path [str] - path to df that maps additional information on to each animal
        animal_info_df_map_cols [list of str] - cols to map to each animal_id, must contain col for 'animal_id'
        count_channel_names [list of str] - aliases to assign to colocal ids (becomes col names for aggregated counts)
    """
    # PARAMS
    ##################################################################################################
    #TODO ensure all manually defined values are accessible here
    # allow aidf to be none
    animal_info_df_path = r'D:\ReijmersLab\TEL\slides\8-8-22 TEL Project - FC vs FC+EXT TetTag Experiment.xlsx'
    animal_info_df_map_cols = {'treatment':'animal_id', 'treatment.1':'group', 'sex':'sex', 'strain':'strain'}
    # count_channel_names = ['nDapi', 'nZif', 'nGFP', 'nBoth']
    # clc_nuc_info = {3:{'intersecting_label_column':'ch0_intersecting_label', 'intersecting_colocal_id':1}}
    OUTDIR = r'D:\ReijmersLab\TEL\slides\quant_data\counts_test_newpipeline_2024_0121'
    
    # ARGUMENTS
    #######################################################
    TEST = bool(0)
    MULTIPROCESS = bool(1)
    MODE = ['reduce_by_region', 'compile_animal_rpdfs'][0]
    THE_DATE = datetime.now().strftime('%Y_%m%d_%H%M%S')    
    COHORTS = ['cohort2', 'cohort3', 'cohort4']
    animals_upto = 1 if TEST else 1

    ##################################################################################################
    # MAIN
    start_time = timeit.default_timer()
    ac = AnimalsContainer()
    ac.init_animals()
    animals = ug.flatten_list([ac.get_animals(COHORT) for COHORT in COHORTS]) [:animals_upto]
    THRESHOLD_DICTS = ac.ImgDB.get_threshold_params()
    COUNT_CHANNEL_NAMES = ac.ImgDB.get_count_channel_names()
    CLC_NUC_INFO = ac.ImgDB.get_clc_nuc_info()
    COLOCALIZATION_PARAMS = ac.ImgDB.colocalizations

    # animals = ac.get_animals(['TEL17', 'TEL24', 'TEL48', 'TEL44', 'TEL53', 'TEL55'])
    dispatchers = get_dispatchers(
        animal_info_df_path, animal_info_df_map_cols, animals, 
        COUNT_CHANNEL_NAMES, CLC_NUC_INFO,
        THRESHOLD_DICTS, OUTDIR, THE_DATE, MODE, 
        colocalization_params=COLOCALIZATION_PARAMS, TEST=TEST)
    
    ####################
    # run
    self = dispatchers[0]

    datums = self.an.get_valid_datums(['rpdf_paths', 'region_df_paths', 'geojson_regions_dir_paths'])
    print(len(datums))

    collected_dfs = []
    for datum in datums[18:19]:
        img_name = self.get_image_name(datum)
        rpdf = pd.read_csv(datum.rpdf_paths)
        region_df = pd.read_csv(datum.region_df_paths)
        
        # check extant regionprops vs filters, or maybe move this out to init disps if possible?
        # verify_threshold_params(self.threshold_dict, rpdf)

        ###############################################################################
        # for colocalizations, map colocal (intersecting) labels to their intensities
        for clc_params in self.colocalization_params:
            for clc_i in range(len(clc_params['coChs'])-1):
                clc_ch = clc_params['coChs'][clc_i]
                clc_id = clc_params['coIds'][clc_i]
                intensity_col = f"ch{clc_ch}_intensity"
                if intensity_col in rpdf.columns: continue # if already did this col 
                itx_col = f"ch{clc_ch}_intersecting_label"
                rpdf[intensity_col] = rpdf[itx_col].map(
                    rpdf.loc[rpdf['colocal_id'] == clc_id].set_index('label')['intensity_mean'])
        # replaced /////
        # rpdf['zif_intensity'] = rpdf['ch0_intersecting_label'].map(
        #     rpdf.loc[rpdf['colocal_id'] == 1].set_index('label')['intensity_mean'])
                

        
        ###############################################################################
        # # filter nuclei
        # if self.TEST: self.rpdfs_og.append(rpdf)
        # rpdf_filtered, fcs = filter_nuclei(self.threshold_dict, rpdf, group_labels_col=None, return_labels=False, 
        #     clc_nucs_info=self.clc_nucs_info)
        
        # self.filter_counts[img_name] = fcs

        #####################
                # INPUT
        threshold_dict= self.threshold_dict
        rpdf_coloc = rpdf 
        group_labels_col=None 
        return_labels=False 
        clc_nucs_info=self.clc_nucs_info
        all_colocal_ids = sorted(list(ac.ImgDB.colocal_ids.keys()))
        #####################
        thresholded_rpdfs, valid_labels, invalid_labels, filtered_counts = {}, {}, {}, {'init':get_colocal_id_counts(rpdf_coloc, all_colocal_ids), 'final':None}
        # prepare filter filtered_counts for each colocal_id
        for c in all_colocal_ids: filtered_counts[c] = {}
        # get present colocal_ids
        present_colocal_ids = sorted(rpdf_coloc['colocal_id'].unique())
        # assign cols to df to track labels by img_name
        rpdf_coloc, new_cols, label_col, clc_nucs_info = prepare_group_label_columns(rpdf_coloc, group_labels_col, clc_nucs_info)
        

        # MAIN CLC LOOP FOR FILTERING
        #############################
        for colocal_id in all_colocal_ids:
            cdf = rpdf_coloc.loc[rpdf_coloc['colocal_id'] == colocal_id]
            
            all_labels = set(cdf[label_col].to_list())


            # threshold regionprops
            if colocal_id in threshold_dict:
                # TODO add method
                cdf, rp_filtercounts = region_prop_table_filter(cdf, threshold_dict[colocal_id], colocal_id) 
                for k,v in rp_filtercounts.items():
                    filtered_counts[colocal_id][k] = v
            
            # cdf = clean_up_colocal_nuclei(cdf, colocal_id, clc_nucs_info, valid_labels, group_labels_col)
            if (clc_nucs_info is not None) and (colocal_id in clc_nucs_info):
                # this was reworked to suport multiple intersecting labels
                # get col name with intersecting label and intersecting colocal id
                clc_info = clc_nucs_info[colocal_id]
                for intersecting_label_column, intersecting_colocal_id in zip(clc_info["intersecting_label_columns"], clc_info["intersecting_colocal_ids"]):
                    # intersecting_label_column, intersecting_colocal_id = \
                    #     clc_nucs_info[colocal_id]['intersecting_label_column'], clc_nucs_info[colocal_id]['intersecting_colocal_id']
                    drp_prev_col = \
                        intersecting_label_column if not 'grouped_col' in clc_nucs_info[colocal_id] else clc_nucs_info[colocal_id]['grouped_col']
                
                    # remove  nuclei that are colocal with same object
                    cdf = drop_duplicate_colocalizations(cdf, intersecting_label_column, group_by_col=group_labels_col)
                    filtered_counts[colocal_id][f'post_drop_duplicates_ch{intersecting_colocal_id}'] = get_colocal_id_counts(cdf, [colocal_id])[colocal_id]
                    # remove any colocal nuclei where overlapping zif nuc was removed previously (e.g. area/axis length)
                    cdf = drop_previously_removed(cdf, drp_prev_col, valid_labels[intersecting_colocal_id])
                    filtered_counts[colocal_id][f'post_drop_previously_removed_ch{intersecting_colocal_id}'] = get_colocal_id_counts(cdf, [colocal_id])[colocal_id]

            valid_labels[colocal_id] = cdf[label_col].to_list()
            invalid_labels[colocal_id] = list(all_labels - set(valid_labels[colocal_id]))
            thresholded_rpdfs[colocal_id] = cdf

        thresholded_rpdf = pd.concat(thresholded_rpdfs.values())
        filtered_counts['final'] = get_colocal_id_counts(thresholded_rpdf, all_colocal_ids)
        # if return_labels:
        #     return thresholded_rpdf, filtered_counts, valid_labels, invalid_labels
        # return thresholded_rpdf, filtered_counts
        # print('~'*70)
        # print_fcounts(filtered_counts)

        count_tabulate = {k:v for k,v in filtered_counts.items() if k not in ['init','final']}
        df_idx = list(count_tabulate[list(count_tabulate.keys())[np.argmax(np.array([len(d) for d in count_tabulate.values()]))]].keys())
        count_f_df = pd.DataFrame.from_dict(count_tabulate).reindex(df_idx)
        print(tabulate(count_f_df, headers='keys'), flush=True)
















    # validation
    #####################
    """
        since here I am comparing to old counts where regions did not include all regions, some variation is expected
            especially in larger regions, but should be exact for highest st_levels
    """
    def format_df(df):
        df['reg_id'] = df['reg_id'].astype('int')
        df['region_areas'] = df['region_areas'].apply(lambda x: rp.pixel_to_um(x))
        return df
    def sum_over_imgnames(df, sum_over_cols=['nDapi', 'nZif', 'nGFP', 'nBoth', 'region_areas']):
        return df.groupby(['reg_id', 'region_sides'])[sum_over_cols].sum().reset_index()

    def merge_and_compare_dfs(df_test, df_gt, columns_to_compare):
        # clean up so both have same reg_id, but note which ones are different
        merged_df = pd.merge(df_test, df_gt, on=['reg_id', 'region_sides'], suffixes=('_test', '_gt'))

        # Create a DataFrame to hold the comparison results
        comparison_results = pd.DataFrame()
        for column in columns_to_compare:
            comparison_results[column + '_difference'] = merged_df[column + '_test'] - merged_df[column + '_gt']

        # Add 'reg_id' and 'region_sides' to the comparison results
        comparison_results['reg_id'] = merged_df['reg_id']
        comparison_results['region_sides'] = merged_df['region_sides']
        return merged_df, comparison_results

    # load validation df, get subset with this image
    all_data_df_path_GT =  r"D:\ReijmersLab\TEL\slides\quant_data\counts\2023_0827_151059_quant_data.csv"
    addf_GT = pd.read_csv(all_data_df_path_GT)
    addf_GT = addf_GT.rename(columns={'reg_side':'region_sides', 'reg_area':'region_areas', 'nBoth':'nGFP+Zif'})
    

    # propogate counts for test df
    from compile_data import get_nuclei_per_region_df
    self.ont = arhfs.Ontology()
    counts_df = get_nuclei_per_region_df(
        thresholded_rpdf, region_df, self.ont, datum.geojson_regions_dir_paths, 
        count_channel_names=ac.ImgDB.get_count_channel_names()
    )
    print('counts_df len:', len(counts_df), 'shape', counts_df.shape, flush=True)


    # format and merge to compare counts
    addf = format_df(counts_df)
    addf_GT = format_df(addf_GT[addf_GT['img_name']==img_name]) # this img df

    # reduce counts to total per region (combine images)
    sum_over_cols = ['nDapi', 'nZif', 'nGFP', 'nGFP+Zif', 'region_areas']
    df_test = sum_over_imgnames(addf, sum_over_cols=sum_over_cols).assign(condition='test')
    df_gt = sum_over_imgnames(addf_GT, sum_over_cols=sum_over_cols).assign(condition='GT')

    merged_df, comparison_results = merge_and_compare_dfs(
        df_test, df_gt, columns_to_compare=sum_over_cols
    )
    print(tabulate(comparison_results.describe(), headers='keys'))


    comparison_cols = ['nDapi_difference', 'nZif_difference', 'nGFP_difference', 'nGFP+Zif_difference']
    diffs_by_regid = {}
    threshold = 0
    for col in comparison_cols:
        threshed_diffs = comparison_results[abs(comparison_results[col])>threshold]
        print(col, len(threshed_diffs), f"sum of diffs: {threshed_diffs[col].sum()}")
        if len(threshed_diffs)>0:
            diffs_by_regid[col] = threshed_diffs.groupby('reg_id')[col].apply(lambda x: x.abs().sum()).to_dict()
    
    plt_df = []
    for col, vd in diffs_by_regid.items():
        for reg_id, count in vd.items():
            plt_df.append({
                'colocal_id':col, 'reg_id':reg_id, 'count':count
            })
    plt_df = pd.DataFrame(plt_df).assign(
        reg_name = lambda df: arhfs.get_attributes_for_list_of_ids(self.ont.ont_ids, df['reg_id'].values, 'name'),
        st_level = lambda df: arhfs.get_attributes_for_list_of_ids(self.ont.ont_ids, df['reg_id'].values, 'st_level')
    )

    import seaborn as sns
    fig,ax = plt.subplots(figsize=(18,8))
    bp = sns.barplot(data=plt_df, x='reg_name', y='count', hue='colocal_id', dodge=True, ax=ax)#, hue_order=[plt_df.colocal_id.unique()])
    sp = sns.swarmplot(data=plt_df, x='reg_name', y='count', hue='colocal_id', dodge=True, ax=ax)#, hue_order=[plt_df.colocal_id.unique()])
    ax.set_xticks(ax.get_xticks(), [xt.get_text() for xt in ax.get_xticklabels()], rotation=-45, ha='left')
    ax.set_yscale('symlog')
    sp.legend(bbox_to_anchor=(1.2, 1.0))
    for x in [1,10, 100, 100]:
        ax.axhline(x, c='gray', ls='--')
    plt.show()




    
        