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
        threshold_dict, rpdf_coloc, all_colocal_ids,
        group_labels_col='img_name', return_labels=False, 
        clc_nucs_info=None
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
            cdf, rp_filtercounts = region_prop_table_filter(cdf, threshold_dict[colocal_id], colocal_id) 
            for k,v in rp_filtercounts.items(): filtered_counts[colocal_id][k] = v
        
        # cdf = clean_up_colocal_nuclei(cdf, colocal_id, clc_nucs_info, valid_labels, group_labels_col)
        if (clc_nucs_info is not None) and (colocal_id in clc_nucs_info):
            # this was reworked to suport multiple intersecting labels
            # get col name with intersecting label and intersecting colocal id
            clc_info = clc_nucs_info[colocal_id]
            for intersecting_label_column, intersecting_colocal_id in zip(clc_info["intersecting_label_columns"], clc_info["intersecting_colocal_ids"]):
                drp_prev_col = \
                    intersecting_label_column if not 'grouped_col' in clc_info else clc_info['grouped_col']
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

def pretty_print_fcounts(filtered_counts):
    """ print counts after each filtering step as a str formatted table """
    count_tabulate = {k:v for k,v in filtered_counts.items() if k not in ['init','final']}
    df_idx = list(count_tabulate[list(count_tabulate.keys())[np.argmax(np.array([len(d) for d in count_tabulate.values()]))]].keys())
    count_f_df = pd.DataFrame.from_dict(count_tabulate).reindex(df_idx)
    print(tabulate(count_f_df.fillna(''), headers='keys'), flush=True)
    

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
    """ assign col to df to track labels by img_name """
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
    """ all_colocal_ids is a sorted list of every colocal id found in ImgDB """
    vc = rpdf.value_counts('colocal_id').to_dict()
    return {i:0 if i not in vc else vc[i] for i in all_colocal_ids}


def get_nuclei_per_region_df(rpdf, region_df, ont, geojson_path, count_channel_names=None):
    """
    Params:
        count_channel_names [list of str] - aliases to assign to colocal ids (col names for aggregated counts)
    """
    geojson_objs = rp.load_geojson_objects(geojson_path)
    regionPoly_list = rp.extract_polyregions(geojson_objs, ont) 
    
    propogate_counts_t0 = dt()
    region_count_df = pool_centroids_by_region(rpdf, count_channel_names=count_channel_names)
    region_count_df = propogate_region_counts(regionPoly_list, region_count_df, region_df, ont, count_channel_names)
    toadd = dict(zip(['region_name', 'acronym', 'st_level'], ['name', 'acronym', 'st_level']))
    region_count_df = pd.merge(region_count_df, region_df, left_index=True, right_index=True)
    region_count_df = region_count_df.rename(columns={'region_ids':'reg_id'})

    for col, ont_attr in toadd.items():
        region_count_df[col] = arhfs.get_attributes_for_list_of_ids(ont.ont_ids, region_count_df['reg_id'].to_list(), ont_attr)
    cpropogate_counts_t1 = dt()
    return region_count_df 

def pool_centroids_by_region(
        rpdf, 
        count_channel_names=['nDapi', 'nZif', 'nGFP', 'nBoth'], 
        info_cols={'reg_id': int, 'st_level': int, 'region_name': str, 'reg_side': str, 'acronym': str, 'region_area': float}
    ):
    """ aggregate centroids by poly_index to generate counts of nuclei by colocal_id per region
        # Group by 'region_id' and 'colocal_id', and count the occurrences
        # Pivot the table to get 'colocal_id' as separate columns for each 'region_id'
    """
    pivot_table = (rpdf
        .groupby(['poly_index', 'colocal_id'])
        .size()
        .reset_index(name='count')
        .pivot_table(index='poly_index', columns='colocal_id', values='count', fill_value=0)
        .assign(poly_index=lambda df: df.index.values)
        .reset_index(drop=True)
        .rename(columns=dict(zip(range(len(count_channel_names)), count_channel_names))) # rename colocal id cols to name of stain
        .assign(**{col: pd.Series(dtype=dt) for col, dt in info_cols.items()})
    )
    for row_i, row in pivot_table.iterrows(): # add info
        pivot_table.loc[row_i, info_cols.keys()] = rpdf[rpdf['poly_index']==row['poly_index']][info_cols.keys()].values[0]

    return pivot_table



def propogate_region_counts(regionPoly_list, pivot_table, region_df, ont, count_channel_names):
    # for compiling need to explode the region heirarchy based on the assigned lowest level region ids so all parent regions are quantified
    all_region_counts = {}
    all_regPoly_reg_ids = [p.obj_id for p in regionPoly_list]
    
    for rowi, row in pivot_table.iterrows():
        base_reg_id, base_side, base_poly_index = row['reg_id'], row['reg_side'], int(row['poly_index'])
        base_counts = lambda: row[count_channel_names].to_dict() # need to create this every time or it overinflates the values 
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

def get_parent_poly_ids(region_df, parent_ids, side):
    # map region ids to poly_index
    out = []
    for pid in parent_ids:
        reg_df_row = region_df[(region_df['region_ids']==pid) & (region_df['region_sides']==side)]
        assert len(reg_df_row) == 1
        out.append(int(reg_df_row['poly_index'].values[0]))
    return out

def add_to_count_dict(fpi, all_region_counts, base_counts):
    if fpi not in all_region_counts:
        all_region_counts[fpi] = base_counts()
    else:
        bc = base_counts()
        for k,v in all_region_counts[fpi].items():
            all_region_counts[fpi][k] = v + bc[k]
    return all_region_counts  


def inspect_output(all_data_df, inspect_region_id=295, 
                   groupbyCols = ['reg_id', 'region_name', 'st_level', 'sex', 'strain', 'group', 'cohort', 'animal_id'], 
                   df_calcs = {'reactivation': lambda df: df['nGFP+Zif']/df['nGFP'], 'groupcohort': lambda df: df['group']+df['cohort']},
                   sig_diff_groups=['FC', 'FC+EXT'], sig_diff_col = 'reactivation', 
                   summary_df_cols = ['groupcohort', 'animal_id', 'nGFP', 'nZif', 'nGFP+Zif', 'reactivation', 'nGFP+Zif+Dapi'],
    ):
    # get nan counts 
    dfCols = all_data_df.columns # all_data_df.select_dtypes('object')
    nan_count_dict = dict(zip(dfCols,[len(all_data_df[dfCols].query(f'`{c}`.isna()')) for c in dfCols]))
    print(f"nan values found in {sum(np.array(list(nan_count_dict.values()))>0)} columns")
    for k,v in nan_count_dict.items():
        if v > 0: print(f"\tcol: {k}, nan count: {v}")
    
    # get counts for a specific region
    andf = all_data_df.groupby(groupbyCols,as_index=False).sum().sort_values('animal_id')
    areg_df = andf[andf['reg_id']==inspect_region_id].sort_values(['animal_id','st_level'])

    # perform any calculations on summarized data
    plotdf = areg_df.assign(**df_calcs)
    pval = scipy.stats.f_oneway(*[plotdf[plotdf['group']==hv][sig_diff_col].values for hv in sig_diff_groups]).pvalue

    dataByAn = plotdf.sort_values(['group', sig_diff_col])[summary_df_cols]
    print(tabulate(dataByAn, headers='keys', tablefmt='heavy_outline'))

    print(plotdf.groupby('group').mean(numeric_only=True).to_dict()) 
    print(f"p-value: {pval}")
    print()

    # fig,ax = plt.subplots()
    # sns.barplot(data=plotdf, x='region_name', y='reactivation', hue='group', dodge=True, ax=ax, edgecolor=".5", facecolor=(0, 0, 0, 0), linewidth=3)
    # sns.swarmplot(data=plotdf, x='region_name', y='reactivation', hue='group', dodge=True, ax=ax)
    # ax.set_title(f"p-value: {pval}")
    # handles, labels = [[lhl[0]] + [lhl[1]] for lhl in ax.get_legend_handles_labels()]
    # ax.legend(handles, labels, bbox_to_anchor=(1.25, 1.05))
    # plt.show()

def verify_threshold_params(threshold_dict, rpdf):
    ignore_missing_regex = 'ch\d+_intensity'
    errors = {}
    for clc_id, prop_dict in threshold_dict.items():
        for prop, values in prop_dict.items():
            if prop not in rpdf.columns:
                if re.match(ignore_missing_regex, prop): # ignore errors for these cols that are calculated while thresholding
                    continue
                if clc_id not in errors:
                    errors[clc_id] = []
                errors[clc_id].append(prop)
    if len(errors)>0:
        raise KeyError(print(f"threshold props not found in rpdf:\n\t{errors}\n\trpdf cols: {rpdf.columns.to_list()}"))
    return 0


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
            rpdf_filtered, fcs = filter_nuclei(self.threshold_dict, rpdf, self.all_colocal_ids, group_labels_col=None, return_labels=False, 
                clc_nucs_info=self.clc_nucs_info)
            pretty_print_fcounts(fcs)




            self.filter_counts[img_name] = fcs
            if self.TEST: self.rpdfs_filtered.append(rpdf_filtered)

            # get nuclei counts per region
            counts_df = get_nuclei_per_region_df(rpdf_filtered, region_df, ont, datum.geojson_regions_dir_paths, 
                                                 count_channel_names=self.count_channel_names)
            print('counts_df len:', len(counts_df), 'shape', counts_df.shape, flush=True)
            # counts_df = get_nuclei_per_region_df(rpdf_filtered, region_df, self.an.base_dir, datum.geojson_regions_dir_paths)

            # add animal info
            collected_dfs.append(self.map_animal_info(counts_df, animal_info_row, datum))
        
        animal_df = pd.concat(collected_dfs, ignore_index=True)
        # print(f'completed {self.an.animal_id} in {timeit.default_timer()-animal_st_time}.\n')
        return animal_df

    def rpdf_compile(self, outdir=r'D:\ReijmersLab\TEL\slides\quant_data\byAnimal_rpdfs\blaOnly_0816'):
        """ 
        only compile the rpdfs for each animal into one huge df
            region_ids_only (None, List) --> get for all region or just a subset
        
        """
        animal_st_time = timeit.default_timer()
        animal_info_row = self.aidf[self.aidf['animal_id']==self.an.animal_id]
        datums = self.an.get_valid_datums(['rpdf_paths', 'region_df_paths'])

        collected_dfs = []
        for datum in datums:
            img_name = Path(datum.fullsize_paths).stem[:-4]
            rpdf = pd.read_csv(datum.rpdf_paths)
            region_df = pd.read_csv(datum.region_df_paths)

            rpdf = rpdf.loc[rpdf['colocal_id'] > 0]

            # map poly_i in region_ids_list to atlas_id
            poly_i_map = dict(zip(region_df.index.values, region_df['region_ids'].values))
            region_ids_lists = [[int(poly_i_map[v]) for v in ast.literal_eval(val)] for val in rpdf['region_ids_list'].values]
            rpdf['region_ids_list'] = region_ids_lists

            extract_region_id = 295
            rpdf = rpdf.loc[rpdf.apply(lambda x: extract_region_id in x['region_ids_list'],axis=1)]
            
            
            # filter nuclei
            rpdf_filtered, filter_counts = filter_nuclei(self.threshold_dict, rpdf, group_labels_cols=None, return_labels=False, 
                clc_nucs_info={3:{'intersecting_label_column':'ch0_intersecting_label', 'intersecting_colocal_id':1}})

            # TODO add in parsing for aidf
            collected_dfs.append(rpdf_filtered
                .assign(
                    animal_id = self.an.animal_id,
                    img_name = img_name,
                    group = animal_info_row['treatment'].values[0],
                    sex =  animal_info_row['sex'].values[0],
                    strain =  animal_info_row['strain'].values[0],
            ))
        animal_rpdf = pd.concat(collected_dfs, ignore_index=True)
        THE_DATE = datetime.now().strftime('%y_%m%d')
        outdir_base = ug.verify_outputdir(outdir)
        rpdf_outpath = os.path.join(outdir_base, 
                                    f"{self.an.animal_id}_rpdf_t-{self.threshold_dict['zif_intensity_threshold']}_{THE_DATE}.csv")
        self.write_csv(rpdf_outpath, animal_rpdf)
        print(f'completed {self.an.animal_id} in {timeit.default_timer()-animal_st_time}.\n')
        return 0

    

def worker(disp):
    """worker function to run disp and return dataframe"""
    try:
        return disp.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def run_parallel_disps(disps, MODE='run'):
    """run disp.run() for each disp in disps in parallel and concatenate results"""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_dfs = list(executor.map(worker, disps))
    final_df = pd.concat(result_dfs, ignore_index=True)
    return final_df


def get_dispatchers(
        animals, count_channel_names, clc_nucs_info, all_colocal_ids,
        THRESHOLD_DICTS, outdir, THE_DATE, MODE, 
        animal_info_df_path=None, animal_info_df_map_cols=None, 
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
                all_colocal_ids = all_colocal_ids,
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

"""
#############################################################################################################################################
NOTES
~~~~~
    - intensity thresholds
        - cohort3 == 530
        - cohort2 == 894
            FC zif mean intensity: 894.6 --> keep: 3407503 / 8105278 (42.041 %)
            EXT zif mean intensity: 849.0 --> keep: 2065443 / 6080181 (33.97 %)

    - lowest pval so far
        - threshold_dict = dict(area_threshold = 700, intersectionP_threshold = 0.70, zif_intensity_threshold = 530)
    )
    previous thresholds:
    # 'cohort2':{
        #     'area_threshold_min': 75, 'area_threshold_max': 350,
        #     'intersectionP_threshold': 0.45,
        #     'zif_intensity_threshold': 550, 'gfp_intensity_threshold': 100,
        #     'axis_major_length_min': 5, 'axis_major_length_max': 85, 
        #     'axis_minor_length_min': 4, 'axis_minor_length_max': 31
        # },

        # 'cohort3':{
        #     'area_threshold_min': 75, 'area_threshold_max': 350, 
        #     'intersectionP_threshold': 0.45, 
        #     'zif_intensity_threshold': 360, 'gfp_intensity_threshold': 100, 
        #     'axis_major_length_min': 5, 'axis_major_length_max': 85, 
        #     'axis_minor_length_min': 4, 'axis_minor_length_max': 31
        # },

        # 'cohort4':{
        #     'area_threshold_min': 75, 'area_threshold_max': 350, 
        #     'intersectionP_threshold': 0.45, 
        #     'zif_intensity_threshold': 375, 'gfp_intensity_threshold': 160, 
        #     'axis_major_length_min': 12, 'axis_major_length_max': 85, 
        #     'axis_minor_length_min': 7, 'axis_minor_length_max': 31
        # },
#############################################################################################################################################
"""

#############################################################################################################################################
# conda
# conda activate stardist
# python "C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\compile_data_2023_0602.py"
#############################################################################################################################################

# def get_threshold_dicts():
#     # set thresholds
#     THRESHOLD_DICTS = {
        
#         'cohort2': {
#             0:{"intensity_mean": (1, None), "area": (50, 1000), "axis_major_length": (8,85), "axis_minor_length": (7,31), "eccentricity":(None, 2.9)},
#             1:{"intensity_mean": (50, None), "area": (50, 1000), "axis_major_length": (7,85), "axis_minor_length": (7,31), "eccentricity":(None, 2.9)}, 
#             2:{"intensity_mean": (20, None), "area": (130, 1000), "axis_major_length": (12,85), "axis_minor_length": (8,31), "eccentricity":(None, 2.9)},
#             3:{"intensity_mean": (20, None), "zif_intensity":(75, None), "intersection_p":(0.33, None), "area": (130, 850), "axis_major_length": (12,85), "axis_minor_length": (8,31), "eccentricity":(None, 2.9)},
#         },
#         'cohort3': {
#             0:{"intensity_mean": (1, None), "area": (50, 1000), "axis_major_length": (8,85), "axis_minor_length": (7,31), "eccentricity":(None, 2.9)},
#             1:{"intensity_mean": (85, None), "area": (50, 1000), "axis_major_length": (7,85), "axis_minor_length": (7,31), "eccentricity":(None, 2.9)}, 
#             2:{"intensity_mean": (45, None), "area": (130, 1000), "axis_major_length": (12,85), "axis_minor_length": (8,31), "eccentricity":(None, 2.9)},
#             3:{"intensity_mean": (45, None), "zif_intensity":(85, None), "intersection_p":(0.33, None), "area": (130, 850), "axis_major_length": (12,85), "axis_minor_length": (8,31), "eccentricity":(None, 2.9)},
#         },
#         'cohort4': { # may need to increase zif thresh abit
#             0:{"intensity_mean": (1, None), "area": (50, 1000), "axis_major_length": (8,85), "axis_minor_length": (7,31), "eccentricity":(None, 2.9)},
#             1:{"intensity_mean": (100, None), "area": (50, 1000), "axis_major_length": (7,85), "axis_minor_length": (7,31), "eccentricity":(None, 2.9)}, 
#             2:{"intensity_mean": (25, None), "area": (130, 1000), "axis_major_length": (12,85), "axis_minor_length": (8,31), "eccentricity":(None, 2.9)},
#             3:{"intensity_mean": (25, None), "zif_intensity":(100, None), "intersection_p":(0.33, None), "area": (130, 850), "axis_major_length": (12,85), "axis_minor_length": (8,31), "eccentricity":(None, 2.9)},
#         },
#     }
            
#     return THRESHOLD_DICTS

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
    OUTDIR = r'D:\ReijmersLab\TEL\slides\quant_data\counts_test_newpipeline_2024_0121'
    
    # ARGUMENTS
    #######################################################
    TEST = bool(0)
    MULTIPROCESS = bool(0)
    MODE = ['reduce_by_region', 'compile_animal_rpdfs'][0]
    THE_DATE = datetime.now().strftime('%Y_%m%d_%H%M%S')    
    GET_ANIMALS = ['cohort2', 'cohort3', 'cohort4']
    animals_upto = 1

    # MAIN
    ##################################################################################################
    start_time = timeit.default_timer()
    ac = AnimalsContainer()
    ac.init_animals()
    animals = ac.get_animals(GET_ANIMALS)[:animals_upto if not TEST else 1]
    THRESHOLD_DICTS = ac.ImgDB.get_threshold_params()
    COUNT_CHANNEL_NAMES = ac.ImgDB.get_count_channel_names()
    CLC_NUC_INFO = ac.ImgDB.get_clc_nuc_info()
    COLOCALIZATION_PARAMS = ac.ImgDB.colocalizations
    ALL_COLOCAL_IDS = sorted(list(ac.ImgDB.colocal_ids.keys()))

    # animals = ac.get_animals(['TEL17', 'TEL24', 'TEL48', 'TEL44', 'TEL53', 'TEL55'])
    dispatchers = get_dispatchers(
        animals, COUNT_CHANNEL_NAMES, CLC_NUC_INFO, ALL_COLOCAL_IDS,
        THRESHOLD_DICTS, OUTDIR, THE_DATE, MODE, 
        animal_info_df_path=animal_info_df_path, animal_info_df_map_cols=animal_info_df_map_cols, 
        colocalization_params=COLOCALIZATION_PARAMS, TEST=TEST,
    )

    if TEST:
        result = dispatchers[0].run()

    elif MODE == 'reduce_by_region':  # normal
        # aggregate nuclei counts per region/side per animal, reduces regionprop dfs (all nuclei) into one df containing all animals
        all_data_df = run_parallel_disps(dispatchers) if MULTIPROCESS else pd.concat([disp.run() for disp in dispatchers], ignore_index=True)
        # save the data
        all_data_df.to_csv(os.path.join(OUTDIR, f"{THE_DATE}_quant_data.csv"), index=False)
        inspect_output(all_data_df, inspect_region_id=295)   
        all_data_df_path_GT =  r"D:\ReijmersLab\TEL\slides\quant_data\counts\2023_0827_151059_quant_data.csv"
        addf_GT = pd.read_csv(all_data_df_path_GT)
        addf_GT = addf_GT.rename(columns={'reg_side':'region_sides', 'reg_area':'region_areas', 'nBoth':'nGFP+Zif'})
        addf_GT[(addf_GT['animal_id']=='TEL15') & (addf_GT['reg_id']==295)]['nGFP+Zif'].sum()
        all_data_df[(all_data_df['animal_id']=='TEL15') & (all_data_df['reg_id']==295)]['nGFP+Zif'].sum()
    
    
    elif MODE == 'compile_animal_rpdfs': # if do not want to aggregate nuclei
        # compile all img rpdfs into single large df with one row for each nuclei in all animals
        for disp in dispatchers:
            disp.rpdf_compile()

    else:
        raise ValueError(f"Mode '{MODE}' not recognized, but be one of ['reduce_by_region', 'compile_animal_rpdfs']")

    print('completed in:', timeit.default_timer()-start_time)

    





    


    


        

        




    



