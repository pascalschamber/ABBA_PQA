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
import tifffile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
from utilities.utils_image_processing import read_img, print_array_info, convert_16bit_image
import utilities.utils_general as ug
import core_regionPoly as rp
import utilities.utils_plotting as up
import compile_data as compile 
import compile_data_2023_0602 as compile_old

# ac = AnimalsContainer()
# ac.init_animals()
# tel15 = ac.get_animals('TEL15')
# this_datum = tel15.d[26]
# img, _ = read_img(this_datum.fullsize_paths)
# nuc_img = tifffile.imread(this_datum.quant_dir_paths) # 1652,4400 |2100,4800 == [2154:2923, 6923:7846], np.array(np.rint((np.array([1400, 4500])/0.650)),dtype='int16')
# qupath_coords = np.array(np.rint(np.array([[1652,4400][::-1], [2100,4800][::-1]])/0.650), dtype='int16')
# crop_fs = img[6769:7385, 2542:3231, 0]
# crop_nuc = nuc_img[6769:7385, 2542:3231, 0]

# up.show(up.overlay(crop_fs, mask=crop_nuc, image_cmap='red'))
# # above shows that the issue is that it is not being filtered.

#############################################################################################################################################
# also load rpdfs for both to see if counts are a product of compiling error or true values
# rpdf_test = pd.read_csv(ac.get_animals('TEL15').d[26].rpdf_paths)
# rpdf = rpdf_test.assign(eccentricity = rpdf_test['axis_major_length']/rpdf_test['axis_minor_length'])
# rpdf['zif_intensity'] = rpdf['ch0_intersecting_label'].map(
#     rpdf.loc[rpdf['colocal_id'] == 1].set_index('label')['intensity_mean'])
# rpdf_filtered, fcs = compile.filter_nuclei(compile.get_threshold_dicts()['cohort2'], rpdf, group_labels_col=None, return_labels=False)
# rpdf_filtered[(rpdf_filtered['reg_side']=='Right') & (rpdf_filtered['reg_id']==30002) & (rpdf_filtered['colocal_id']==1)]


#############################################################################################################################################
def format_df(df):
    df['reg_id'] = df['reg_id'].astype('int')
    df['region_areas'] = df['region_areas'].apply(lambda x: rp.pixel_to_um(x))
    return df
def sum_over_imgnames(df):
    return df.groupby(['reg_id', 'region_sides'])['nDapi', 'nZif', 'nGFP', 'nBoth', 'region_areas'].sum().reset_index()

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

# df to test
all_data_df_path =  r"D:\ReijmersLab\TEL\slides\quant_data\counts_test_newpipeline_2024_0121\2024_0122_232800_quant_data.csv"
addf = pd.read_csv(all_data_df_path)

# df to compare to
all_data_df_path_GT =  r"D:\ReijmersLab\TEL\slides\quant_data\counts\2023_0827_151059_quant_data.csv"
addf_GT = pd.read_csv(all_data_df_path_GT)
addf_GT = addf_GT.rename(columns={'reg_side':'region_sides', 'reg_area':'region_areas'})

# filter so same animals
addf = format_df(addf[addf['animal_id']=='TEL15'])
addf_GT = format_df(addf_GT[addf_GT['animal_id']=='TEL15'])

addf[(addf['img_name']=='TEL15_s026_2-4') & (addf['reg_id']==30002)]
addf_GT[(addf_GT['img_name']=='TEL15_s026_2-4') & (addf_GT['reg_id']==30002)]
# rpdf_test[(rpdf_test['reg_side']=='Right') & (rpdf_test['reg_id']==30002) & (rpdf_test['colocal_id']==1)]

# reduce counts to total per region (combine images)
df_test = sum_over_imgnames(addf).assign(condition='test')
df_gt = sum_over_imgnames(addf_GT).assign(condition='GT')

merged_df, comparison_results = merge_and_compare_dfs(df_test, df_gt, columns_to_compare=['nDapi', 'nZif', 'nGFP', 'nBoth', 'region_areas'])
comparison_results.describe()

comparison_cols = ['nDapi_difference', 'nZif_difference', 'nGFP_difference', 'nBoth_difference',]
threshold = 1000
for col in comparison_cols:
    print(col, len(comparison_results[comparison_results[col]>threshold]))

#############################################################################################################################################
# 1st step should be to add region area to the comparisons
    # if that is substantially large then these differences make sense.
# TODO was thresholding actually performed on the test set?
# how do the counts compare before thresholding


#############################################################################################################################################
import seaborn as sns
for col in comparison_cols:

    fig,ax = plt.subplots()
    pltdf = comparison_results.assign(abs_diff=comparison_results[col].abs()).sort_values('abs_diff', ascending=False)


import matplotlib.pyplot as plt
import pandas as pd

# Assuming comparison_results is your DataFrame from the previous step
# (Add your actual comparison results DataFrame here)

# Check for missing regions
# Assuming df_test and df_gt are your original DataFrames
missing_in_test = set(df_gt[['reg_id', 'region_sides']].itertuples(index=False)) - set(df_test[['reg_id', 'region_sides']].itertuples(index=False))
missing_in_gt = set(df_test[['reg_id', 'region_sides']].itertuples(index=False)) - set(df_gt[['reg_id', 'region_sides']].itertuples(index=False))

if missing_in_test:
    print("Regions missing in test:", missing_in_test)
if missing_in_gt:
    print("Regions missing in GT:", missing_in_gt)

# Plot the largest differences
# Calculate the maximum difference across all count columns
comparison_results['max_difference'] = comparison_results[[col for col in comparison_results.columns if '_difference' in col]].abs().max(axis=1)

# Sort the results by the maximum difference
sorted_results = comparison_results.sort_values(by='max_difference', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
for column in ['nDapi_difference', 'nZif_difference', 'nGFP_difference', 'nBoth_difference']:
    plt.plot(sorted_results['region_sides'], sorted_results[column], label=column)

plt.xlabel('Region Side')
plt.ylabel('Count Difference')
plt.title('Comparison of Counts Between Test and GT')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()




