import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import seaborn as sns
import os
import skimage
from tifffile import imread
from pathlib import Path
import scipy
from datetime import datetime
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
import utilities.utils_plotting as up
import utilities.utils_atlas_region_helper_functions as arhfs
from utilities.utils_data_management import AnimalsContainer


today = datetime.now().strftime('%Y_%m%d')


def normalize_01(image):
    return (image-np.min(image))/(np.max(image)-np.min(image)) 
def normalize_minus1_1(image):
    return 2 * normalize_01(image) - 1

def get_st_level_df(fdf, st_level, groupby, bycohort=False):
    # change the full df to take the sum for each animal
    stdf = fdf.copy(deep=True)
    
    if st_level is not None: # optionally filter by st_level
        stdf = stdf.loc[stdf['st_level'] == st_level]

    stdf = add_percentages_to_df(stdf)

    # create a df that takes the mean along each group and structure
    index_cols = ['reg_id', 'region_name', groupby]
    if bycohort: index_cols.append('cohort')
    gdf = stdf.groupby(index_cols).mean(numeric_only=True).reset_index().sort_values(['region_name', groupby])
    gdf = add_percentages_to_df(gdf)

    return stdf, gdf

def add_percentages_to_df(gdf):
    gdf = gdf.copy(deep=True)
    gdf['nSum'] = gdf['nDapi'] + gdf['nZif'] + gdf['nGFP']

    gdf['pDapi'] = gdf['nDapi'].div(gdf['nSum'])
    gdf['pZif'] = gdf['nZif'].div(gdf['nSum'])
    gdf['pGFP'] = gdf['nGFP'].div(gdf['nSum'])
    gdf['pBoth'] = gdf['nBoth'].div(gdf['nSum'])
    gdf['pZifInGfp'] = gdf['nBoth'].div(gdf['nGFP'])
    gdf['pReactivationZif'] = gdf['nBoth'].div(gdf['nZif'])
    return gdf

def get_yvar_stats(st_level, gdf, yvar, region_name, round_to=6, p=True):
    ''' get the values of a variable and compared it between fc and ext groups '''
    region_df = gdf[gdf[st_level] == region_name][[st_level, 'group', yvar]]
    yvar_row_fc, yvar_row_ext = region_df.values
    yvar_fc, yvar_ext = yvar_row_fc[-1], yvar_row_ext[-1]
    if p:
        print(f'{yvar} fc/ext ({round(yvar_fc,round_to)}, {round(yvar_ext,round_to)}) ratio: {round(yvar_fc/yvar_ext,round_to)}')
    return region_df, yvar_fc, yvar_ext

def align_regionnames_for_dfGroups(df, groupby):
    # get only regions that appear in all groups
    unique_regions_per_group = df.groupby(groupby)['region_name'].apply(set).to_list()
    intersection = list(set.intersection(*unique_regions_per_group))
    df = df[df['region_name'].isin(intersection)] 
    return df

def get_region_differences_by_group(df, yvar, groupby, groups, abs_val=True):
    ''' get the by group df to have regions that match so comparison can occur '''
    # df = by_group_df_sums
    # abs_val = True

    abs_func = abs if abs_val else no_abs

    # cleaned regions so all regions match
    df = df.dropna(how='any') # prepare to drop regions if one group doesn't have data
    df = align_regionnames_for_dfGroups(df, groupby)

    # get the df with means by group for all regions at this st level    
    gdf_piv = df.pivot(index = ['reg_id', 'region_name'], columns=groupby, values=yvar).reset_index()
    gdf_piv = gdf_piv.dropna(how='any')
    gdf_piv=gdf_piv.replace(0.0, 0.001) # replace 0 values with low number that doesn't make it explode
    df = gdf_piv.assign(
        diff = lambda x: x[groups[0]] - x[groups[1]]
    )
    df = df.assign(abs_diff = df['diff'].abs())
    df = df.dropna(how='any').sort_values('abs_diff',ascending=False)

    # get the absolute ratio, e.g. if ext was higher ratio would be less than 1
    df = df.assign(ratio = lambda x: x[groups[0]] / x[groups[1]])
    diffs_gthan1 = df.loc[df['ratio']>=1.0]
    diffs_lthan1 = df.loc[df['ratio']<1.0]
    diffs_gthan1 = diffs_gthan1.assign(ratio_norm = normalize_01(diffs_gthan1['ratio'].values))
    diffs_lthan1 = diffs_lthan1.assign(ratio_norm = normalize_01(1/diffs_lthan1['ratio'].values)*-1)
    df = pd.concat([diffs_gthan1,diffs_lthan1], ignore_index=True)
    df = df.assign(ratio_norm_abs = df['ratio_norm'].abs())
    # doesn't work correctly, would need to think about #TODO if valuable
    # df = df.assign(diffs_log = np.log10(df['ratio'].values))
    # df = df.assign(diffs_log_normed = normalize_minus1_1(df['diffs_log'].values))

    return df

def get_rdbg_pvals(rdbg, by_animal_df_sums, yvar, groups):
    # oneway anova for signficance
    reg_pvals = []
    for areg in rdbg['region_name'].values:
        regdf = by_animal_df_sums[by_animal_df_sums['region_name'] == areg]
        regvals = [regdf[regdf['group'] == grp][yvar].values for grp in groups]
        pval = scipy.stats.f_oneway(*regvals).pvalue
        reg_pvals.append(pval)
    rdbg['p_value'] = reg_pvals
    return rdbg


def test_cohort_differences(andf, trend_thresh=0.01):
    # compare the difference between brain regions for each group and cohort 

    # 07/14/23 results
    # 30/731 regions differ in trend by more than 0.20, # 4%
    # 72/731 regions differ in trend by more than 0.10, # 10%
    # 133/731 regions differ in trend by more than 0.05, # 18%
    # 222/731 regions differ in trend by more than 0.01, # 30%

    # apply region differences by cohort
    by_animal_df_sums, by_group_df_sums = get_st_level_df(andf, None, 'group', bycohort=True)
    rdbg2 = get_region_differences_by_group(by_group_df_sums.loc[by_group_df_sums['cohort'] =='cohort2'], 'pZifInGfp', 'group', ['FC', 'FC+EXT'],  abs_val=True).reset_index()
    rdbg3 = get_region_differences_by_group(by_group_df_sums.loc[by_group_df_sums['cohort'] =='cohort3'], 'pZifInGfp', 'group', ['FC', 'FC+EXT'], abs_val=True).reset_index()
    rdbg = pd.concat([
        rdbg2.assign(cohort = 'cohort2'), 
        rdbg3.assign(cohort = 'cohort3')], ignore_index=True)
    rdbg = align_regionnames_for_dfGroups(rdbg, 'cohort').drop(columns=['index']).sort_values('region_name') 

    # remove regions we dont care about
    drop_st_names = ['tract', 'nerve', 'ventricle', 'bundle']
    for drpname in drop_st_names:
        rdbg = rdbg.loc[~rdbg['region_name'].str.contains(drpname)]

    # iterate through regions counting instances where cohort trends diverge by specified threshold
    divergers = {}
    for areg in rdbg['region_name'].unique():
        rows = rdbg[rdbg['region_name'] == areg]

        # get diffs for each cohort and compare them
        fc2, fc3 = rows[rows['cohort'] == 'cohort2']['FC'].values[0], rows[rows['cohort'] == 'cohort3']['FC'].values[0]
        ext2, ext3 = rows[rows['cohort'] == 'cohort2']['FC+EXT'].values[0], rows[rows['cohort'] == 'cohort3']['FC+EXT'].values[0]

        diff2, diff3 = rows[rows['cohort'] == 'cohort2']['diff'].values[0], rows[rows['cohort'] == 'cohort3']['diff'].values[0]

        if diff2>0 and diff3>0: # all is good
            continue
        elif diff2<0 and diff3<0: # all is good
            continue
        else:
            diffdiff = abs(abs(diff2) - abs(diff3))
            if diffdiff > trend_thresh:
                divergers[areg] = diffdiff#[fc2, fc3, ext2, ext3]

    print('num diveregers:', len(divergers), len(rdbg)/2)
    return divergers



    




def get_top_n_diffs(df, diff_var, top_n=50, drop_no_sig_diff_thresh=False):
    ''' get the top values of a column, apply a threshold to only return those above a thresh if desired '''
    if drop_no_sig_diff_thresh:
        df = df[df[diff_var] > drop_no_sig_diff_thresh]
    df = df.sort_values(diff_var, ascending=False)
    top_n = top_n if top_n < len(df) else len(df)
    return df[:top_n]

def no_abs(x):
    return x

# def get_st_names_contain_str(df, regions_to_find, region_name_col):
#     ''' # get subregions of a specific structure '''
#     cols_to_search = set(st_levels).intersection(set(df.columns))
#     regions_found = []
#     for region_to_find in regions_to_find:
#         mask = np.column_stack([df[col].str.contains(region_to_find, na=False) for col in cols_to_search])
#         region_df = df.loc[mask.any(axis=1)]
#         for el in list(region_df[region_name_col].unique()):
#             regions_found.append(el)
#     return list(set(regions_found))


def apply_heat_map_to_array(atlas_slice, top_vals):
    # convert region_name to atlas id
    region_ids = region_names_to_atlas_ids(top_vals.index)
    region_vals = top_vals.values
    return map_heatmap(atlas_slice, region_ids, region_vals)

# @jit(nopython=True)
def map_heatmap(atlas_slice, region_ids, region_vals):
    atlas_region_heatmap = np.zeros_like(atlas_slice).astype('float64')
    for i in np.arange(region_ids.shape[0]):
        atlas_region_heatmap = np.where(atlas_slice==region_ids[i], region_vals[i], atlas_region_heatmap)
    return atlas_region_heatmap


def region_names_to_atlas_ids(list_of_names):
    reg_id_map=pd.read_excel(r'D:\ReijmersLab\TEL\slides\onts_id_reg.xlsx')
    return np.array([reg_id_map[reg_id_map['name'] == name]['id'].values[0] for name in list_of_names]).astype('int16')

def get_regions_of_interest_by_st_level():
    return({
        'st4':[
            'Abducens nucleus', 'Accessory facial motor nucleus', 'Accessory olfactory bulb', 'Accessory supraoptic group', 'Agranular insular area', 'Anterior amygdalar area', 'Anterior cingulate area', 'Anterior hypothalamic nucleus', 'Anterior olfactory nucleus', 'Anterior pretectal nucleus', 'Anterior tegmental nucleus', 'Anterior group of the dorsal thalamus', 'Anterodorsal preoptic nucleus', 'Visual areas', 'Anteroventral periventricular nucleus', 'Anteroventral preoptic nucleus', 'Arcuate hypothalamic nucleus', "Barrington's nucleus", 'Basolateral amygdalar nucleus, anterior part', 'Basolateral amygdalar nucleus, posterior part', 'Basolateral amygdalar nucleus, ventral part', 'Basomedial amygdalar nucleus, anterior part', 'Basomedial amygdalar nucleus, posterior part', 'Bed nuclei of the stria terminalis', 'Bed nucleus of the accessory olfactory tract', 'Bed nucleus of the anterior commissure', 'Caudoputamen', 'Central amygdalar nucleus', 'Intralaminar nuclei of the dorsal thalamus', 'Central linear nucleus raphe', 'Cerebellum', 'Claustrum', 'Copula pyramidis', 'Cortical amygdalar area', 'Cortical subplate', 'Crus 1', 'Crus 2', 'Cuneiform nucleus', 'Hippocampal region', 'Dentate nucleus', 'Medial septal complex', 'Auditory areas', 'Cochlear nuclei', 'Dorsal motor nucleus of the vagus nerve', 'Dorsal nucleus raphe', 'Dorsal peduncular area', 'Dorsal premammillary nucleus', 'Dorsal tegmental nucleus', 'Dorsal terminal nucleus of the accessory optic tract', 'Dorsomedial nucleus of the hypothalamus', 'Ectorhinal area', 'Edinger-Westphal nucleus', 'Endopiriform nucleus, dorsal part', 'Endopiriform nucleus, ventral part', 'Retrohippocampal region', 'Facial motor nucleus', 'Fastigial nucleus', 'Zona incerta', 'Flocculus', 'Frontal pole, cerebral cortex', 'Fundus of striatum', 'Gigantocellular reticular nucleus', 'Globus pallidus, external segment', 'Globus pallidus, internal segment', 'Gustatory areas', 'Hippocampal formation', 'Hypothalamus', 'Inferior colliculus, central nucleus', 'Inferior colliculus, dorsal nucleus', 'Inferior colliculus, external nucleus', 'Inferior olivary complex', 'Inferior salivatory nucleus', 'Infracerebellar nucleus', 'Infralimbic area', 'Intercalated amygdalar nucleus', 'Interfascicular nucleus raphe', 'Geniculate group, ventral thalamus', 'Intermediate reticular nucleus', 'Medial group of the dorsal thalamus', 'Interpeduncular nucleus', 'Interposed nucleus', 'Interstitial nucleus of Cajal', 'Parabrachial nucleus', 'Lateral amygdalar nucleus', 'Epithalamus', 'Lateral hypothalamic area', 'Mammillary body', 'Lateral group of the dorsal thalamus', 'Lateral preoptic area', 'Lateral reticular nucleus', 'Lateral septal nucleus', 'Lateral terminal nucleus of the accessory optic tract', 'Vestibular nuclei', 'Laterodorsal tegmental nucleus', 'Linear nucleus of the medulla', 'Lingula (I)', 'Lobule II', 'Lobule III', 'Lobules IV-V', 'Locus ceruleus', 'Magnocellular nucleus', 'Magnocellular reticular nucleus', 'Main olfactory bulb', 'Medial amygdalar nucleus', 'Geniculate group, dorsal thalamus', 'Medial preoptic area', 'Medial preoptic nucleus', 'Medial pretectal area', 'Medial terminal nucleus of the accessory optic tract', 'Median eminence', 'Median preoptic nucleus', 'Medulla', 'Midbrain', 'Midbrain reticular nucleus', 'Midbrain reticular nucleus, retrorubral area', 'Midbrain trigeminal nucleus', 'Motor nucleus of trigeminal', 'Nodulus (X)', 'Nucleus accumbens', 'Nucleus ambiguus', 'Nucleus incertus', 'Nucleus of Darkschewitsch', 'Midline group of the dorsal thalamus', 'Nucleus of the brachium of the inferior colliculus', 'Nucleus of the lateral lemniscus', 'Nucleus of the lateral olfactory tract', 'Nucleus of the optic tract', 'Nucleus of the posterior commissure', 'Nucleus of the solitary tract', 'Nucleus of the trapezoid body', 'Perihypoglossal nuclei', 'Nucleus raphe magnus', 'Nucleus raphe obscurus', 'Nucleus raphe pallidus', 'Nucleus raphe pontis', 'Nucleus sagulum', 'Nucleus x', 'Nucleus y', 'Oculomotor nucleus', 'Olfactory areas', 'Olfactory tubercle', 'Olivary pretectal nucleus', 'Orbital area', 'Pallidum', 'Parabigeminal nucleus', 'Paraflocculus', 'Paragigantocellular reticular nucleus', 'Paramedian lobule', 'Parapyramidal nucleus', 'Parastrial nucleus', 'Parasubthalamic nucleus', 'Paraventricular hypothalamic nucleus', 'Paraventricular hypothalamic nucleus, descending division', 'Parvicellular reticular nucleus', 'Pedunculopontine nucleus', 'Periaqueductal gray', 'Peripeduncular nucleus', 'Perirhinal area', 'Periventricular hypothalamic nucleus, anterior part', 'Periventricular hypothalamic nucleus, intermediate part', 'Periventricular hypothalamic nucleus, posterior part', 'Periventricular hypothalamic nucleus, preoptic part', 'Piriform area', 'Piriform-amygdalar area', 'Pons', 'Pontine central gray', 'Pontine gray', 'Pontine reticular nucleus', 'Pontine reticular nucleus, caudal part', 'Posterior amygdalar nucleus', 'Posterior hypothalamic nucleus', 'Posterior pretectal nucleus', 'Posterodorsal preoptic nucleus', 'Postpiriform transition area', 'Precommissural nucleus', 'Prelimbic area', 'Preparasubthalamic nucleus', 'Somatomotor areas', 'Somatosensory areas', 'Principal sensory nucleus of the trigeminal', 'Red nucleus', 'Reticular nucleus of the thalamus', 'Retrochiasmatic area', 'Retrosplenial area', 'Rostral linear nucleus raphe', 'Septofimbrial nucleus', 'Septohippocampal nucleus', 'Simple lobule', 'Spinal nucleus of the trigeminal, interpolar part', 'Spinal nucleus of the trigeminal, oral part', 'Striatum', 'Subceruleus nucleus', 'Subfornical organ', 'Sublaterodorsal nucleus', 'Subparafascicular area', 'Subparafascicular nucleus', 'Subparaventricular zone', 'Substantia innominata', 'Substantia nigra, compact part', 'Substantia nigra, reticular part', 'Subthalamic nucleus', 'Superior central nucleus raphe', 'Superior colliculus, motor related, deep gray layer', 'Superior colliculus, motor related, deep white layer', 'Superior colliculus, motor related, intermediate gray layer', 'Superior colliculus, motor related, intermediate white layer', 'Superior colliculus, optic layer', 'Superior colliculus, superficial gray layer', 'Superior colliculus, zonal layer', 'Superior olivary complex', 'Suprachiasmatic nucleus', 'Supragenual nucleus', 'Supraoptic nucleus', 'Supratrigeminal nucleus', 'Taenia tecta', 'Tegmental reticular nucleus', 'Temporal association areas', 'Thalamus', 'Triangular nucleus of septum', 'Trochlear nucleus', 'Tuberal nucleus', 'Vascular organ of the lamina terminalis', 'Ventral group of the dorsal thalamus', 'Ventral premammillary nucleus', 'Ventral tegmental area', 'Ventral tegmental nucleus', 'Ventrolateral preoptic nucleus', 'Ventromedial hypothalamic nucleus', 'Visceral area', 'alveus', 'amygdalar capsule', 'anterior commissure, olfactory limb', 'anterior commissure, temporal limb', 'arbor vitae', 'brachium of the inferior colliculus', 'brachium of the superior colliculus', 'cerebal peduncle', 'cerebellar commissure', 'cerebral aqueduct', 'choroid plexus', 'cingulum bundle', 'columns of the fornix', 'corpus callosum, anterior forceps', 'corpus callosum, extreme capsule', 'corpus callosum, posterior forceps', 'corpus callosum, splenium', 'corticospinal tract', 'crossed tectospinal pathway', 'doral tegmental decussation', 'dorsal acoustic stria', 'dorsal fornix', 'dorsal hippocampal commissure', 'dorsal spinocerebellar tract', 'external capsule', 'external medullary lamina of the thalamus', 'facial nerve', 'fasciculus retroflexus', 'fiber tracts', 'fimbria', 'fourth ventricle', 'genu of corpus callosum', 'genu of the facial nerve', 'habenular commissure', 'inferior cerebellar peduncle', 'inferior colliculus commissure', 'internal capsule', 'lateral lemniscus', 'lateral olfactory tract, body', 'lateral recess', 'lateral ventricle', 'mammillary peduncle', 'mammillotegmental tract', 'mammillothalamic tract', 'medial forebrain bundle', 'medial lemniscus', 'medial longitudinal fascicle', 'middle cerebellar peduncle', 'motor root of the trigeminal nerve', 'nigrostriatal tract', 'oculomotor nerve', 'optic chiasm', 'optic nerve', 'optic tract', 'posterior commissure', 'principal mammillary tract', 'pyramid', 'rubrospinal tract', 'sensory root of the trigeminal nerve', 'spinal tract of the trigeminal nerve', 'stria medullaris', 'stria terminalis', 'subependymal zone', 'superior cerebelar peduncles', 'superior cerebellar peduncle decussation', 'superior colliculus commissure', 'supraoptic commissures', 'third ventricle', 'trapezoid body', 'trochlear nerve', 'uncinate fascicle', 'ventral hippocampal commissure', 'ventral spinocerebellar tract', 'ventral tegmental decussation', 'vestibular nerve', 'medial corticohypothalamic tract', 'dorsal limb', 'direct tectospinal pathway', 'Declive (VI)', 'olfactory nerve layer of main olfactory bulb'
            ],

        'st5':[
            
            'Cortical subplate','Olfactory areas','Isocortex','Basolateral amygdalar nucleus','Basomedial amygdalar nucleus','Claustrum','Hippocampal formation','Flocculus',
            'Endopiriform nucleus', 'Lateral amygdalar nucleus',

            # 'Medulla, motor related',  'Periventricular zone',  'Striatum-like amygdalar nuclei', 'Hypothalamic medial zone', 'Pretectal region', 
            # 'Anterior tegmental nucleus', 'Thalamus, polymodal association cortex related', 'Periventricular region', 'Pons, motor related',  
            #  'Pallidum, caudal region', 'Striatum dorsal region', 'Midbrain raphe nuclei', 'Cerebellum',  'Copula pyramidis', 
            #  'Ansiform lobule', 'Cuneiform nucleus',  'Dentate nucleus', 'Pallidum, medial region', 'Medulla, sensory related', 
            # 'Dorsal terminal nucleus of the accessory optic tract', 'Edinger-Westphal nucleus', , 'Fastigial nucleus', 'Hypothalamic lateral zone', 
            #  'Striatum ventral region', 'Pallidum, dorsal region', 'Hypothalamus', 'Inferior colliculus', 'Interposed nucleus', 'Periaqueductal gray', 
            # 'Pons, sensory related', 'Lateral septal complex', 'Lateral terminal nucleus of the accessory optic tract', 
            # 'Pons, behavioral state related', 'Lingula (I)', 'Central lobule', 'Culmen', 'Pallidum, ventral region', 'Thalamus, sensory-motor cortex related', 
            # 'Medial terminal nucleus of the accessory optic tract', 'Median eminence', 'Medulla', 'Midbrain', 'Midbrain reticular nucleus', 
            # 'Midbrain reticular nucleus, retrorubral area', 'Midbrain trigeminal nucleus', 'Nodulus (X)', 'Nucleus of the brachium of the inferior colliculus', 
            # 'Medulla, behavioral state related', 'Nucleus sagulum', 'Oculomotor nucleus', 'Pallidum', 'Parabigeminal nucleus', 'Paraflocculus', 'Paramedian lobule', 
            # 'Pedunculopontine nucleus', 'Pons', 'Posterior amygdalar nucleus', 'Red nucleus', 'Simple lobule', 'Striatum', 'Substantia nigra, compact part', 
            # 'Substantia nigra, reticular part', 'Superior colliculus, motor related', 'Superior colliculus, sensory related', 'Thalamus', 'Trochlear nucleus', 
            # 'Ventral tegmental area', 'Ventral tegmental nucleus', 'alveus', 'amygdalar capsule', 'anterior commissure, olfactory limb', 'anterior commissure, temporal limb', 
            # # 'arbor vitae', 'brachium of the inferior colliculus', 'brachium of the superior colliculus', 'cerebal peduncle', 'cerebellar commissure', 'cerebral aqueduct', 
            # # 'choroid plexus', 'cingulum bundle', 'postcommissural fornix', 'corpus callosum, anterior forceps', 'corpus callosum, extreme capsule', 
            # # 'corpus callosum, posterior forceps', 'corpus callosum, splenium', 'corticospinal tract', 'crossed tectospinal pathway', 'doral tegmental decussation', 
            # # 'dorsal acoustic stria', 'dorsal fornix', 'hippocampal commissures', 'dorsal spinocerebellar tract', 'external capsule', 'external medullary lamina of the thalamus', 
            # # 'facial nerve', 'fasciculus retroflexus', 'fiber tracts', 'fimbria', 'fourth ventricle', 'genu of corpus callosum', 'genu of the facial nerve', 'habenular commissure', 
            # # 'inferior cerebellar peduncle', 'inferior colliculus commissure', 'internal capsule', 'lateral lemniscus', 'lateral olfactory tract, body', 'lateral recess', 
            # # 'lateral ventricle', 'mammillary peduncle', 'mammillotegmental tract', 'mammillothalamic tract', 'medial forebrain bundle', 'medial lemniscus', 
            # # 'medial longitudinal fascicle', 'middle cerebellar peduncle', 'motor root of the trigeminal nerve', 'nigrostriatal tract', 'oculomotor nerve', 'optic chiasm', 
            # # 'optic nerve', 'optic tract', 'posterior commissure', 'principal mammillary tract', 'pyramid', 'rubrospinal tract', 'sensory root of the trigeminal nerve', 
            # # 'spinal tract of the trigeminal nerve', 'stria medullaris', 'stria terminalis', 'subependymal zone', 'superior cerebelar peduncles', 
            # # 'superior cerebellar peduncle decussation', 'superior colliculus commissure', 'supraoptic commissures', 'third ventricle', 'trapezoid body', 'trochlear nerve', 
            # # 'uncinate fascicle', 'ventral spinocerebellar tract', 'ventral tegmental decussation', 'vestibular nerve', 'dorsal limb', 'direct tectospinal pathway', 
            # # 'Declive (VI)', 'olfactory nerve layer of main olfactory bulb', 'root'
            ],
        'st6':[
            'Medulla', 'Cortical plate', 'Hypothalamus', 'Striatum', 'Midbrain, motor related', 'Thalamus', 'Pons', 'Cortical subplate', 'Pallidum', 
            'Midbrain, behavioral state related', 'Cerebellum', 'Hemispheric regions', 'Dentate nucleus', 'Fastigial nucleus', 'Midbrain, sensory related', 
            'Interposed nucleus', 'Vermal regions', 'Midbrain', 'fornix system', 'amygdalar capsule', 'anterior commissure, olfactory limb', 
            # 'anterior commissure, temporal limb', 'arbor vitae', 'cochlear nerve', 'brachium of the superior colliculus', 'cerebal peduncle', 
            # 'cerebellar commissure', 'cerebral aqueduct', 'choroid plexus', 'cingulum bundle', 'corpus callosum, anterior forceps', 'corpus callosum, extreme capsule', 
            # 'corpus callosum, posterior forceps', 'corpus callosum, splenium', 'corticospinal tract', 'crossed tectospinal pathway', 
            # 'doral tegmental decussation', 'inferior cerebellar peduncle', 'external medullary lamina of the thalamus', 'facial nerve', 'epithalamus related', 
            # 'fiber tracts', 'fourth ventricle', 'genu of corpus callosum', 'genu of the facial nerve', 'internal capsule', 'lateral olfactory tract, general', 
            # 'lateral recess', 'lateral ventricle', 'mammillary related', 'medial forebrain bundle', 'cervicothalamic tract', 'medial longitudinal fascicle', 
            # 'middle cerebellar peduncle', 'motor root of the trigeminal nerve', 'nigrostriatal tract', 'oculomotor nerve', 'optic chiasm', 'optic nerve', 'optic tract', 
            # 'posterior commissure', 'pyramid', 'rubrospinal tract', 'sensory root of the trigeminal nerve', 'stria terminalis', 'subependymal zone', 
            # 'superior cerebelar peduncles', 'superior colliculus commissure', 'supraoptic commissures', 'third ventricle', 'trochlear nerve', 'ventral tegmental decussation', 
            # 'vestibular nerve', 'direct tectospinal pathway', 'olfactory nerve layer of main olfactory bulb', 'root'
            ],
        'st7':[
            'Hindbrain', 'Cerebral cortex', 'Interbrain', 'Cerebral nuclei', 'Midbrain', 'Cerebellum', 'Cerebellar cortex', 'Cerebellar nuclei', 'cerebrum related', 
            #    'olfactory nerve', 'arbor vitae', 'vestibulocochlear nerve', 'optic nerve', 'corticospinal tract', 'cerebellar commissure', 
            #    'cerebral aqueduct', 'choroid plexus', 'corpus callosum', 'tectospinal pathway', 
            #    'cerebellar peduncles', 'thalamus related', 'facial nerve', 'hypothalamus related', 
            #    'fiber tracts', 'fourth ventricle', 'lateral recess', 'lateral ventricle', 'dorsal roots', 'oculomotor nerve', 'trigeminal nerve', 
            #    'cerebral nuclei related', 'rubrospinal tract', 'subependymal zone', 'third ventricle', 'trochlear nerve', 'root'
            ],
        'st8':[
            'Brain stem', 'Cerebrum', 'Cerebellum', 'medial forebrain bundle system',
            'cranial nerves', 'cerebellum related fiber tracts',
            'lateral forebrain bundle system', 
            'extrapyramidal fiber systems', 'fiber tracts']})

def old_main():



    st_level_keep_cols_dict = get_regions_of_interest_by_st_level()
    st_levels = [
        'st1', 'st2', 'st3', 'st4', 'st5', 'st6', 'st7',
        'st8', 'st9', 'st10'
    ]
    groups=['FC', 'FC+EXT']
    cols_to_exclude = [
    'animal_id', 'group', 'sex', 'strain', 'atlas_id','atlas_name', ]
    cols_to_sum = [       
    'nDapi', 'nZif', 'nGFP', 'nBoth', 'atlas_region_areas']

    bla_sub_region_names = [
        'Basolateral amygdalar nucleus, anterior caudal part',
        'Basolateral amygdalar nucleus, anterior lateral part',
        'Basolateral amygdalar nucleus, anterior medial part',
        'Basolateral amygdalar nucleus, posterior part',
        'Basolateral amygdalar nucleus, ventral part'
    ]
    animals_to_exclude = ['TEL38','TEL40','TEL42','TEL46','TEL47']
    ###################################################
    reg_id_map=pd.read_excel(r'D:\ReijmersLab\TEL\slides\onts_id_reg.xlsx')
    # load the reference_atlas labels to be able to extract the value atlas shape is (1320, 800, 1140) ZYX
    atlas_labels = imread(r'D:\ReijmersLab\TEL\ccf2017_atlas_images\ccf2017_labels_bla-subdivisions_20230302.tif')
    ###################################################

    run_date = '0414'
    outdir = os.path.join(r'C:\Users\pasca\Box\Reijmers Lab\Frank\TEL Cohort', f'2023_{run_date}_quantification_significant\\figures')
    input_dir_base = r'D:\ReijmersLab\TEL\quant_testing\test\testing_region_quantificiation_' + run_date
    input_paths = sorted([os.path.join(input_dir_base, c) for c in os.listdir(input_dir_base) if 'final_region_counts' in c])
    if not os.path.exists(outdir): os.makedirs(outdir)

    for input_i in list(range(len(input_paths))[:1]):
        ###################################################
        input_path = input_paths[input_i]
        input_fn = Path(input_path).stem
        print('using fn:', input_fn)
        fdf = pd.read_csv(input_path)
        # drop animals with little data
        fdf = fdf[~fdf['animal_id'].isin(animals_to_exclude)]
        
        # print general stats about expression in the BLA
        df_st_level = 'st5'
        stdf5, gdf5 = get_st_level_df(fdf, df_st_level)
        for yvar in ['pZif', 'pGFP', 'pBoth', 'pZifInGfp']:
            comp_zifingfp = get_yvar_stats(df_st_level, gdf5, yvar, 'Basolateral amygdalar nucleus', round_to=6)

        bla_regionnames = get_st_names_contain_str(fdf, ['Basolateral amygdalar nucleus'], 'atlas_name')
    
        
        

        



        # plot atlas image by diff
        ###################################################
        def plot_atlas_region_heatmaps(top_diff_dict_noAbs, atlas_slice_indicies, SAVE=False):
            for atlas_z_slice in atlas_slice_indicies[:]:
                for y_var in list(top_diff_dict_noAbs.keys()):
                    plot_dict = top_diff_dict_noAbs[y_var]
                    plot_data = plot_dict['data']
                    plot_ydiff = plot_dict['diff_val']
                    p_value_thresh = plot_dict['p_value_thresh']

                    if len(plot_data) == 0:
                        print(f'skiping bc no data {input_fn}_region_heatmap_st{st_level}_pVal{p_value_thresh}_z{atlas_z_slice}_{y_var}.svg')
                        continue

                    # atlas_z_slice = 550
                    atlas_slice = atlas_labels[atlas_z_slice,...]
                    atlas_outline = np.where(skimage.filters.sobel(atlas_slice)>0.0000, 255, 0)
                    # atlas_outline = np.where(skimage.morphology.skeletonize(atlas_outline)>0, 255, 0)
                    # plt.imshow(atlas_outline)
                    atlas_region_heatmap = apply_heat_map_to_array(atlas_slice, plot_data)
                    
                    fig,ax = plt.subplots()
                    hm_cmap = sns.color_palette("coolwarm", as_cmap=True)
                    # hm_cmap = 'seismic'
                    # hm_cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

                    ol = ax.imshow(atlas_outline, cmap='binary', alpha=0.5)
                    hm = ax.imshow(atlas_region_heatmap, norm=colors.CenteredNorm(), cmap=hm_cmap,
                            zorder=0, aspect="auto", alpha=0.5
                            )
                    ax.axis('off')
                    cax = plt.axes([0.9, 0.2, 0.025, 0.6])
                    cbar = plt.colorbar(mappable=hm, cax=cax)
                    cbar.ax.get_yaxis().labelpad = 15
                    cbar.ax.set_ylabel(f'{plot_ydiff} {y_var[1:]}', rotation=270)
                    if SAVE:
                        fig.savefig(
                            os.path.join(
                            outdir,
                            f'{input_fn}_region_heatmap_st{st_level}_pVal{p_value_thresh}_z{atlas_z_slice}_{y_var}.svg'
                        ), dpi=300, bbox_inches='tight')
                    plt.show()
        if bool(1):
            st_level = 'st1'
            st_level_structs = [el for el in fdf[st_level].unique()]
            st_level_keep_cols_dict[st_level] = st_level_structs[:]
            # get p[yvar] for each region by groups
            by_animal_df_sums, by_group_df_sums = get_st_level_df(fdf, st_level)
            
            top_n = 500
            p_value_thresh = 0.15
            yvar_heatmaps = ['pZifInGfp'] #'pGFP','pBoth', 
            top_diff_dicts = {}
            diff_val = 'diffs_log'
            for yi in range(len(yvar_heatmaps)):
                yvar = yvar_heatmaps[yi]
                rdbg = get_region_differences_by_group(by_group_df_sums, st_level, yvar, abs_val=True).reset_index()
                
                # oneway anova for signficance (p<0.05)
                regions_pval_df2 = (by_animal_df_sums.dropna(subset=[yvar])
                        .groupby(st_level)
                        .apply(lambda x: pd.Series({'p_value': scipy.stats.f_oneway(*[x[x['group'] == g][yvar].values for g in groups]).pvalue}))).reset_index()
                # regions_pval_df2[regions_pval_df2['p_value'] < 0.1]
                rdbg = rdbg.merge(regions_pval_df2[[st_level, 'p_value']], on=st_level, how='left')
                rdbg = rdbg[rdbg['p_value'] < p_value_thresh].dropna(subset=['p_value'])

                # drop st_level names that contain specific phrases
                drop_st_names = ['tract', 'nerve', 'ventricle', 'bundle']
                drop_names = get_st_names_contain_str(rdbg, drop_st_names, st_level)
                rdbg = rdbg[~rdbg[st_level].isin(drop_names)]
                top_vals = rdbg[st_level]

                plot_df = rdbg.set_index(st_level)[diff_val]

                # append dict
                top_diff_dicts[yvar] = {
                    'data':plot_df,
                    'order':top_vals,
                    'diff_val':diff_val,
                    'top_n':top_n,
                    'p_value_thresh':p_value_thresh
                }

            plot_atlas_region_heatmaps(
                top_diff_dicts, np.arange(350,1051,100),
                SAVE=bool(1)
            )







    # # get heatmap as volume
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as mcolors

    # # Generate the diverging color map
    # blue = np.array([0, 0, 255, 255]) / 255.0
    # white = np.array([255, 255, 255, 0]) / 255.0
    # red = np.array([255, 0, 0, 255]) / 255.0

    # colors = [blue, white, red]
    # cmap_name = 'my_diverging_cmap'
    # cm = mcolors.ListedColormap(colors, cmap_name)

    # # Generate a sample 3D volume
    # volume = apply_heat_map_to_array(atlas_labels, top_diff_dict_noAbs['pZif'])

    # volume = np.zeros_like(atlas_labels)
    # volume_shape = volume.shape
    # volume = np.random.rand(*volume_shape)
    # # Normalize the volume to range [0, 1]
    # volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # # Apply the color map to the volume
    # rgba_volume = cm(volume)

    # # Display a slice from the middle of the volume with the applied color map
    # slice_index = volume_shape[0] // 2
    # plt.imshow(rgba_volume[slice_index, :, :])

    # cax = plt.axes([0.9, 0.2, 0.025, 0.6])
    # cbar = plt.colorbar(cm, cax=cax)
    # plt.colorbar()
    # plt.show()


###################################################################################################
def verify_output_dir(adir):
    if not os.path.exists(adir): os.mkdir(adir)
    return adir

def load_dfs(df_paths):
    if len(df_paths) > 1:
        fdf = pd.concat([pd.read_csv(df_path) for df_path in df_paths], ignore_index=True)
    else:
        df_paths = df_paths[0]
        fdf = pd.read_csv(df_paths)
    print(df_paths)
    fdf = clean_init_df(fdf)
    return fdf

def clean_init_df(adf):
    # drop notFound regions
    adf = adf[adf['region_name'] != 'notFound']
    # drop 'unnamed' col
    if 'Unnamed: 0' in adf.columns:
        adf = adf.drop(columns=['Unnamed: 0'])
    # are there any nan vals?
    objCols = adf.select_dtypes('object').columns
    # [len(adf[objCols].query(f'{c}.isna()')) for c in objCols] # get nan counts
    return adf

def reduce_df(fdf, groupbyCols, animals_to_remove):
    andf = fdf.groupby(groupbyCols,as_index=False).sum(numeric_only=True).sort_values('animal_id')
    # add cohorts
    if 'cohort' not in andf.columns.to_list():
        andf['cohort'] = [ac.get_animal_cohort(anid)['cohort_name'] for anid in andf['animal_id'].values]
    andf['groupcohort'] = andf['group'] + '-' + andf['cohort']
    # remove an animal
    andf = andf.loc[~andf['animal_id'].isin(animals_to_remove)]

    andf = andf.assign(
        reg_area = andf['reg_area'].apply(lambda x: pixel_to_mm(x)),
        nTotal = andf['nDapi'] + andf['nZif'] + andf['nGFP'] + andf['nBoth'],
    )
    andf = andf.assign(
        animal_int = andf.apply(lambda x: int(x['animal_id'][3:]), axis=1),
        reactivation = andf['nBoth']/andf['nGFP'],
        
        dapi_density = andf['nDapi']/andf['reg_area'],
        zif_density = andf['nZif']/andf['reg_area'],
        gfp_density = andf['nGFP']/andf['reg_area'],
        both_density = andf['nBoth']/andf['reg_area'],
        
        pDapi = andf['nDapi']/andf['nTotal'],
        pZif = andf['nZif']/andf['nDapi'],
        pGFP = andf['nGFP']/andf['nDapi'],
        pBoth = andf['nBoth']/andf['nDapi'],
    )
    andf = andf.assign(
        reactivation_density = andf['both_density']/andf['gfp_density']
    )
    return andf

def pixel_to_mm(pixel_area, pixel_size_in_microns=0.650):
    # Convert pixel_size_in_microns to square millimeters
    pixel_area_in_mm = (pixel_size_in_microns / 1000) ** 2 * pixel_area
    return pixel_area_in_mm

def num_regions_by_colocal_id(andf, nThresh=0):
    ''' a test to see how many regions were detected '''
    result_dicts = []
    print_str = f"num regions overall, by animal, (n>{nThresh}): {len(andf['reg_id'].unique())}, "
    for col in ['nDapi', 'nZif', 'nGFP', 'nBoth']:
        print_str += f"{col}: {len(andf.loc[andf[col]>0]['reg_id'].unique())}, "
    print(print_str)

    for cohort in andf['cohort'].unique():
        an_print_str = f'\t{cohort}:'
        for col in ['nDapi', 'nZif', 'nGFP', 'nBoth']:
            coldf = andf.loc[(andf[col]>nThresh) & (andf['cohort'] == cohort)]
            col_len = len(coldf['reg_id'].unique())
            an_print_str += f"{col}: {col_len}, "
            for animal_id in coldf['animal_id'].unique():
                animaldf = coldf[coldf['animal_id']==animal_id]
                result_dicts.append({'cohort':cohort, 'thresh':nThresh, 'animal_id':animal_id, 'cellType':col, 'nRegions':len(animaldf['reg_id'].unique())})
        print(an_print_str)
    return result_dicts

def check_regions_by_animal(andf, outdir, SAVE=False):
    # check number of regions with a given number of cells
    results = []
    threshes = [0, 100, 1000, 10000]
    for t in threshes:
        for res in num_regions_by_colocal_id(andf, nThresh=t):
            results.append(res)
    ncolocdf = pd.DataFrame(results)
    fig, axs = plt.subplots(1,4, figsize=(12,3), sharey=True)
    for i in range(len(threshes)):
        sns.barplot(data=ncolocdf.loc[ncolocdf['thresh']==threshes[i]], x='cellType', y='nRegions', hue='cohort', ax=axs[i], edgecolor=".5", facecolor=(0, 0, 0, 0), linewidth=3)
        axs[i].legend([],[])
        sns.stripplot(data=ncolocdf.loc[ncolocdf['thresh']==threshes[i]], x='cellType', y='nRegions', hue='cohort', dodge=True, ax=axs[i], legend=True if i == len(threshes)-1 else False)
        axs[i].set_title(f'nRegions w/ counts > {threshes[i]}')
        
            
    fig.suptitle('number of unique regions by cohort')
    plt.tight_layout()
    if SAVE:
        fig.savefig(os.path.join(outdir, f'{today}_nRegions_cell_counts.svg'), bbox_inches='tight', dpi=300)
    plt.show() 

def plot_byGroup_values(andf, var):
    # simple plots comparing a variable across groupcohort
    data = andf.assign(animal_int = andf['animal_int'].astype('str')).dropna(axis=0, how='any')
    palette = dict(zip(
        ['CTX-cohort4', 'FC+EXT-cohort2', 'FC+EXT-cohort3', 'FC+EXT-cohort4', 'FC-cohort2', 'FC-cohort3', 'FC-cohort4'], 
        # ['gray', 'blue', 'purple', 'green', 'red', 'orange', 'pink']
        ['#4d4d4d', '#4575b4', '#74add1', '#abd9e9', '#d73027', '#fc8d59', '#fee090'],
    ))
    palette_sp = {k:'k' for k in palette}
    
    fig,axs=plt.subplots(1,2, figsize=(8,4), sharey=True)
    # sns.barplot(data=data, x='animal_int', y=var, hue='groupcohort', dodge=False, palette=palette, hue_order=palette.keys())
    
    sns.barplot(ax=axs[0],data=data.groupby('groupcohort', as_index=False).mean(), x='groupcohort', y=var, hue='groupcohort', dodge=False, palette=palette, hue_order=palette.keys())
    # sns.swarmplot(ax=axs[0],data=data, x='groupcohort', y=var, hue='groupcohort', dodge=False, palette=palette_sp, hue_order=palette.keys())
    axs[0].set_title('brain-wide mean')
    axs[0].legend().remove()
    
    sns.barplot(ax=axs[1],
        data=data.loc[data['reg_id']==295].groupby(['groupcohort'], as_index=False).mean(numeric_only=True),
        x='reg_id', y=var, hue='groupcohort', palette=palette, hue_order=palette.keys())
    sns.swarmplot(ax=axs[1],
        data=data.loc[data['reg_id']==295],
        x='reg_id', y=var, hue='groupcohort', palette=palette_sp, hue_order=palette.keys(), dodge=True)
    
    axs[1].set_title('BLA mean')
    axs[1].legend(bbox_to_anchor=(1.8, 1.0))
    plt.show()

    #######################################################################
    # plot reactivation difference and p val
    pval_df,all_vals = [], {'FC':[], 'FC+EXT':[]}
    for cohort in sorted(data['cohort'].unique()):
        vals, means = [], []
        for grp in ['FC', 'FC+EXT']:
            vs = data.loc[(data['cohort']==cohort) & (data['group']==grp) & (data['reg_id']==295)]['reactivation']
            vals.append(vs.values); means.append(vs.mean())
            all_vals[grp].extend(vs.to_list())
        pval_df.append({'cohort':cohort, 'pval':scipy.stats.f_oneway(*vals).pvalue, 'mean_diff':means[0]-means[1]})
    
    pval_df = pd.DataFrame(pval_df)
    
    all_vals_p = scipy.stats.f_oneway(all_vals['FC'], all_vals['FC+EXT']).pvalue
    all_vals_diff = np.array(all_vals['FC']).mean() - np.array(all_vals['FC+EXT']).mean()
    pval_df = pval_df.append(pd.DataFrame([['combined', all_vals_p, all_vals_diff]], columns=pval_df.columns.to_list()), ignore_index=True)
    print(tabulate(pval_df, headers='keys', tablefmt="heavy_outline"))


def get_regions_containing_string(astr):
    ''' returns a list of names and ids of regions whose name contains the str '''
    bla_reg_names, bla_reg_ids = [], []
    for reg_name in fdf['region_name'].unique():
        if astr in reg_name.lower():
            bla_reg_names.append(reg_name)
            # get id of reg name
            bla_reg_ids.append(fdf[fdf['region_name']==reg_name]['reg_id'].values[0])
    return bla_reg_names, bla_reg_ids


def get_n_animals_per_group(adf, groupping_var, group_order):
    return [len(adf[adf[groupping_var]==grp]['animal_id'].unique()) for grp in group_order] # get counts in order specified

    

def sort_list_by_order(list_to_sort, order_list):
    ''' sort a list in the order specified by another list '''
    order_dict = {value: index for index, value in enumerate(order_list)}
    return sorted(list_to_sort, key=lambda x: order_dict.get(x, float('inf')))


def get_palette_by_hue(hue_var):
    if hue_var == 'groupsex':
        hue_order=['FC+EXT-f', 'FC+EXT-m', 'FC-f', 'FC-m', 'CTX-f', 'CTX-m',]
        palette=dict(zip(hue_order, ['purple', 'blue', 'pink', 'red', '#4d4d4d', 'gray']))
    elif hue_var == 'group':
        hue_order=['FC+EXT', 'FC', 'CTX']
        palette=dict(zip(hue_order, ['blue', 'red', 'gray']))
    elif hue_var == 'cohort':
        hue_order=['cohort2', 'cohort3', 'cohort4']
        palette=dict(zip(hue_order, ['green', 'orange', 'purple']))
    elif hue_var == 'groupcohort':
        hue_order=['FC+EXT-cohort2', 'FC+EXT-cohort3', 'FC+EXT-cohort4', 'FC-cohort2', 'FC-cohort3', 'FC-cohort4', 'CTX-cohort4']
        palette=dict(zip(hue_order, ['#4575b4', '#74add1', '#abd9e9', '#d73027', '#fc8d59', '#fee090', '#4d4d4d']))       
        # palette=dict(zip(hue_order, ['blue', 'purple', 'green', 'red', 'orange', 'pink', 'gray',]))
    # elif hue_var == 'groupcohortsex':
    #     hue_order=['FC-cohort2-f', 'FC-cohort2-m','FC-cohort3-f', 'FC-cohort3-m','FC+EXT-cohort2-f','FC+EXT-cohort2-m', 'FC+EXT-cohort3-f', 'FC+EXT-cohort3-m']
    #     palette=dict(zip(hue_order, ['pink', 'red', 'yellow', 'orange', 'purple', 'blue', 'green', 'teal']))
    else:
        raise ValueError(f'hues are not defined for hue_var: {hue_var}')
    return hue_order, palette

def plot_byAnimal_overview(andf, vars):
    for var in vars:
        plot_byGroup_values(andf, var)
    
    dataByAn = andf.loc[andf['reg_id']==295].sort_values(['group', 'reactivation'])[
        ['groupcohort', 'animal_id', 'nGFP', 'nZif', 'nBoth', 'reactivation']]
    print(tabulate(dataByAn, headers='keys', tablefmt='heavy_outline'))


"""
###################################################################################################
# compiled versions
        # '2023_0529_all_quant_data.csv',
        # '2023_0603_all_quant_data_cohort2.csv',
        # '2023_0603_all_quant_data.csv',
        # '2023_0603_all_quant_data_cohort2_t894.csv',

# thresh effects
2023_0603_all_quant_data_cohort2_t894.csv
    byAnMean {'FC': 0.07665338826305895, 'FC+EXT': 0.06004202383486188}
    bySumMean {'FC': 0.07500635647088737, 'FC+EXT': 0.07300817933959407}
2023_0603_all_quant_data_cohort2.csv
    byAnMean {'FC': 0.1356213403340714, 'FC+EXT': 0.0740843851578501}
2023_0603_all_quant_data.csv
    byAnMean {'FC': 0.04390600116734903, 'FC+EXT': 0.034242215784317104}
    bySumMean {'FC': 0.04222081486172683, 'FC+EXT': 0.03714799281006591}
2023_0529_all_quant_data.csv
    byAnMean {'FC': 0.16198586890977912, 'FC+EXT': 0.14099857292116258}
    bySumMean {'FC': 0.1612835127717965, 'FC+EXT': 0.15008987417615338}

cohort 3 and 2 no thresh
'2023_0529_all_quant_data.csv' + '2023_0603_all_quant_data_cohort2.csv'
    byAnMean {'FC': 0.15026830065390903, 'FC+EXT': 0.11423289781583759}
cohort 3 and 2 thresh 530, 894
'2023_0603_all_quant_data.csv' + '2023_0603_all_quant_data_cohort2_t894.csv' 
    byAnMean {'FC': 0.058460395432109, 'FC+EXT': 0.035013326223357545}

'2023_0713_quant_data_cohort2_t-100-650-0.45-600-125.csv', FC: 0.156, EXT: 0.086
'2023_0713_quant_data_cohort3_t-120-350-0.45-250-150.csv', FC: 0.114, EXT: 0.080

previous files 7/19/23
        # '2023_0626_quant_data_cohort2_t-0.csv',
        # '2023_0626_quant_data_cohort3_t-0.csv',
        # '2023_0626_quant_data_cohort2_t-600.csv',
        # '2023_0626_quant_data_cohort3_t-400.csv',
        # '2023_0710_quant_data_cohort3_t-400.csv'
        # '2023_0711_quant_data_cohort2_t-100-650-0.45-600-125.csv',
        # '2023_0711_quant_data_cohort3_t-120-350-0.45-250-200.csv',
        # '2023_0711_quant_data_cohort3_t-120-350-0.45-250-175.csv',
        # '2023_0713_quant_data_cohort2_t-100-650-0.45-600-125.csv',

###################################################################################################
# num regions overall
    - 0718
        num regions overall (n>0): 814, nDapi: 814, nZif: 810, nGFP: 811, nBoth: 785, 
            cohort2:nDapi: 810, nZif: 793, nGFP: 806, nBoth: 758, 
            cohort3:nDapi: 810, nZif: 801, nGFP: 808, nBoth: 754, 
        num regions overall (n>100): 814, nDapi: 814, nZif: 810, nGFP: 811, nBoth: 785, 
            cohort2:nDapi: 798, nZif: 559, nGFP: 500, nBoth: 250, 
            cohort3:nDapi: 774, nZif: 540, nGFP: 544, nBoth: 276, 
        num regions overall (n>1000): 814, nDapi: 814, nZif: 810, nGFP: 811, nBoth: 785, 
            cohort2:nDapi: 690, nZif: 338, nGFP: 150, nBoth: 47, 
            cohort3:nDapi: 573, nZif: 312, nGFP: 166, nBoth: 50, 
        num regions overall (n>10000): 814, nDapi: 814, nZif: 810, nGFP: 811, nBoth: 785, 
            cohort2:nDapi: 336, nZif: 92, nGFP: 26, nBoth: 10, 
            cohort3:nDapi: 150, nZif: 75, nGFP: 29, nBoth: 10, 
###################################################################################################
"""

############################################################################################################################################
############################################################################################################################################
# init/read data
############################################################################################################################################
# useful funcs
# get region ids containing a specific string
    # bla_reg_names, bla_reg_ids = get_regions_containing_string('lateral amygdala') # get all subregions containing bla

# for the 0719 version counts for zif are close, gfp and dapi is not

# NOTES 8/16/23
    # counts from 0806 values (FC, EXT)
    # reactivation: .15, .8
    # nZif: 2500, 1200-1500
    # nGFP: 700-1100
    # nBoth: 100-150, 50-75
############################################################################################################################################

if __name__ == '__main__':

    SMALL_REGION_ANALYSIS = bool(0)
    SAVE_GRAPHS_SRA = bool(0)
    WHOLE_BRAIN_ANALYSIS = bool(1)
    SAVE_GRAPHS_WBA = bool(0)
    SAVE_GRAPHS_NREGIONS = bool(0)

    ac = AnimalsContainer()
    ont_ids = arhfs.load_ontology()
    base_dir = r'D:\ReijmersLab\TEL\slides\quant_data\counts'
    graph_outdir = verify_output_dir(os.path.join(r'C:\Users\pasca\Box\Reijmers Lab\Frank\TEL Project\quantification', f'{today}_quantification'))
    
    df_paths = [os.path.join(base_dir, fn) for fn in [
        # '2023_0716_quant_data_cohort2_t-75-350-0.45-550-150.csv',
        # '2023_0713_quant_data_cohort3_t-120-350-0.45-250-150.csv',
        # '2023_0719_quant_data_cohort2_t-75-350-0.45-550-150.csv',
        # '2023_0719_quant_data_cohort3_t-120-350-0.45-250-150.csv',
        # '2023_0806_quant_data_cohort2_t-75-350-0.45-550-100-5-85-4-31.csv',
        # '2023_0806_quant_data_cohort3_t-75-350-0.45-360-100-5-85-4-31.csv',

        # '2023_0817_quant_data_cohort4_t-75-350-0.45-315-145-12-85-7-31.csv',
        # '2023_0817_quant_data_cohort4_t-75-350-0.45-375-160-12-85-7-31.csv',
        '2023_0827_151059_quant_data.csv'
    ]]

    # clean up data
    fdf = load_dfs(df_paths)
    # create a df summing all images together for each animal, preserving groupping variables
    groupbyCols = ['reg_id', 'region_name', 'st_level', 'sex', 'strain', 'group', 'cohort', 'animal_id']
    animals_to_remove =  ['TEL16', 'TEL36', 'TEL49', 'TEL52', 'TEL61']
    andf = reduce_df(fdf, groupbyCols, animals_to_remove)    
    andf.to_csv(os.path.join(graph_outdir, '2023_0827_151059_quant_data_byAnimal.csv'), index=False)
    plot_byAnimal_overview(andf, ['reactivation', 'zif_density', 'pZif'][:-1])
    


    
    ##############################################################################################################
    # PARAMS - SMALL REGION ANALYSIS
    parent_region_id = 295
    x_var = 'region_name'
    # YVARS = ['reg_area', 'nDapi', 'nZif', 'nGFP', 'nBoth', 'reactivationGFP'][:] # 'reactivationDapi', 'reactivationZif', 
    YVARS = ['reg_area', 'dapi_density', 'zif_density', 'gfp_density', 'both_density', 'reactivation', 'reactivation_density'] 
    hue_vars = ['group', 'groupcohort', 'cohort', 'groupsex', 'groupcohortsex'][:-1] # cohort
    plot_outname_base = f'{today}_reg{parent_region_id}'

    regions_containing_str = 'lateral amygdala'
    region_to_remove = None #'Basolateral amygdalar nucleus, ventral part'

    ######################
    # get regions to plot
    bla_reg_names, bla_reg_ids = get_regions_containing_string(regions_containing_str) # get all subregions containing bla
    # stls, bla_reg_ids = zip(*arhfs.gather_ids_by_st_level(ont_ids[parent_region_id])) # OR by region id
    # xorder = bla_reg_names
    # x_axis_labels = bla_reg_names

    # OR manually specify order and labels
    xorder = [
            'Basolateral amygdalar nucleus',
            'Basolateral amygdalar nucleus, posterior part',
            'Basolateral amygdalar nucleus, ventral part',
            'Basolateral amygdalar nucleus, anterior part',
            'Basolateral amygdalar nucleus, anterior caudal part',
            'Basolateral amygdalar nucleus, anterior lateral part',
            'Basolateral amygdalar nucleus, anterior medial part',
            'Lateral amygdalar nucleus',        
    ]
    
    x_axis_labels = ['BLA','BLAp','BLAv', 'BLAa', 'BLAac','BLAal','BLAam', 'LA'] # ,'BLAv'
    

    
    # tests
    ##############################################################################################################
    if bool(0):
        # check number of regions with a given number of cells
        check_regions_by_animal(andf, graph_outdir, SAVE=SAVE_GRAPHS_NREGIONS)

        if SAVE_GRAPHS_SRA: # save all data
            andf.to_csv(os.path.join(graph_outdir, '2023_0716_all_data.csv'), index=False)

        andf.groupby('groupcohort').mean(numeric_only=True)

    
    







    
    # ###########################################################################################################################################
    # ###########################################################################################################################################
    # whole-brain analysis
    # ###########################################################################################################################################
    # ###########################################################################################################################################
    
    # ###################################################
        
    # for differential compariason remove regions unique to only one group
    ###################################################
    def plot_boxplots(top_diff_dict, st_level, input_fn, groups, outdir, SAVE=False):

        # for y_var in ['pZif','pGFP','pBoth']:
        for y_var in list(top_diff_dict.keys()):
            plot_dict = top_diff_dict[y_var]
            # params: colors and groups
            group_palette = dict(zip(['FC', 'FC+EXT'], [(15/255,153/255,178/255,0.5),(3/255,78/255,97/255,0.5)]))
            fc_patch = mpatches.Rectangle((0, 0), 1, 1, color=(15/255,153/255,178/255,0.5), alpha=0.3)
            fcext_patch = mpatches.Rectangle((0, 0), 1, 1, color=(3/255,78/255,97/255,0.5), alpha=0.3)
            
            swarmplot_hue = 'groupcohort'
            swarmplot_groups, swarmplot_palette = get_palette_by_hue(swarmplot_hue)

            if 'order' in list(plot_dict.keys()): # using top n diffs
                plot_df = plot_dict['data']
                order = plot_dict['order']
                top_n_str = plot_dict['top_n']
                diff_val_str = plot_dict['diff_val']
                p_value_thresh = plot_dict['p_value_thresh']
            else:
                plot_df = by_animal_df_sums
                order = sorted(by_animal_df_sums['region_name'].unique()).values
            
            if len(plot_df) == 0:
                print(f'skiping bc no data {input_fn}_{st_level}-highest_delta_top{top_n_str}by{diff_val_str}_pVal{p_value_thresh}_{y_var}_boxplots.svg')
                continue

            

            fig,ax = plt.subplots(figsize=(25,8))
            sns.barplot(data=plot_df, x='region_name', y=y_var, hue='group', 
                        palette=group_palette, hue_order=groups, dodge=True, order=order,
                        ax=ax, alpha=0.3, 
                        
            )
            # swarmplot where cohorts are dodged but groups overlap (i.e. fc2 and fc3 are not dodged, but fc,ext are)
            for cohortn, cohdf in plot_df.groupby('cohort'):
                spg = [el for el in list(cohdf[swarmplot_hue].unique())]
                spp = {k:v for k,v in swarmplot_palette.items() if k in spg}
                sns.swarmplot(data=cohdf, x='region_name', y=y_var, order=order,
                              hue=swarmplot_hue, palette=spp, hue_order=spg, 
                            ax=ax, legend=False, warn_thresh=1.0,  dodge=True, 
                )
            ax.set_xlabel(f'Brain structures (st{st_level})')
            if y_var[0].startswith('n'): # log scale for n counts 
                ax.set_yscale('log')
            plt.xticks(rotation=-45, ha='left')
            ax.legend([],[])
            ax.legend(handles=[fc_patch, fcext_patch], labels=['FC', 'FC+EXT'], loc='upper right')

            if SAVE:
                fig_name = f'{input_fn}_{st_level}-highest_delta_top{top_n_str}by{diff_val_str}_pVal{p_value_thresh}_{y_var}_boxplots.svg'
                fig.savefig(os.path.join(outdir, fig_name), dpi=300, bbox_inches='tight')
            plt.show()

    if WHOLE_BRAIN_ANALYSIS:
        groupby = 'group'
        outdir = ''

        st_level = None
        top_n = 10
        p_value_thresh = 0.05 
        yvars_plots = ['reactivation_density', 'pZif', 'pGFP', 'pReactivationZif', 'pZifInGfp', 'nBoth', 'pBoth'][:1]
        diff_val = 'ratio'
        cohorts_wba = ['cohort2', 'cohort3'][:]
        TREND_THRESH = 1

        # set number of animals by cohort specified
        if len(cohorts_wba) == 1:
            andf = andf[andf['cohort'] == cohorts_wba[0]]
        if len(cohorts_wba) > 1:
            FILTER_N_ANIMALS_FC, FILTER_N_ANIMALS_EXT = 16, 13
        elif cohorts_wba[0] == 'cohort2':
            FILTER_N_ANIMALS_FC, FILTER_N_ANIMALS_EXT = 7, 5
        elif cohorts_wba[0] == 'cohort3':
            FILTER_N_ANIMALS_FC, FILTER_N_ANIMALS_EXT = 8, 8



        input_fn = today
        groups, palette = get_palette_by_hue(groupby)
        del(groups[groups.index('CTX')])
        del(palette['CTX'])
        
        

        # for i in range(1, 12):
        #     print(i,
        #     len(andf.loc[andf['st_level']==str(i)]['region_name'].unique()))

        # st_levels = [str(i) for i in range(1,12)]
        # st_level = st_levels[7]
        # st_level_structs = [el for el in andf.loc[andf['st_level']==st_level]['region_name'].unique()]
        # st_level_keep_cols_dict = get_regions_of_interest_by_st_level()
        # structs_of_interest = st_level_keep_cols_dict['st' + st_level]
        

        # get p[yvar] for each region by groups
        andf = andf.loc[andf['group']!='CTX']
        by_animal_df_sums, by_group_df_sums = get_st_level_df(andf, st_level, groupby)

        

        top_diff_dicts = {}
        for yi in range(len(yvars_plots)):
            
            yvar = yvars_plots[yi]
            
            rdbg = get_region_differences_by_group(by_group_df_sums, yvar, groupby, groups, abs_val=True)
            rdbg = get_rdbg_pvals(rdbg, by_animal_df_sums, yvar, groups)
            
            
            

            # FILTERS 
            #################################################################################
            print(yvar)
            print('len before filters', len(rdbg))
            
           
            # filter observations
            NFC,NEXT = [], []
            for areg in rdbg['region_name']:
                fcdf = andf[(andf['region_name'] == areg) & (andf['group'] == 'FC') ]
                extdf = andf[(andf['region_name'] == areg) & (andf['group'] == 'FC+EXT')]
                
                # filter nboth 
                fcdf = fcdf[fcdf.nBoth>0]
                extdf = extdf[extdf.nBoth>0]
                # filter variance
                NFC.append(len(fcdf))
                NEXT.append(len(extdf))

            rdbg = rdbg.assign(nFC = NFC, nEXT=NEXT)
            rdbg = rdbg[(rdbg.nFC>FILTER_N_ANIMALS_FC) & (rdbg.nEXT>FILTER_N_ANIMALS_EXT)]
            print('len after nAnimal, nBoth filters', len(rdbg))


            # drop st_level names that contain specific phrases
            drop_st_names = ['tract', 'nerve', 'ventricle', 'bundle']
            for drpname in drop_st_names:
                rdbg = rdbg.loc[~rdbg['region_name'].str.contains(drpname)]
            print('len after drop regionname filters', len(rdbg))

            # filter pval
            rdbg = rdbg[rdbg['p_value'] < p_value_thresh].dropna(subset=['p_value'])
            print('len after pval filters', len(rdbg))


            # rdbg = rdbg.loc[rdbg['region_name'].str.contains('amygdala')]


            # drop regions where cohort2 and 3 diverge in trend (% diff between fc and ext)
            # divergers = test_cohort_differences(andf, trend_thresh=TREND_THRESH)
            # regions_to_remove = list(divergers.keys())
            # rdbg = rdbg.loc[~rdbg['region_name'].isin(regions_to_remove)]
            # print('len after drop divergers', len(rdbg))

            # get the trenders per group 
            fc_gthan, ext_gthan = rdbg[rdbg['diff']>0], rdbg[rdbg['diff']<0]
            print(f'n total: {len(rdbg)}, fc higher: {len(fc_gthan)}, ext higher: {len(ext_gthan)}')

            # grpdf = by_animal_df_sums.groupby(['region_name', 'group'], as_index=False).sum(numeric_only=True)
            # for region in rdbg.sort_values('ratio', ascending=False)['region_name'].values:
            #     print(grpdf.loc[grpdf['region_name']==region]['nGFP'])
                

            # get the top values
            top_vals = get_top_n_diffs(rdbg, diff_val, top_n=top_n, drop_no_sig_diff_thresh=False)['region_name'].to_list()
            # rdbg = rdbg[rdbg['region_name'].isin(top_vals)].sort_values(diff_val, ascending=False)

            # apply the get the top vals to the plotting data that includes each animal individually
            plot_df = by_animal_df_sums[by_animal_df_sums['region_name'].isin(top_vals)]
            # append dict
            top_diff_dicts[yvar] = {
                'data':plot_df,
                'order':top_vals,
                'diff_val':diff_val,
                'top_n':top_n,
                'p_value_thresh':p_value_thresh
            }
            print()

        
        plot_boxplots(
            top_diff_dicts, st_level, input_fn, groups, outdir, 
            SAVE=SAVE_GRAPHS_WBA
        )

       











############################################################################################################################################
############################################################################################################################################
# small regions ANALYSIS 
############################################################################################################################################
############################################################################################################################################
    

    if SMALL_REGION_ANALYSIS:      
        # filter regions that don't have any data for any group
        bla_reg_ids = [i for i in bla_reg_ids if i in andf['reg_id'].unique()]
        bla_reg_names = arhfs.get_attributes_for_list_of_ids(ont_ids, bla_reg_ids, 'name')
        bla_acronyms = arhfs.get_attributes_for_list_of_ids(ont_ids, bla_reg_ids, 'acronym')
        bla_stLevels = arhfs.get_attributes_for_list_of_ids(ont_ids, bla_reg_ids, 'st_level')

        # remove a region from plots
        if region_to_remove is not None:
            # TODO remove counts from parent region

            remv_idx = bla_reg_names.index(region_to_remove)
            del(bla_reg_names[remv_idx])
            del(bla_reg_ids[remv_idx])
            remv_idx_x_axis = x_axis_labels.index(region_to_remove)
            del(xorder[remv_idx_x_axis])
            del(x_axis_labels[remv_idx_x_axis])


        bladf = andf.loc[andf['reg_id'].isin(bla_reg_ids)]

        BLAONLYDF = bladf.loc[bladf['reg_id'] == 295]
        BLAONLYDF['reactivation'] = list(BLAONLYDF['nBoth'].values/BLAONLYDF['nGFP'].values)
        BLAONLYDF = BLAONLYDF.sort_values('reactivation')
        print('group-cohort_byAnMean', BLAONLYDF.groupby(['cohort','group' ])['reactivation'].mean(numeric_only=True).to_dict())
        print('groupMean', BLAONLYDF.groupby(['group' ])['reactivation'].mean(numeric_only=True).to_dict())

        sumBLAdf = BLAONLYDF.groupby(['group', 'cohort']).sum(numeric_only=True)
        sumBLAdf['reactivation'] = sumBLAdf['nBoth'].values/sumBLAdf['nGFP'].values
        print('bySumMean', sumBLAdf['reactivation'].to_dict()) # better to calc by an mean instead

        # find animals with low gfp/area ratios
        bladf = bladf.assign(nGfpPerArea = bladf['nGFP']/bladf['reg_area']).sort_values('nGfpPerArea')
        


        # PLOT
        #################################################################
        plotdf = bladf.assign(
            groupsex = bladf.group + '-' + bladf.sex,
            groupcohort = bladf.group + '-' + bladf.cohort,
            groupcohortsex = bladf.group + '-' + bladf.cohort + '-' + bladf.sex,
            reactivationDapi = bladf.nBoth/bladf.nDapi,
            reactivationZif = bladf.nBoth/bladf.nZif,
            reactivationGFP = bladf.nBoth/bladf.nGFP,
        )

        
        for hue_var in hue_vars:
            hue_order, palette = get_palette_by_hue(hue_var)
            # get n animals per group
            group_ns = get_n_animals_per_group(plotdf, hue_var, hue_order)
            
            #################################################################
            # plot 
            for y_var in YVARS:
                fig,ax = plt.subplots(figsize=(18,6))
                sp = sns.swarmplot(
                    data=plotdf, x=x_var, y=y_var, hue=hue_var,
                    order=xorder, hue_order=hue_order, palette=dict(zip(hue_order, ['k']*len(hue_order))),
                    dodge=True, legend=False, ax=ax,
                )
                # bp = sns.boxplot(
                #     data=anbla_sumdfs, x=x_var, y=y_var, hue=hue_var,
                #     order=xorder, hue_order=hue_order, palette=palette,
                #     dodge=True, ax=ax, boxprops={'facecolor':'none'}, saturation=1, showfliers=False,
                # )
                barplot = sns.barplot(
                    data=plotdf, x=x_var, y=y_var, hue=hue_var,
                    order=xorder, hue_order=hue_order, palette=palette,
                    dodge=True, ax=ax, 
                    edgecolor='white',
                    # edgecolor=".5", facecolor=(0, 0, 0, 0), linewidth=3,
                )

                if hue_var not in ['groupcohortsex', 'groupcohort', 'groupsex']:
                    # p values
                    # oneway anova for signficance for each group for each x value
                    significance_dict = {x:0 for x in xorder}
                    for x_val in xorder:
                        # get a df of just this x val
                        adf = plotdf[plotdf[x_var] == x_val]
                        # get list of values for each group at this yvar
                        grp_yvals = [adf[adf[hue_var]==hv][y_var].values for hv in hue_order] 

                        # get pvals between groups if only 2 hue_vars
                        if len(hue_order) == 2:
                            pval = scipy.stats.f_oneway(*grp_yvals).pvalue
                            significance_dict[x_val] = pval
                            # print(f'{x_val} --> {pval}, {[np.mean(arr) for arr in grp_yvals]}')
                        
                        # if more groups, need to calculate significance between each
                        else:
                            pval_dict = {i:[] for i in range(len(grp_yvals)-1)}
                            other_vals = grp_yvals.copy()
                            for grpi, grp_vals in enumerate(grp_yvals):
                                if len(other_vals) == 1: break
                                other_vals.pop(0)
                                pvals = [scipy.stats.f_oneway(grp_vals, ov).pvalue for ov in other_vals]
                                pval_dict[grpi] = pvals
                            significance_dict[x_val] = pval_dict

                    # annotate plot with pvals
                    def map_pval(pv):
                        ''' map a pval to a char and color '''
                        pval_char = 'ns'
                        for k,v in pval_char_map.items():
                            if pv < k:
                                pval_char = v
                        pval_str_color = pval_char_color_map[pval_char]
                        return pval_char, pval_str_color
                    pval_char_map = {1:'ns', 0.1:'*', 0.051:'**', 0.011: '***'} # map pval to char (val) if less than (key)
                    pval_char_color_map = {'ns':'#bababa', '*':'#404040', '**':'#f4a582', '***':'#ca0020'}
                    ypos = plotdf[y_var].max() * 1.01
                    ygrp_decay = plotdf[y_var].max() / 23.33 #0.03 # decrease y of sig line for each drawing
                    xcenters = np.linspace(-0.3, 0.3, len(hue_order)) #[-0.3, -0.1, 0.1, 0.3]
                    
                    if len(hue_order) == 2:
                        for sdi, sd in enumerate(significance_dict):
                            pval_char, pval_str_color = map_pval(significance_dict[sd])
                            pval_str = f'p={round(significance_dict[sd], 4)}'
                            plt.text(x=sdi, y=ypos, s=pval_str, c=pval_str_color, horizontalalignment='center')
                    else:
                        for sdi, sd in enumerate(significance_dict):
                            sig_bar_counter = 0
                            for grpi, pvals in significance_dict[sd].items():
                                for pvi, pv in enumerate(pvals):
                                    # get centers of each hue-split group for each x value
                                    xi = xcenters[grpi] + sdi
                                    xf = xcenters[grpi+1+pvi] + sdi
                                    yi = ypos - (sig_bar_counter*ygrp_decay)
                                    sig_bar_counter+=1

                                    pval_char, pval_str_color = map_pval(pv)

                                    # plot line connecting groups being compared
                                    plt.plot([xi,xf], [yi,yi], c='k')

                                    # plot pval as char
                                    pval_str = f'{round(pv, 2)}'
                                    plt.text(x=(xi+xf)/2, y=yi+(ygrp_decay/6), s=pval_str, c=pval_str_color, horizontalalignment='center')
                                


                # axis params
                if any([len(ll) > 6 for ll in x_axis_labels]): # ROTATE_X_LABELS if they are long
                    ax.set_xticks(list(range(len(x_axis_labels))),x_axis_labels, ha='left', rotation=-45)
                else:
                    ax.set_xticks(list(range(len(x_axis_labels))),x_axis_labels)
                ax.set_title(y_var)
                if y_var == 'reactivationGFP':
                    ax.set_ylim(0, 0.7)

                handles, labels = [lhl[:len(hue_order)] for lhl in ax.get_legend_handles_labels()]
                labels = [f'{labels[i]} (n={group_ns[i]})' for i in range(len(hue_order))]
                ax.legend(handles, labels, bbox_to_anchor=(1.12, 1.05))
                
                g_outname = f"{plot_outname_base}_by{hue_var.upper()}-{y_var}.svg"
                if SAVE_GRAPHS_SRA:
                    fig.savefig(os.path.join(graph_outdir, g_outname), bbox_inches='tight', dpi=300)
                plt.show()



        # plot animal_id as hue
        if bool(0):
        ###################################################################################################################################
            this_y_var = 'reactivationGFP'
            hue_var = 'group'
            hue_order=['FC', 'FC+EXT']
            palette=dict(zip(hue_order, ['red', 'blue']))
            fig,axs = plt.subplots(1,3,figsize=(24,8))
            fig.autofmt_xdate(rotation=-45, ha='center')
            for grp_i, group in enumerate(hue_order):
                ax = axs[grp_i]
                anpdf = plotdf[plotdf[hue_var] == group]
                anpdf = anpdf.replace(dict(zip(xorder, range(len(xorder))))).sort_values('region_name')
                lp = sns.lineplot(data=anpdf, x=x_var, y=this_y_var, ax=ax, color='k')
                lp2 = sns.lineplot(data=anpdf, x=x_var, y=this_y_var, ax=axs[2], color=palette[group])
                sp = sns.lineplot(data=anpdf, x=x_var, y=this_y_var, hue='animal_id', ax=ax)
                ax.set_xticks(list(range(len(x_axis_labels))),x_axis_labels)
                # handles, labels = [lhl[:len(hue_order)] for lhl in ax.get_legend_handles_labels()]
                ax.legend(bbox_to_anchor=(1.01, 1.05))
                ax.set_ylim(0,0.4)
                ax.set_title(group)
                # plt.xticks(rotation=-90) 

            axs[2].set_xticks(list(range(len(x_axis_labels))),x_axis_labels)
            axs[2].set_ylim(0,0.4)
            axs[2].set_title('FC vs EXT')
            # plt.xticks(rotation=-90)
            fig.suptitle(f'{this_y_var} by animal')
            plt.show()


        if bool(0):
        ###################################################################################################################################
            # plot frank's counts vs freezing percent
            fz_df_path = r'D:\ReijmersLab\TEL\slides\franks_counts.xlsx'
            fzdf = pd.read_excel(fz_df_path)
            fzdf['pFreezing'] = fzdf['pFreezing']/100
            fzdf['animal_id'] = [f'TEL{dfv}' for dfv in fzdf['animal_id'].values]
            fzdf = fzdf[fzdf['animal_id'] != 'TEL42']
            fig,axs = plt.subplots(1,2)
            for grpi, grp in enumerate(['FC', 'EXT']):
                ax = axs[grpi]
                plt_df = fzdf[fzdf['group'] == grp].sort_values('pFreezing')
                g = sns.lineplot(data=plt_df, y='pFreezing', x='pBoth', ax=ax)
                ax.set_title(grp)
                ax.set_ylim(0,1)
                ax.set_xlim(0,1)
            fig.suptitle('fz data freezing vs reactivation (nzif+ngfp / ngfp)')
            plt.show()

            # plot my counts vs franks
            mydf = plotdf[plotdf['region_name'] == xorder[0]]
            mydf['pFreezing'] = fzdf['pFreezing'].values

            ccolors = ['purple', 'red']
            labels = ['ps', 'fz']
            Y_VARIABLES = [
                ['nZif','# Zif'], 
                ['nGFP', '# GFP'], 
                ['nBoth']*2, 
                ['reactivationGFP', 'pBoth'],
            ]
            plt_titles = [yv[0] for yv in Y_VARIABLES]

            fig,axs = plt.subplots(4,1, figsize=(12,10))
            for plt_i, y_vals in enumerate(Y_VARIABLES):
                ax = axs[plt_i]
                
                for dfi, df in enumerate([mydf, fzdf]):
                    pltdf = df.sort_values('animal_id')
                    this_y_var = y_vals[dfi]
                    g=sns.lineplot(data=pltdf, x='animal_id', y=this_y_var, c=ccolors[dfi], ax=ax, estimator=None, label=labels[dfi])
                    g2=sns.scatterplot(data=pltdf, x='animal_id', y=this_y_var, c=ccolors[dfi], ax=ax, label=labels[dfi])

                if plt_i == 0:
                    handles, labels = [[lhl[0]] + [lhl[2]] for lhl in ax.get_legend_handles_labels()]
                    ax.legend(handles, labels, bbox_to_anchor=(1.12, 1.05))
                else:
                    ax.get_legend().remove()
                
                if plt_i != len(Y_VARIABLES)-1:
                    ax.set_xlabel(None)

            fig.suptitle(f'fz vs ps total counts for each animal (nZif, nGFP, nReactivation, nZif+nGFP / nGFP)')
            plt.show()


            # tel30_s014_counts_path = r'D:\ReijmersLab\TEL\slides\ABBA_projects\byAnimal\TEL30\counts\TEL30_s014_3-9_region_counts_df.csv'
            # df = pd.read_csv(tel30_s014_counts_path)
            # bladf30 = df[df['reg_id'] == 295]
            # # for right side counts should be zif:45 , gfp:5, dapi: ~509, both:0
            # # instead of 252	5	0	0

            # # create a df 


            # # combine counts for all images of each animal
            # df = fdf.groupby(['region_name', 'reg_id', 'animal_id', 'group', 'sex', 'strain','st_level'], as_index=False).sum()
            # bladf = df[df['reg_id'] == 295]
            # bladf['reactivation'] = bladf['nBoth']/bladf['nGFP']
            # # calc average over groups
            # bladf.groupby('group').mean('reactivation')
            # sns.boxplot(data=bladf, x='group', y='reactivation')
            # fig,ax = plt.subplots(figsize=(15,6))
            # g = sns.barplot(data=bladf, x='animal_id', y='reactivation', ax=ax)
            # plt.show()










