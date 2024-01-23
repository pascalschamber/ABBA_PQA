from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from scipy.spatial import procrustes
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.stats import pearsonr, mannwhitneyu
import scipy.stats as stats


import math
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import utilities.utils_general as ug
import utilities.utils_plotting as up
import utilities.utils_image_processing as u2

from quantification import graphing_2023_0603 as graphing

from datetime import datetime
TODAY = datetime.now().strftime('%Y_%m%d')




def load_by_animal_reactivation_data(df_paths, ac, animals_to_remove):
    ############################################################
    # clean up data
    # animals_to_remove = ['TEL16', 'TEL36']
    # TEL36 - low zif, 'TEL16' - very high bla reactivation
    # TEL22,27 - low num regions in posterior areas
    ############################################################

    fdf = graphing.load_dfs(df_paths)
    # create a df summing all images together for each animal, preserving groupping variables
    groupbyCols = ['reg_id', 'region_name', 'st_level', 'sex', 'strain', 'group', 'animal_id']
    andf = fdf.groupby(groupbyCols,as_index=False).sum(numeric_only=True).sort_values('animal_id')
    # add cohorts
    andf['cohort'] = [ac.get_animal_cohort(anid)['cohort_name'] for anid in andf['animal_id'].values]
    andf['groupcohort'] = andf['group'] + '-' + andf['cohort']

    # convert px_area to mm2 
    
    andf = andf.assign(
        reactivation = andf['nBoth']/andf['nGFP'],
        zif_density = andf['nZif']/andf['reg_area'],
        pZif = andf['nZif']/andf['nDapi'],
        gfp_density = andf['nGFP']/andf['reg_area'],
        animal_int = andf.apply(lambda x: int(x['animal_id'][3:]), axis=1)
    )

    # remove an animal
    andf = andf.loc[~andf['animal_id'].isin(animals_to_remove)]
    # andf = andf.loc[andf['cohort'].isin(['cohort2', 'cohort3'][:1])]


    andf = andf.sort_values(['groupcohort', 'animal_int'])
    return andf

def pixel_to_mm(pixel_area, pixel_size_in_microns=0.650):
    # Convert pixel_size_in_microns to square millimeters
    pixel_area_in_mm = (pixel_size_in_microns / 1000) ** 2 * pixel_area
    return pixel_area_in_mm



def plot_num_regions_per_animal(andf):
    ont_slice = arhfs.parent_ontology_at_st_level(ont_ids, 8)
    parent_region_names = [el['name'] for el in ont_slice]
    st_order_names = arhfs.filter_regions_by_st_level(ont_ids, ont_slice)

    regs_per_an = []
    for dfn, adf in andf.groupby('animal_id'):
        adf = adf.loc[adf['nZif'] > 100]
        all_regions = adf['region_name'].unique()
        # get regions we care about
        st8_or_higher_regs = [el for el in st_order_names if el in all_regions]
        st8_only_regs = [d['name'] for d in ont_slice if d['name'] in st8_or_higher_regs]

        regs_per_an.append({
            'anid':dfn[3:], 'total_num_regions':len(all_regions), 
            'st8_or_higher': len(st8_or_higher_regs),
            'st8_only_regs': len(st8_only_regs),
        })
    regs_per_an = pd.DataFrame(regs_per_an).melt(
        id_vars=['anid'], value_vars=['total_num_regions','st8_or_higher', 'st8_only_regs'], var_name='count_type', value_name='count')


    fig,ax = plt.subplots(figsize=(10,10))
    sns.barplot(data=regs_per_an, x='anid', y='count', hue='count_type', ax=ax, dodge=False)
    ax.axhline(len(andf['region_name'].unique()), c='b')
    ax.axhline(len([el for el in andf['region_name'].unique() if el in st_order_names]), c='orange')
    ax.axhline(len([el for el in andf['region_name'].unique() if el in parent_region_names]), c='g')
    ax.legend(loc='upper right')
    plt.show()


    # inspect missing regions for potential animal exclusion
    st8_only_regs = st_order_names
    andf22 = [el for el in andf.loc[andf['animal_id'] == 'TEL22']['region_name'].unique() if el in st8_only_regs]
    andf27 = [el for el in andf.loc[andf['animal_id'] == 'TEL27']['region_name'].unique() if el in st8_only_regs]
    andf_rest = [el for el in andf.loc[~andf['animal_id'].isin(['TEL22', 'TEL27'])]['region_name'].unique() if el in st8_only_regs]
    andf22_missing = set(andf_rest).difference(set(andf22))
    andf27_missing = set(andf_rest).difference(set(andf27))
    print(len(andf22_missing), len(andf27_missing))

    
    andf22_has_parents = set(arhfs.get_st_parents(ont_ids, andf.loc[andf['animal_id'] == 'TEL22']['region_name'].unique(), 5).values())
    andf27_has_parents = set(arhfs.get_st_parents(ont_ids, andf.loc[andf['animal_id'] == 'TEL27']['region_name'].unique(), 5).values())
    andf_rest_has_parents = set(arhfs.get_st_parents(ont_ids, andf.loc[~andf['animal_id'].isin(['TEL22', 'TEL27'])]['region_name'].unique(), 5).values())
    print(len(andf22_has_parents), len(andf27_has_parents), len(andf_rest_has_parents))

    andf22_missing_parents = set(arhfs.get_st_parents(ont_ids, andf22_missing, 5).values())
    andf27_missing_parents = set(arhfs.get_st_parents(ont_ids, andf27_missing, 5).values())
    andf_rest_parents = set(arhfs.get_st_parents(ont_ids, andf_rest, 5).values())
    print(len(andf22_missing_parents), len(andf27_missing_parents), len(andf_rest_parents))

    # get resulting missing number of child regions

def verify_num_regions_per_animal_by_st_level(ont_ids, andf, parent_st_level=5, child_st_level=8):
    from collections import Counter
    possible_stlvl_regions = arhfs.get_st_parents(ont_ids, arhfs.filter_regions_by_st_level(ont_ids, arhfs.parent_ontology_at_st_level(ont_ids, child_st_level)), parent_st_level)
    possible_stlvl_regions = list(set(possible_stlvl_regions.values()))
    animal_st_level_counts = []
    
    if andf.index.name == 'animal_int': # handle pca formatted too
        andf = andf.reset_index().melt(id_vars='animal_int', var_name='region_name').rename(columns={'animal_int':'animal_id'})
    
    for anid, andff in andf.groupby('animal_id'):
        regions = andff['region_name'].unique()
        st_level5 = arhfs.get_st_parents(ont_ids, regions, parent_st_level)
        c = Counter(st_level5.values())
        animal_st_level_counts_dict={'animal_id':anid}
        for k in possible_stlvl_regions:
            if k in c:
                animal_st_level_counts_dict[k] = c[k]
            else:
                animal_st_level_counts_dict[k] = 0
        animal_st_level_counts.append(animal_st_level_counts_dict)

    animal_st_level_counts = pd.DataFrame(animal_st_level_counts)
    fig,ax=plt.subplots(figsize=(10,10))
    sns.barplot(animal_st_level_counts.melt(id_vars='animal_id'), x='variable', y='value', dodge=False,ax=ax)
    sns.stripplot(animal_st_level_counts.melt(id_vars='animal_id'), x='variable', y='value', dodge=False,ax=ax)
    ax.set_xticks(ax.get_xticks(), [t.get_text() for t in ax.get_xticklabels()], rotation=-45, ha='left')
    ax.set_yscale('symlog')
    plt.show()
    return animal_st_level_counts



def plot_byGroup_values(andf, var, region_id=295):
    palette = dict(zip(['CTX-cohort4', 'FC+EXT-cohort2', 'FC+EXT-cohort3', 'FC+EXT-cohort4', 'FC-cohort2', 'FC-cohort3', 'FC-cohort4'], 
                       ['gray', 'blue', 'purple', 'lightblue', 'red', 'orange', 'pink']))
    
    data = andf.assign(animal_int = andf['animal_int'].astype('str')).dropna(axis=0, how='any')
    sns.barplot(data=data, x='animal_int', y=var, hue='groupcohort', dodge=False, palette=palette, hue_order=palette.keys())
    plt.title(f'by animal mean: {var}')
    plt.show()

    sns.barplot(data=data.groupby('groupcohort', as_index=False).mean(), x='groupcohort', y=var, hue='groupcohort', dodge=False, palette=palette, hue_order=palette.keys())
    plt.legend(loc='lower right')
    plt.title(f'by groupcohort mean: {var}')
    plt.show()

    sns.barplot(
        data=data.loc[data['reg_id']==region_id].groupby(['groupcohort'], as_index=False).mean(numeric_only=True),
        x='reg_id', y=var, hue='groupcohort', palette=palette, hue_order=palette.keys())
    plt.legend(loc='lower right')
    plt.title(f'region: {region_id}')
    plt.show()




def PCA_format(andf, val_col='reactivation'):
    '''
    # Return a df where indices are animals, columns are brain regions and values are determined by val_col
    '''
    index_order = list(andf['animal_int'].unique()) # set index order by groupcohort and animal id 
    df = andf.pivot(index='animal_int', columns='region_name', values=val_col).dropna(axis=1, how='any').reindex(index_order)    
    # check number of brain regions before and after dropping na
    regions_before, regions_after = len(andf['region_name'].unique()), len(df.columns)
    print(f'regions before: {regions_before}, regions after:{regions_after}')
    return df

def PCA_format_animal_groups(andf):
    index_order = list(andf['animal_int'].unique())
    angroupdf = andf.groupby(['sex', 'strain', 'group', 'groupcohort', 'animal_id'], as_index=False).mean(numeric_only=True).set_index('animal_int')
    angroupdf = angroupdf.reindex(index_order)
    return angroupdf

# Define a custom scaling function
def custom_scale(corr_df):
    # # scale a df from range -1 to 1, preserving number of negative correlations
    vals = corr_df.values.flatten()
    vmin, vmax = abs(vals.min()), vals.max()
    rescaled_vals = np.zeros_like(vals)
    for i_v, v in enumerate(vals):
        if v>0:
            rescaled_vals[i_v] = v/vmax
        else:
            rescaled_vals[i_v] = v/vmin
    reshaped = np.reshape(rescaled_vals, corr_df.values.shape)
    rescaled_df = pd.DataFrame(data=reshaped, index=corr_df.index, columns=corr_df.columns)

    return rescaled_df



def scree_plots(pca, title=None):
    # compare number of components needed to explain variability
    plt.figure()
    plt.axhline(0.7, c='k', ls='--', alpha=0.5)
    plt.axhline(0.8, c='k', ls='--', alpha=0.75)
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), 'b-o')
    plt.xlim(0,25)
    plt.title(f'Scree Plot for correlation matrix {title}')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()

    

def get_top_region_by_component(loadings_df, andf, top_n):
    all_top_regions = []
    for component in loadings_df.index:
        pc_loadings = loadings_df.loc[component].abs().sort_values(ascending=False)

        # filter non relevant regions
        drop_indicies = []
        st_match = [int(list(set(andf.loc[andf['region_name']==val]['st_level'].values))[0]) for val in pc_loadings.index.to_list()]
        for rni, rn in enumerate(pc_loadings.index.to_list()):
            st_level = st_match[rni]
            if filter_region_names(rn, st_level):
                drop_indicies.append(rn)
            
        top_regions = pc_loadings.drop(drop_indicies)[:top_n].index.to_list()
        all_top_regions.extend(top_regions)

    regions_to_get = set(all_top_regions)
    print(len(regions_to_get))
    return all_top_regions

def filter_region_names(region_name, st_level):
    ''' return True if should be dropped '''
    if st_level < 8:
        return True
    for drp_name in ['tract', 'nerve', 'ventric', 'bundle', 'fiber']:
        if drp_name in region_name:
            return True
    return False



def plot_optimal_kmeans_clusters():
    # Decide on the number of clusters you want to identify. This could be based on domain knowledge, 
    # or you might use something like the elbow method to help choose a good number.
    # elbow
    # The Elbow Method is a technique for choosing the optimal number of clusters in k-means clustering. The basic idea is that you run k-means for many different numbers of clusters and calculate the within-cluster sum of squares (WCSS) for each number of clusters. The WCSS decreases with each additional cluster, but the rate of decrease tends to drop off at some point, forming an "elbow" shape in the graph. The number of clusters at the elbow is often a good choice.
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scores)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # silhouette analysis
    # Silhouette scores range from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.
    sil = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters = k).fit(scores)
        labels = kmeans.labels_
        sil.append(silhouette_score(scores, labels, metric = 'euclidean'))

    plt.plot(range(2, 11), sil)
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

def mantel_test(matrix1, matrix2):
    """
    ##########################################################
    Compute Mantel test on two distance matrices.

    # usage
    corr, p_value = mantel_test(corr_mat1, corr_mat2)
    print("Correlation: ", corr)
    print("P-value: ", p_value)
    ##########################################################
    """

    # flatten matrices and remove redundant (diagonal, lower-triangular) elements
    matrix1 = squareform(matrix1)
    matrix2 = squareform(matrix2)
    # compute correlation between matrices
    corr, p_value = pearsonr(matrix1, matrix2)
    return corr, p_value



def procrustes_test(matrix1, matrix2):
    # Procrustes analysis requires two 2D matrices. If your correlation matrices are 2D, you can use them directly. 
    # Otherwise, you need to find a way to convert them into 2D matrices (e.g., by using PCA or some other method).
    mtx1, mtx2, disparity = procrustes(matrix1, matrix2)
    print("Disparity: ", disparity)
    return disparity


'''
########################################################################################################################
FC vs EXT correlation matric comparison
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1) having directly 'clustered' the data into FC vs EXT, 
2) calculate auto-correlation matrix for each
3) Perform eigen decomposition on the correlation matrices
4) perform PCA on each one
5) compare the results
    PCA
    - Explained Variance Ratio
    - Scree Plots
    - Component loading (which original variables have a similar structure)
    - Cosine similarity (calculate some distance or similarity measure between the pairs of principal components)

    Other tests
        - Both the Mantel test and the Procrustes analysis assume that the order of the elements in the matrices is meaningful, 
        - and changing the order can lead to completely different results.
        - the Mantel test can only detect a certain type of relationship (monotonic) and other types of relationships may be missed. Procrustes analysis minimizes the sum of square differences, so it's most appropriate when the differences are normally distributed and the correlation is linear.

        Mantel and partial Mantel tests
        - The Mantel test is a permutation test of the correlation between two distance matrices
            - null hypothesis is that the two distance matrices are unrelated, 
            - while the alternative hypothesis is that the two matrices are related
        - However, Mantel tests can be flawed in the presence of spatial auto-correlation and return erroneously low p-values
        - Remember that this test doesn't take into account multiple testing, so if you are performing this test multiple times, you should adjust the p-value accordingly.

        Procrustes Analysis
        - form of statistical shape analysis used to analyse the distribution of a set of shapes
            - Procrustes analysis minimizes the difference between two data sets by adjusting the scale, rotation, and translation of one to match the other.
        - result is disparity, i.e. the sum of the square errors. 
            - If the disparity is small, it means the two matrices are similar to each other after optimally scaling, translating, and rotating them.

    




BRAIN REGIONS RELATED TO FC
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from Roy paper
Among the most robustly activated brain regions by memory encoding were the 
entorhinal cortex (EC), anterior cingulate cortex (CgA), amygdala, mediodorsal thalamus (MD), hippocampus, and para-subiculum (ParaS) (Fig. 1b, c).
Importantly, as it has been reported that these structures play a crucial role in contextual fear learning and memory22, 
this activation pattern supports the accuracy of our brain-wide activity-mapping results

# get regions containing
entorhinal, anterior cingulate, amygdala, mediodorsal thalamus, hippocampus, subiculum, field ca.
########################################################################################################################

'''

def check_ttest_assumptions(df, animal_indicies, groupdf, group_col, group_vals, val_cols):
    # Sample data
    analysis_df = df.loc[animal_indicies,:].assign(group=df.loc[animal_indicies,:].index.map(groupdf[group_col]))

    results = {}
    for component in val_cols:
        group1 = [analysis_df[analysis_df['group'] == group_vals[0]][component].to_list()][0]
        group2 = [analysis_df[analysis_df['group'] == group_vals[1]][component].to_list()][0]

        # Check Normality
        # Visually using histograms
        # plt.hist(group1, alpha=0.5, label='Group 1')
        # plt.hist(group2, alpha=0.5, label='Group 2')
        # plt.legend(loc='upper right')
        # plt.show()

        # Using Shapiro-Wilk Test
        stat1, p1 = stats.shapiro(group1)
        stat2, p2 = stats.shapiro(group2)

        grp1_gauss = 1 if p1 > 0.05 else 0
        grp2_gauss = 1 if p2 > 0.05 else 0
        # Check Equality of Variances using Levene's test
        stat, p = stats.levene(group1, group2)
        variances = 1 if p > 0.05 else 0
        results[component] = sum([grp1_gauss, grp2_gauss, variances])

        # if p1 > 0.05:
        #     print('Group 1: Data looks Gaussian (fail to reject H0)')
        # else:
        #     print('Group 1: Data does not look Gaussian (reject H0)')
        # if p2 > 0.05:
        #     print('Group 2: Data looks Gaussian (fail to reject H0)')
        # else:
        #     print('Group 2: Data does not look Gaussian (reject H0)')
        # if p > 0.05:
        #     print('Variances look equal (fail to reject H0)')
        # else:
        #     print('Variances do not look equal (reject H0)')

    res_counts, failed_cols = {'passed':0, 'failed2':0, 'failed1':0, 'failed0':0}, []
    for k,v in results.items():
        if v == 3:
            res_counts['passed']+=1
        else:
            res_counts[f'failed{v}']+=1
            failed_cols.append(k)

    return res_counts, failed_cols




def find_significant_components(pca_df, group_df, group_col, groups, pval_thresh=0.05, test_func=None, prnt=False):
    """
    Find PCA components that significantly separate groups.

    Args:
    - pca_df (pd.DataFrame): DataFrame with PCA components.
    - group_df (pd.DataFrame): DataFrame with group information.
    - group_col (str): Column name in group_df which has the group labels.

    Returns:
    - List[Tuple[str, float]]: List of significant components and their p-values.

    # Example usage:
        group_col, groups = 'group', ['FC', 'FC+EXT']
        check_assump_results, failed_cols = check_ttest_assumptions(pca_df, animal_indicies, angroupdf_both, group_col, groups, pca_df.columns.to_list())
        print(check_assump_results,'\n', failed_cols)
        significant_components = find_significant_components(pca_df, angroupdf_both, group_col, groups, pval_thresh=0.05)
        for component, p_value in significant_components:
            print(f"{component}: p={p_value:.5f}")
    """
    from scipy.stats import ttest_ind
    # Map group labels from group_df to pca_df based on index
    pca_df = pca_df.copy(deep=True)
    pca_df['group'] = pca_df.index.map(group_df[group_col])
    pca_df = pca_df.loc[pca_df['group'].isin(groups)]
    
    # Extract unique groups
    unique_groups = pca_df['group'].unique()
    
    if len(unique_groups) != 2:
        raise ValueError("The group_col should have exactly 2 unique values.")
    
    group_1, group_2 = groups
    
    significant_components, non_sig_components = [], []

    # Iterate over each component column
    for component in pca_df.columns:
        if component != 'group':  # we skip our added 'group' column
            group1_data = pca_df[pca_df['group'] == group_1][component]
            group2_data = pca_df[pca_df['group'] == group_2][component]

            if test_func is None:
                t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=bool(1))  # Two-sample T-test
            else:
                t_stat, p_value = test_func(group1_data, group2_data)

            # Check if p_value is below a certain threshold (e.g., 0.05 for 95% confidence)
            if p_value < pval_thresh:
                significant_components.append((component, p_value))
            else:
                non_sig_components.append((component, p_value))

    # Sort significant components based on p-values
    significant_components = sorted(significant_components, key=lambda x: x[1])
    non_sig_components = sorted(non_sig_components, key=lambda x: x[1])
    if prnt: print_significant_components(significant_components, non_sig_components)
    return significant_components, non_sig_components


def print_significant_components(significant_components, non_sig_components):
    print('significant components:')
    for component, p_value in significant_components:
        print(f"{component}: p={p_value:.5f}")
    print('non significant components:')
    for component, p_value in non_sig_components:
        print(f"{component}: p={p_value:.5f}")


def scale_data(df, scale_metric=None):
    if scale_metric is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
    elif scale_metric == 'min_max':
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)
    return scaled_data

def PCA_analysis(df, scaled_data, num_components, scale_metric=None):
    # scaled_data = scale_data(df, scale_metric=scale_metric)
    pca =  PCA(n_components=num_components)
    scores = pca.fit_transform(scaled_data)
    pca_df = (pd.DataFrame(scores)
        .assign(animal_int = df.index.values)
        .set_index('animal_int'))
    loadings_df = pd.DataFrame(pca.components_, columns=df.columns)
    return pca, pca_df, loadings_df

def plot_PCA(
        adf, pca_df, angroupdf, VALUE, 
        distances=[50,35,25], plot_dendrogram=True, plot_ngrouppercluster=True, plot_legend=True,
        marker_edge_color_sex = True,
        pca_component_axes=None, w_panels=None, base_figsize=None, save_path=None,
        marker_scaling=1.0, set_aspect='auto',
    ):
    ''' 
        Description:
            scatterplots of pca components and cluster dendrogram 
        ARGS:
            pca_component_axes (None, list[tup]) --> pca component axes to plot
    '''
    # adf = df.loc[animal_indicies, :].copy(deep=True)
    # angroupdf = angroupdf_both.copy(deep=True)

    component_cols, num_components = pca_df.columns.to_list(), pca_df.shape[1]
    dend_df = pca_df.assign(animal_id = adf.index.values)
    dend_df = dend_df.assign(groupcohort = dend_df['animal_id'].map(angroupdf['groupcohort']))
    Z = linkage(dend_df.drop(['animal_id', 'groupcohort'], axis=1), 'ward')
    # return Z, None, None

    # create a df to hold num animals in each group by cluster
    ngroup_per_cluster, nsex_per_cluster = [], []

    # create clusters
    for max_d in distances:
        # max_d = 2  # you can change this value to get different number of clusters
        clusters = fcluster(Z, max_d, criterion='distance')

        # now you can add these clusters back to your original dataframe
        adf['h_cluster'] = clusters
        adf[f'h_cluster_{max_d}'] = clusters
        print(max_d, sum(adf['h_cluster']), adf['h_cluster'].unique())

        pca_df2 = pd.DataFrame(pca_df).assign(
            animal_int = adf.index.values,
        ).set_index('animal_int')

        angroupdf['h_cluster'] = angroupdf.index.map(adf['h_cluster'])
        angroupdf[f'h_cluster_{max_d}'] = angroupdf.index.map(adf[f'h_cluster_{max_d}'])
        for i in component_cols:
            angroupdf[i] = angroupdf.index.map(pca_df2[i])
        angroupdf['animal_int'] = angroupdf.index.values

        # plot each animal id by cluster, colored by cluster and styled (markers) by groupcohort or fcgroup
        ################################
        num_clusters = len(np.unique(clusters))
        hue_order=['FC+EXT-cohort2', 'FC+EXT-cohort3', 'FC+EXT-cohort4', 'FC-cohort2', 'FC-cohort3', 'FC-cohort4', 'CTX-cohort4']
        palette=dict(zip(hue_order, ['#4575b4', '#74add1', '#abd9e9', '#d73027', '#fc8d59', '#fee090', '#e0e0e0'])) 
        # palette = dict(zip(['FC+EXT-cohort2', 'FC+EXT-cohort3', 'FC-cohort2', 'FC-cohort3'], ['blue', 'purple', 'red', 'orange']))

        marker_palette = dict(zip(sorted(np.unique(clusters)), ["o", "*", "s", "p",  "D", "^", 'd', 'P']*5))
        size_palette = dict(zip(sorted(np.unique(clusters)), np.array([150, 350, 135, 125, 100, 125, 100, 100]*5) * np.array([marker_scaling])))
        # marker_palette = dict(zip(range(num_clusters), [f"${i}$" for i in range(8)]))

        # configure marker edge colors for sex 
        mec_male, mec_female = 'blue', 'pink'
        blank_handle = plt.Line2D([0], [0], marker='s', color='white', markerfacecolor='white', markeredgecolor='white', label='sex', markersize=10)
        male_handle = plt.Line2D([0], [0], marker='s', color='white', markerfacecolor='white', markeredgecolor=mec_male, label='Male', markersize=10)
        female_handle = plt.Line2D([0], [0], marker='s', color='white', markerfacecolor='white', markeredgecolor=mec_female, label='Female', markersize=10)
        sex_palette = {'m': mec_male, 'f': mec_female} if marker_edge_color_sex else {'m': 'w', 'f': 'w'}
        sp_edgecolors = angroupdf['sex'].map(sex_palette).tolist()

        # create a df to hold num animals in each group by cluster
        for i in sorted(np.unique(clusters)):
            ngrp_clust, ngrp_clust_sex = {'max_d':max_d, 'h_cluster':i}, \
                {'max_d':max_d, 'h_cluster':i, 
                    'm':len(angroupdf.loc[(angroupdf['h_cluster']==i) & (angroupdf['sex']=='m')]), 'f':len(angroupdf.loc[(angroupdf['h_cluster']==i) & (angroupdf['sex']=='f')])}
            for grp in ['FC+EXT-cohort2', 'FC+EXT-cohort3', 'FC+EXT-cohort4', 'FC-cohort2', 'FC-cohort3', 'FC-cohort4', 'CTX-cohort4']:
                ngrp_clust[grp] = len(angroupdf.loc[(angroupdf['h_cluster']==i) & (angroupdf['groupcohort']==grp)])
                ngrp_clust_sex[f'{grp}-m'] = len(angroupdf.loc[(angroupdf['h_cluster']==i) & (angroupdf['groupcohort']==grp) & (angroupdf['sex']=='m')])
                ngrp_clust_sex[f'{grp}-f'] = len(angroupdf.loc[(angroupdf['h_cluster']==i) & (angroupdf['groupcohort']==grp) & (angroupdf['sex']=='f')])

            ngroup_per_cluster.append(ngrp_clust)
            nsex_per_cluster.append(ngrp_clust_sex)
            
        
            
        ######################################################
        # plot axis formatting
        w_panels = 2 if (w_panels is None) else w_panels
        pca_component_axes = [(i, i+1) for i in list(range(0, num_components, 2))][:w_panels] if pca_component_axes is None else pca_component_axes
        h_panels = math.ceil(len(pca_component_axes)/w_panels)

        
        base_figsize = np.array([16, 4]) if base_figsize is None else base_figsize
        fig_w, fig_h = base_figsize * np.array([w_panels/2, h_panels])
        fig,axs = plt.subplots(h_panels, w_panels, figsize=(fig_w, fig_h))
        for ax_i, pca_index in enumerate(pca_component_axes):
            ax = axs if w_panels == 1 else axs[ax_i] if h_panels==1 else axs[ax_i//w_panels, ax_i%w_panels]
            ax.set_aspect(set_aspect)
            ax.set_xlabel(f'PCA component {pca_index[0]}')
            ax.set_ylabel(f'PCA component {pca_index[1]}')

            plot_leg = True if tuple((ax_i//w_panels, ax_i%w_panels))==(0,w_panels-1) else False
            sp = sns.scatterplot(
                data=angroupdf, x=pca_index[0], y=pca_index[1], hue='groupcohort', palette=palette, style='h_cluster', markers=marker_palette,
                ax=ax, size='h_cluster',sizes=size_palette, edgecolors='k', hue_order=hue_order,
                edgecolor=sp_edgecolors,
                legend=plot_leg,
            )
            # add animal id annotations
            id_spc = 0.0 # annotation spacer
            for rowi, row in angroupdf.iterrows():
                x,y,an_int = row[pca_index[0]], row[pca_index[1]], row['animal_int']
                ax.annotate(f'{an_int}', (x+id_spc,y+id_spc), ha='center', va='center')

        # create legend
        leg_ax = axs if w_panels==1 else axs[1] if h_panels==1 else axs[(0,1)]
        if plot_legend:
            handles, labels = leg_ax.get_legend_handles_labels()
            handles.extend([blank_handle, male_handle, female_handle]); labels.extend(['sex', 'Male', 'Female'])
            leg_ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.3, 1.0))
        else: leg_ax.legend().remove()

        fig.suptitle(f'PCA components: {num_components}, clusters:{num_clusters}, max_d:{max_d} ({VALUE})')
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        #############################################################################
        # dendrogram and n per cluster handled seperate functions
        if plot_dendrogram:
            dend_fig = plot_dendrogram_fig(dend_df, palette, angroupdf, VALUE, Z, max_d, save_path)
        
    if plot_ngrouppercluster:
        ngrpclust_save_path = save_path[:-4]+ f'_nGroupPerCluster' if save_path is not None else None
        plot_ngroup_per_cluster(ngroup_per_cluster, palette, ngrpclust_save_path)
        ngrpclust_save_path_sex = save_path[:-4]+ f'_nSexPerCluster' if save_path is not None else None
        plot_ngroup_per_cluster(nsex_per_cluster, sex_palette, ngrpclust_save_path_sex)

        # plot nsexgroupcohort
        colors = ['#7a0177', '#ae017e', '#dd3497', '#f768a1', '#fa9fb5', '#fcc5c0', '#feebe2', 
                '#f0f9e8', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#08589e'][::-1]
        labels = ['FC+EXT-cohort2-f', 'FC+EXT-cohort3-f', 'FC+EXT-cohort4-f', 'FC-cohort2-f', 'FC-cohort3-f', 'FC-cohort4-f', 'CTX-cohort4-f', 
                    'CTX-cohort4-m', 'FC+EXT-cohort2-m', 'FC+EXT-cohort3-m', 'FC+EXT-cohort4-m', 'FC-cohort2-m', 'FC-cohort3-m', 'FC-cohort4-m'][::-1]
        palette_groupcohortsex = dict(zip(labels, colors))
        ngrpclust_save_path_sexgroup = save_path[:-4]+ f'_nSexGroupPerCluster' if save_path is not None else None
        plot_ngroup_per_cluster(nsex_per_cluster, palette_groupcohortsex, ngrpclust_save_path_sexgroup)

    return ngroup_per_cluster, nsex_per_cluster



def plot_dendrogram_fig(dend_df, palette, angroupdf, VALUE, Z, max_d, save_path):
    # Plotting dendrogram
    ################################
    # colors = dend_df['groupcohort'].map(palette).tolist()
    # color_dict = dict(zip([int(anid) for anid in dend_df['animal_id']], colors))
    color_dict = angroupdf['groupcohort'].map(palette).to_dict()
    color_dict_sex = angroupdf['sex'].map({'m': 'blue', 'f': 'pink'}).to_dict()
    print('dend color dict sex:\n', color_dict_sex)

    def llf(id):
        # Here we use the id (which is an index) to get the corresponding 'animal_id' and 'group'
        return dend_df['animal_id'].values[id]

    fig,ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title(f'Hierarchical Clustering Dendrogram ({VALUE})')
    ax.set_xlabel('sample index')
    ax.set_ylabel('distance')
    dend = dendrogram(Z, leaf_rotation=0, leaf_font_size=8.0, leaf_label_func=llf)
                    
    ax.xaxis.set_ticks_position('bottom')
    for xtick, tickline in zip(ax.xaxis.get_ticklabels(), ax.xaxis.get_major_ticks()):
        xtick.set_color(color_dict[int(xtick.get_text())])
        tickline.tick1line.set_markeredgecolor(color_dict_sex[int(xtick.get_text())])
        tickline.tick1line.set_markeredgewidth(2)
        

    ax.axhline(max_d, c='k', ls='--', alpha=0.4, zorder=0)
    ax.set_xlabel('animal_id')
    if save_path is not None: fig.savefig(save_path[:-4] + f'_dendrogram_maxD-{max_d}.svg', dpi=300, bbox_inches='tight')
    plt.show()
    return (fig,ax)



def plot_ngroup_per_cluster(ngroup_per_cluster_both, palette, save_path):
    # make a stacked barplot of num groupcohort per cluster
    ngroup_per_cluster = pd.DataFrame(ngroup_per_cluster_both)
                
    for max_d, adf in ngroup_per_cluster.groupby('max_d'):
        category_names = list(palette.keys())
        results = {}
        for clust_i, aadf in adf.groupby('h_cluster'):
            results[clust_i] = [aadf[grp].values[0] for grp in palette]
        
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = list(palette.values())

        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(True)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths

            for wsi in range(len(widths)):
                w, s, l = widths[wsi], starts[wsi], labels[wsi]
                if w > 0:
                    rects = ax.barh(l, w, left=s, height=0.5,
                                    label=colname, color=color)
                    ax.bar_label(rects, label_type='center', color='k')
        ax.set_yticks(range(1, len(results)+1), range(1, len(results)+1))
        ax.set_ylabel('cluster')
        ax.set_xticks(range(data_cum.max()+1), range(data_cum.max()+1))
        ax.set_xlabel('count')
        ax.legend(
            handles=[Patch([0], [0], color=c, ec='white', label=lbl) for lbl,c in palette.items()], 
            labels=[lbl for lbl,c in palette.items()], 
            bbox_to_anchor=(1.3, 1.0)
        )
        if save_path is not None: fig.savefig(save_path[:-4]+ f'_maxD-{max_d}.svg', dpi=300, bbox_inches='tight')
        plt.show()

########################################################################################################################
# conda activate autoquant

# PARAMS
########################################################################################################################

if __name__ == '__main__':
    
    ont = arhfs.Ontology()
    ont_ids = ont.ont_ids
    names_dict = ont.names_dict #dict(zip([d['name'] for d in ont_ids.values()], ont_ids.keys()))


    ac = AnimalsContainer()
    ac.init_animals()
    base_dir = r"C:\Users\pasca\Box\Reijmers Lab\Frank\TEL Project\quantification\2023_0827_quantification" #r'D:\ReijmersLab\TEL\slides\quant_data\counts'
    # base_dir = r'C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\network_analysis\data'
    animal_info_df = pd.read_excel(r'D:\ReijmersLab\TEL\slides\8-8-22 TEL Project - FC vs FC+EXT TetTag Experiment.xlsx')
    output_data_dir = r'C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\network_analysis\data'

    df_paths = [os.path.join(base_dir, fn) for fn in [

        # '2023_0716_quant_data_cohort2_t-75-350-0.45-550-150.csv',
        # '2023_0713_quant_data_cohort3_t-120-350-0.45-250-150.csv',
        # '2023_0716_all_data.csv'
        # '2023_0719_quant_data_cohort2_t-75-350-0.45-550-150.csv',
        # '2023_0719_quant_data_cohort3_t-120-350-0.45-250-150.csv',
        
        # '2023_0806_quant_data_cohort2_t-75-350-0.45-550-100-5-85-4-31.csv',
        # '2023_0806_quant_data_cohort3_t-75-350-0.45-360-100-5-85-4-31.csv',
        r"2023_0827_151059_quant_data_byAnimal.csv"

    ]]
    # graph_outdir = ug.verify_outputdir(os.path.join(r'C:\Users\pasca\Box\Reijmers Lab\Frank\TEL Project\quantification', f'{graphing.today}_quantification'))


    # LOAD DATA
    ########################################################################################################################
    # get the reactivation by brain regions for each animal
    animals_to_remove = [
        'TEL16', 'TEL36', 'TEL42', 'TEL49', 'TEL52', 'TEL61', # animals removed for low zif/gfp expression
        'TEL56', # removed b/c ctx animal went through extinction
        'TEL22', 'TEL27',  # removed for missing posterior regions (cerebellar nuclei, cere cortex, pons, medula)
        # 'TEL60', 'TEL41', 'TEL17' # removed for missing cerebellar nuclei
    ]
    andf = load_by_animal_reactivation_data(df_paths, ac, animals_to_remove)
    # andf_nReg_df = verify_num_regions_per_animal_by_st_level(ont_ids, andf, parent_st_level=5, child_st_level=8)


    SAVE_CORR = bool(1)
    VALUE = ['reactivation', 'zif_density', 'gfp_density'][2]
    ST_LEVEL = 5
    MAX_ST_LEVEL = 5 # only get lvls upto to this value
    output_fig_dir = ug.verify_outputdir(os.path.join(
        r'C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\network_analysis', 
        f'figures_{VALUE}', f'{TODAY}_st{ST_LEVEL}'))

    # plot_num_regions_per_animal(andf)
    # plot_byGroup_values(andf, 'zif_density')
    
    if bool(1):
        # df is your data frame where index is animal_id, columns are brain regions and each value is the engram reactivation/zif_density...
        print(VALUE)
        df = PCA_format(andf, val_col=VALUE)
        # nReg_df = verify_num_regions_per_animal_by_st_level(ont_ids, df, parent_st_level=5, child_st_level=ST_LEVEL)

        angroupdf_fc, angroupdf_ext, angroupdf_ctx = PCA_format_animal_groups(andf.loc[andf['group'] == 'FC']), PCA_format_animal_groups(andf.loc[andf['group'] == 'FC+EXT']), PCA_format_animal_groups(andf.loc[andf['group'] == 'CTX'])
        cohort2_anids = angroupdf_fc.loc[angroupdf_fc['groupcohort'] == 'FC-cohort2'].index.to_list() + angroupdf_ext.loc[angroupdf_ext['groupcohort'] == 'FC+EXT-cohort2'].index.to_list()
        cohort3_anids = angroupdf_fc.loc[angroupdf_fc['groupcohort'] == 'FC-cohort3'].index.to_list() + angroupdf_ext.loc[angroupdf_ext['groupcohort'] == 'FC+EXT-cohort3'].index.to_list()
        cohort4_anids = angroupdf_fc.loc[angroupdf_fc['groupcohort'] == 'FC-cohort4'].index.to_list() + angroupdf_ext.loc[angroupdf_ext['groupcohort'] == 'FC+EXT-cohort4'].index.to_list()
        CTX_anids     = angroupdf_ctx.loc[angroupdf_ctx['groupcohort'] == 'CTX-cohort4'].index.to_list()
        
        df_fc, df_ext, df_ctx = df.loc[df.index.isin(angroupdf_fc.index)], df.loc[df.index.isin(angroupdf_ext.index)], df.loc[df.index.isin(angroupdf_ctx.index)]
        ont_slice = arhfs.parent_ontology_at_st_level(ont_ids, ST_LEVEL)
        st_order_names = arhfs.filter_regions_by_st_level(ont_ids, ont_slice, max_st_lvl=MAX_ST_LEVEL)
        ordered_index_fc = [el for el in st_order_names if el in df_fc.columns.to_list()]
        ordered_index_ext = [el for el in st_order_names if el in df_ext.columns.to_list()]
        ordered_index_ctx = [el for el in st_order_names if el in df_ctx.columns.to_list()]
        df_fc, df_ext, df_ctx = df_fc[ordered_index_fc], df_ext[ordered_index_ext], df_ctx[ordered_index_ctx]
        print(f"final number of regions: {[el.shape for el in [df_fc, df_ext, df_ctx]]}")

        # calculate correlation for each dataset
        corr_fc, corr_ext, corr_ctx = df_fc.corr(), df_ext.corr(), df_ctx.corr()
        if SAVE_CORR:
            df_fc.to_excel(os.path.join(output_data_dir, f'{TODAY}_{VALUE}_st{ST_LEVEL}_fc.xlsx'))
            df_ext.to_excel(os.path.join(output_data_dir, f'{TODAY}_{VALUE}_st{ST_LEVEL}_ext.xlsx'))
            df_ctx.to_excel(os.path.join(output_data_dir, f'{TODAY}_{VALUE}_st{ST_LEVEL}_ctx.xlsx'))
    
    if bool(0):
        # t-SNE clustering
        from sklearn.manifold import TSNE
        import hdbscan

        animal_indicies = cohort2_anids + cohort3_anids + cohort4_anids + CTX_anids
        angroupdf_both = pd.concat([angroupdf_fc, angroupdf_ext, angroupdf_ctx]).loc[animal_indicies,:]
        
        data_df = df.loc[animal_indicies, :].copy(deep=True)

        scale_metric = [None, 'min_max'][0]
        scaled_data = scale_data(data_df, scale_metric=scale_metric)
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(scaled_data)

        # Extract clusters using HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_samples=1, gen_min_span_tree=True, min_cluster_size=5)
        cluster_labels = clusterer.fit_predict(tsne_results)
        print(f'clusters: {np.unique(cluster_labels)}')
       


        # Plot t-SNE results with clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='Spectral', s=50)
        plt.colorbar(scatter)
        plt.title('t-SNE with HDBSCAN clusters')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.show()
    

    if bool(0):

        # PCA by st_level 5 region
        ########################################################################################################################
        parent_level_colormap = {
            'Cortical subplate': '#3288bd', 'Isocortex': '#66c2a5', 'Olfactory areas': '#abdda4', 'Pallidum': '#e6f598', 'Hippocampal formation': '#ffffbf', 'Striatum': '#fee08b', 'Thalamus': '#fdae61', 'Hypothalamus': '#f46d43', 'Midbrain': '#d53e4f',
            'Pons':'#980043', 'Medulla':'#756bb1', 'Cerebellar cortex':'#d9d9d9', 
            # 'Cerebellar nuclei':'#969696',
            }
        
        ch2parent = arhfs.get_st_parents(ont_ids, df_fc.columns.to_list(), 5)
        parent2ch = {p:[v for v in ch2parent if ch2parent[v] == p] for p in parent_level_colormap}
        

        component_map = {
            'Cortical subplate': {'num_components': 3, 'max_d': 14},
            'Isocortex': {'num_components': 10, 'max_d': 35},
            'Olfactory areas': {'num_components': 3, 'max_d': 12},
            'Pallidum': {'num_components': 4, 'max_d': 8},
            'Hippocampal formation': {'num_components': 5, 'max_d': 14},
            'Striatum': {'num_components': 4, 'max_d': 14},
            'Thalamus': {'num_components': 6, 'max_d': 20},
            'Hypothalamus': {'num_components': 6, 'max_d': 14},
            'Midbrain': {'num_components': 8, 'max_d': 14},
            'Pons': {'num_components': 6, 'max_d': 13},
            'Medulla': {'num_components': 5, 'max_d': 14},
            'Cerebellar cortex': {'num_components': 3, 'max_d': 8},
            'Cerebellar nuclei': {'num_components': 3, 'max_d': 14}
        }
        
        for PARENT_REGION in list(parent_level_colormap.keys())[:]:
            # get regions at a specific st_level
            children_to_get = parent2ch[PARENT_REGION]
            df_fc_slice, df_ext_slice, df_ctx_slice = df_fc.loc[:, children_to_get], df_ext.loc[:, children_to_get], df_ctx.loc[:, children_to_get]
            df_slice = df.loc[:, children_to_get]


            # perform pca
            SCALE_INDEPENDENT = bool(1)
            num_components, max_distance = component_map[PARENT_REGION]['num_components'], component_map[PARENT_REGION]['max_d']
            scale_metric = [None, 'min_max'][0]
            grouped_an_inds = [cohort2_anids, cohort3_anids, cohort4_anids+CTX_anids]
            animal_indicies = ug.flatten_list(grouped_an_inds)
            angroupdf_both = pd.concat([angroupdf_fc, angroupdf_ext, angroupdf_ctx]).loc[animal_indicies,:]

            scaled_data = np.vstack([scale_data(df_slice.loc[chrt_ids,:], scale_metric=scale_metric) for chrt_ids in grouped_an_inds]) if SCALE_INDEPENDENT else scale_data(df_slice.loc[animal_indicies, :], scale_metric=scale_metric)
            data_df = df_slice.loc[animal_indicies, :].copy(deep=True)
            scaled_df = pd.DataFrame(scaled_data, index=data_df.index, columns=data_df.columns)
            pca, pca_df, loadings_df = PCA_analysis(data_df, scaled_data, num_components, scale_metric=scale_metric)
            # scree_plots(pca, title=PARENT_REGION)

            ngroup_per_cluster, nsex_per_cluster = plot_PCA(
                data_df, pca_df.iloc[:,:], angroupdf_both, f'{VALUE} {PARENT_REGION}', distances=[max_distance], 
                pca_component_axes=[(1,0)], #[(0, 3)], # , (4, 5), (6, 7), (8, 9), (10, 11)],
                w_panels=1, base_figsize=np.array([12,6]), marker_scaling=1.5, set_aspect='auto', # base_figsize=np.array([16,8])
                plot_dendrogram= bool(0), plot_ngrouppercluster = bool(0), plot_legend= bool(0), marker_edge_color_sex = bool(0),
                save_path = [None, os.path.join(output_fig_dir, f'{TODAY}_{VALUE}_{PARENT_REGION}_SCALE_INDEPENDENT-{SCALE_INDEPENDENT}_PCA_components_scatterplot.svg')][1],
            )
        





    # Perform PCA
    ########################################################################################################################
    if bool(0):
        # optimized after looking at scree plots without directly setting number of components
        animal_indicies = cohort2_anids + cohort3_anids + cohort4_anids #+ CTX_anids
        angroupdf_both = pd.concat([angroupdf_fc, angroupdf_ext, angroupdf_ctx]).loc[animal_indicies,:]
        scale_metric = [None, 'min_max'][0]
        num_components = 10
        scaler, pca, pca_df, loadings_df = PCA_analysis(df.loc[animal_indicies, :], num_components, scale_metric=scale_metric)
        scree_plots(pca)

        ngroup_per_cluster, nsex_per_cluster, dend_fig = plot_PCA(
            df.loc[animal_indicies, :], pca_df.iloc[:,:], angroupdf_both, 
            VALUE+' ALL', distances=[45], plot_dendrogram=True, 
            pca_component_axes=None, #[(0, 3)], # , (4, 5), (6, 7), (8, 9), (10, 11)],
            save_path = None #os.path.join(output_fig_dir, f'{TODAY}_{VALUE}_PCA_components_scatterplot.svg'),
        )
    if bool(0):
        
        ######################################
        # if scaling each cohort independently, better to use min/max scale_metric
        # if separating fc vs ctx scale independent, fc vs ext scale together
        SCALE_INDEPENDENT = bool(0)
        num_components = 10
        scale_metric = [None, 'min_max'][0]
        grouped_an_inds = [cohort2_anids, cohort3_anids, cohort4_anids]#+CTX_anids]
        animal_indicies = ug.flatten_list(grouped_an_inds)
        angroupdf_both = pd.concat([angroupdf_fc, angroupdf_ext, angroupdf_ctx]).loc[animal_indicies,:]

        scaled_data = np.vstack([scale_data(df.loc[chrt_ids,:], scale_metric=scale_metric) for chrt_ids in grouped_an_inds]) if SCALE_INDEPENDENT else scale_data(df.loc[animal_indicies, :], scale_metric=scale_metric)
        data_df = df.loc[animal_indicies, :].copy(deep=True)
        scaled_df = pd.DataFrame(scaled_data, index=data_df.index, columns=data_df.columns)
        pca, pca_df, loadings_df = PCA_analysis(data_df, scaled_data, num_components, scale_metric=scale_metric)
        scree_plots(pca)


        grp_col_idx = 1
        group_col, groups = [('groupcohort', ['FC-cohort2', 'FC+EXT-cohort2']), ('group', ['FC', 'FC+EXT']), ('group', ['FC', 'CTX'])][grp_col_idx]
        check_assump_results, failed_cols = check_ttest_assumptions(pca_df, animal_indicies, angroupdf_both, group_col, groups, pca_df.columns.to_list())
        print(check_assump_results,'failed cols:', failed_cols)
        significant_components, non_sig_components = find_significant_components(pca_df, angroupdf_both, group_col, groups, pval_thresh=0.05, test_func=mannwhitneyu, prnt=bool(1))

        
        plot_comps = (1,0); ppca_inds = [slice(None,None), [*plot_comps], [4,1,0]][0]
        ngroup_per_cluster, nsex_per_cluster, dend_fig = plot_PCA(
            data_df, pca_df.iloc[:,ppca_inds], angroupdf_both,
            VALUE+' ALL', distances=[10], plot_dendrogram=True, w_panels=1, base_figsize=np.array([16,8]),
            pca_component_axes=[plot_comps], #[(0, 3)], # , (4, 5), (6, 7), (8, 9), (10, 11)],
            save_path = None #os.path.join(output_fig_dir, f'{TODAY}_{VALUE}_PCA_components_scatterplot.svg'),
        )
        
        ######################################
        # extract loadings of components of interest
        # Positive loadings indicate that a variable and a principal component are positively correlated whereas negative loadings indicate a negative correlation
        components_of_interest = [0,1,4]
        topN = 10
        loadings_of_interest = [loadings_df.iloc[ci, :] for ci in components_of_interest]
        top_loadings_of_interest = [loi.sort_values(ascending=bool(0)).iloc[:topN] for loi in loadings_of_interest]
        top_loadings_of_interest_negative = [loi.sort_values(ascending=bool(1)).iloc[:topN] for loi in loadings_of_interest]
        
        palette_topdf = dict(zip(['FC+EXT', 'FC', 'CTX'][:-1], ['blue', 'red', 'gray'][:-1]))
        palette_sp = {k:'k' for k,v in palette_topdf.items()}
        fig,axs = plt.subplots(1,3, figsize=(20,8), sharey=True)
        for axi, ax in enumerate(axs):
            ax.set_title(f'top10 regions PC-{components_of_interest[axi]}')
            tl = top_loadings_of_interest_negative[axi].index
            top_df = data_df.loc[:, tl].copy(deep=True)
            top_df = top_df.assign(
                group       = top_df.index.map(angroupdf_both['group']),
                # groupcohort = top_df.index.map(angroupdf_both['groupcohort']),
            )
            data_comp = (top_df
                        .melt(id_vars='group')
                        .groupby(['group', 'region_name'], as_index=False).mean()
                        .pivot(index=['region_name'], columns=['group'])
                        .droplevel(0, axis=1))
            data_comp = data_comp.assign(ratio=data_comp['FC']/data_comp['FC+EXT'])


            bp = sns.barplot(data=data_comp.reset_index(), x='region_name', y='ratio', ax=ax, order=tl.to_list(), 
                             edgecolor=".5", facecolor=(0, 0, 0, 0), linewidth=3,)


            # ax2 = ax.twinx()
            # sp = sns.stripplot(data=top_df.melt(id_vars='group'), x='region_name', y='value', hue='group', ax=ax2, dodge=True, 
            #                    order=tl.to_list(), palette=palette_topdf, hue_order=palette_topdf.keys(), legend=False)
            # ax2.set_yscale('log')
            ax.axhline(1.0, c='gray', ls='--', alpha=0.7)
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), ha='left', rotation=-45)
        plt.show()





        
        

        
        

        



        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(pca_df.values)
        clusters = kmeans.predict(pca_df.values)
        clusters_df = pca_df.assign(k_clust = clusters, animal_int=animal_indicies, groupcohort=angroupdf_both.loc[animal_indicies,'groupcohort'].values)


       
        
        # scaler_fc, scaler_ext = StandardScaler(), StandardScaler()
        # scaled_data_fc, scaled_data_ext = scaler_fc.fit_transform(df_fc), scaler_ext.fit_transform(df_ext)
        # pca_fc, pca_ext = PCA(n_components=num_components), PCA(n_components=num_components)
        # scores_fc,scores_ext = pca_fc.fit_transform(scaled_data_fc), pca_ext.fit_transform(scaled_data_ext)
        # pca_df_fc, pca_df_ext = pd.DataFrame(scores_fc), pd.DataFrame(scores_ext)
        # # scree_plots(pca_fc, pca_ext)
        # # Get the loadings
        # loadings_fc, loadings_ext = pca_fc.components_, pca_ext.components_
        # loadings_fc_df, loadings_ext_df = pd.DataFrame(loadings_fc, columns=df_fc.columns), pd.DataFrame(loadings_ext, columns=df_ext.columns)

        # ngroup_per_cluster_FC = plot_PCA(df_fc, pca_df_fc, angroupdf_fc, VALUE+' FC', distances=[50,35,25], num_components=4, plot_dendrogram=True)
        # ngroup_per_cluster_EXT = plot_PCA(df_ext, pca_df_ext, angroupdf_ext, VALUE+' EXT', distances=[50,35,25], num_components=4, plot_dendrogram=True)

        # # heatmaps of the region contributions to each component
        # fig,ax = plt.subplots(figsize=(10,8))
        # hm = sns.heatmap(loadings_fc_df.transpose(), annot=False, cmap='Spectral', ax=ax)
        # plt.show()

        # fig,ax = plt.subplots(figsize=(10,8))
        # hm = sns.heatmap(loadings_ext_df.transpose(), annot=False, cmap='Spectral', ax=ax)
        # plt.show()


        # # get the top regions contributing to each component for each group
        # top_regions_fc, top_regions_ext = get_top_region_by_component(loadings_fc_df, andf, 10), get_top_region_by_component(loadings_ext_df, andf, 10)
        # print(f'number of unique top regions: {len(set(top_regions_fc + top_regions_ext))}')


        # top_loadings = []
        # for cluster in average_scores.index:
        #     pc = top_pcs.loc[cluster]
        #     pc_loadings = loadings_opt.loc[pc].sort_values(ascending=False)  # pc[2:] to remove the 'PC' prefix
        #     # get the reactivations of these regions for each cluster
        #     cluster_animals = angroupdf.loc[angroupdf['h_cluster_35'] == cluster].index.to_list()
        #     reactivations = df.loc[df.index.isin(cluster_animals)]
        #     reactivations = reactivations[regions_to_get]
        #     top_loadings.append(reactivations)

        #     print(f"Top contributing brain regions for cluster {cluster} (driven by {pc}):")
        #     print(pc_loadings.drop(drop_indicies).head(5))

        #     # sns.clustermap(pc_loadings[:25])

        # cluster_df = pd.concat([pd.DataFrame(top_loadings[i].mean()).rename(columns={0:f'{i+1}'}) for i in range(len(top_loadings))], axis=1)





    # generate the linkage matrix
    ########################################################################################################################
        





    



    if bool(0):
        [375, 295, 31, 502, 362, 909]
        ["Ammon's horn", 'bla', 'Anterior cingulate area', 'Subiculum', 'Mediodorsal nucleus of thalamus', 'Entorhinal area']
        []

        # compare brain regions that are influential to pca # TODO
        avg_drop_cols = [
            'sex',
            'strain',
            'group',
            'groupcohort',
            'animal_id',
            'reg_id',
            'reg_area',
            'nDapi',
            'nZif',
            'nGFP',
            'nBoth',
            'reactivation',
            'h_cluster',
            # 0,
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # 7,
            # 8,
            # 9,
            # 10,
            # 11,
            # 12,
            # 13,
            # 14,
            # 15,
            'animal_int',
            'h_cluster_50',
            'h_cluster_45',
            'h_cluster_40',
            # 'h_cluster_35'
        ]

        # 1. Calculate the average PCA scores for each PC within each cluster
        average_scores = angroupdf.drop(avg_drop_cols, axis=1).groupby('h_cluster_35').mean()
        # 2. Identify the PCs that contribute most to each cluster
        # This just identifies the column with the highest absolute value for each row/cluster
        top_pcs = average_scores.abs().idxmax(axis=1)
        # 3. Look at the PCA loadings for those PCs to see which brain regions/variables contribute most
        # This will require going back to the PCA model object itself. Assuming that's called 'pca_model':
        loadings_opt = pd.DataFrame(
            pca_optimal.components_, 
            columns=df.drop(['h_cluster', 'h_cluster_50', 'h_cluster_45', 'h_cluster_40', 'h_cluster_35'], axis=1).columns
        )

        def filter_region_names(region_name, st_level):
            ''' return True if should be dropped '''
            if st_level < 8:
                return True
            for drp_name in ['tract', 'nerve', 'ventric', 'bundle', 'fiber']:
                if drp_name in region_name:
                    return True
            return False

        all_top_regions = []
        for cluster in average_scores.index:
            pc = top_pcs.loc[cluster]
            pc_loadings = loadings_opt.loc[pc].sort_values(ascending=False)  # pc[2:] to remove the 'PC' prefix

            # filter non relevant regions
            drop_indicies = []
            st_match = [int(list(set(andf.loc[andf['region_name'] ==val]['st_level'].values))[0]) for val in pc_loadings.index.to_list()]
            for rni, rn in enumerate(pc_loadings.index.to_list()):
                st_level = st_match[rni]
                if filter_region_names(rn, st_level):
                    drop_indicies.append(rn)
                
            top_regions = pc_loadings.drop(drop_indicies)[:10].index.to_list()
            all_top_regions.extend(top_regions)

        regions_to_get = set(all_top_regions)
        print(len(regions_to_get))


        top_loadings = []
        for cluster in average_scores.index:
            pc = top_pcs.loc[cluster]
            pc_loadings = loadings_opt.loc[pc].sort_values(ascending=False)  # pc[2:] to remove the 'PC' prefix
            # get the reactivations of these regions for each cluster
            cluster_animals = angroupdf.loc[angroupdf['h_cluster_35'] == cluster].index.to_list()
            reactivations = df.loc[df.index.isin(cluster_animals)]
            reactivations = reactivations[regions_to_get]
            top_loadings.append(reactivations)

            print(f"Top contributing brain regions for cluster {cluster} (driven by {pc}):")
            print(pc_loadings.drop(drop_indicies).head(5))

            # sns.clustermap(pc_loadings[:25])

        cluster_df = pd.concat([pd.DataFrame(top_loadings[i].mean()).rename(columns={0:f'{i+1}'}) for i in range(len(top_loadings))], axis=1)

        fig,ax = plt.subplots(figsize=(10,8))
        sns.heatmap(cluster_df, cmap='seismic', ax=ax)
        ax.set_xlabel('cluster')
        fig.savefig('reactivations_by_cluster_for_h_cluster_35.svg', dpi=300, bbox_inches='tight')
        plt.show()

        myg = sns.clustermap(
            cluster_df.T,
            figsize=(12,12),
            # row_cluster=False,
            dendrogram_ratio=(.1, .2),
            cbar_pos=(-0.1, .42, .03, .4),
        )
        # Rotate x-axis labels
        plt.setp(myg.ax_heatmap.get_xticklabels(), rotation=-45, ha='left')

        # Rotate y-axis labels
        plt.setp(myg.ax_heatmap.get_yticklabels(), rotation=0)

        plt.setp(myg.ax_heatmap.set_ylabel('cluster'))
        myg.savefig('reactivations_by_cluster_for_h_cluster_35.svg', dpi=300, bbox_inches='tight')
        plt.show()



        # # correlation = pd.DataFrame(top_loadings[0].mean()).corrwith(pd.DataFrame(top_loadings[1].mean()), axis=1)
        # # correlation = top_loadings[0].corrwith(top_loadings[1], axis=1)
        df1, df2 = pd.DataFrame(top_loadings[0].mean()).rename(columns={0:'1'}), pd.DataFrame(top_loadings[1].mean()).rename(columns={0:'2'})
        cluster_df = pd.concat([df1, df2], axis=1)
        cluster_df = cluster_df.assign(diff=cluster_df.loc[:,'1']-cluster_df.loc[:,'2'])
        print(f"cluster 1 higher: {sum(cluster_df['diff']>0)}, cluster 2 higher: {sum(cluster_df['diff']<0)}")
        # # We use '.T' to transpose the dataframes because we want variables as columns and observations as rows
        # df1_t = df1.T
        # df2_t = df2.T

        # # Compute pairwise correlation between all regions
        # corr_matrix = df1_t.corrwith(df2_t, axis=0)

        # # Convert correlation series to dataframe for the heatmap
        # corr_df = pd.DataFrame(corr_matrix, columns=['correlation'])

        # # Make a 2D grid
        # corr_df['x'] = corr_df.index
        # corr_df['y'] = corr_df.index

        # # Pivot the DataFrame to get it in a grid format
        # pivot = corr_df.pivot(index='y', columns='x', values='correlation')

        # # Plotting
        # plt.figure(figsize=(12,10))
        # sns.heatmap(pivot, cmap="coolwarm", annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        # plt.show()
        fig,ax = plt.subplots(figsize=(10,8))
        mask = np.triu(np.ones_like(top_loadings[1].corr(), dtype=bool))
        sns.heatmap(top_loadings[0].corr() - top_loadings[1].corr()  , center=0.0, cmap='seismic', mask=mask, ax=ax)
        ax.set_xticks(
            ax.get_xticks(), ax.get_xticklabels(),
            ha='left', rotation=-45)
        plt.show()









    # by plotting hm of different clusters' animals
    if bool(0):
        for clust_i, adf in df.groupby('h_cluster'):
            pass


        hmdf = df.drop('h_cluster', axis=1)
        print('inital shape before filters', hmdf.shape)
        # filter out fiber/nerve regions
        # drop st_level names that contain specific phrases
        drop_st_names = ['tract', 'nerve', 'ventric', 'bundle', 'fiber']
        dropped_cols = ['h_cluster_50', 'h_cluster_45', 'h_cluster_40', 'h_cluster_35']
        for drpname in drop_st_names:
            cols = [c for c in hmdf.columns.to_list() if drpname in c]
            dropped_cols.extend(cols)
            # rdbg = rdbg.loc[~rdbg['region_name'].str.contains(drpname)]
        
        hmdf = hmdf.drop(dropped_cols, axis=1)
        print('len after drop regionname filters', hmdf.shape)

        # assign st_level as an index
        # get stlevel for each col (region) by indexing into andf (og data)
        st_match = [int(list(set(andf.loc[andf['region_name'] ==acol]['st_level'].values))[0]) for acol in hmdf.columns.to_list()]
        tuples = list(zip(st_match, hmdf.columns))
        # Convert the list to a MultiIndex
        hmdf.columns = pd.MultiIndex.from_tuples(tuples, names=['st_level', 'region_name'])
        hmdf = hmdf.sort_index(axis=1, level='st_level', ascending=True)


        used_columns = hmdf.columns.get_level_values('st_level').astype('int').isin(range(8,12))
        demodf = hmdf.loc[:, used_columns]
        myg = sns.clustermap(
            demodf.corr(),
            figsize=(30, 30),
            # row_cluster=False,
            # dendrogram_ratio=(.1, .2),
            cbar_pos=(0, .2, .03, .4),
        )
        myg.ax_row_dendrogram.remove()
        myg.savefig('region_corr_clustermap_stLvls8-11.svg', dpi=300, bbox_inches='tight')
        plt.show()


        # # Load the brain networks example dataset
        # demodf = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

        # # Select a subset of the networks
        # used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
        # used_columns = (demodf.columns.get_level_values("network")
        #                         .astype(int)
        #                         .isin(used_networks))
        # demodf = demodf.loc[:, used_columns]

        # # Create a categorical palette to identify the networks
        # network_pal = sns.husl_palette(8, s=.45)
        # network_lut = dict(zip(map(str, used_networks), network_pal))

        # # Convert the palette to vectors that will be drawn on the side of the matrix
        # networks = demodf.columns.get_level_values("network")
        # network_colors = pd.Series(networks, index=demodf.columns).map(network_lut)

        # # Draw the full plot
        # g = sns.clustermap(demodf.corr(), center=0, cmap="vlag",
        #                 row_colors=network_colors, col_colors=network_colors,
        #                 dendrogram_ratio=(.1, .2),
        #                 cbar_pos=(.02, .32, .03, .2),
        #                 linewidths=.75, figsize=(12, 13))

        # g.ax_row_dendrogram.remove()



    # TODO
    # how can I deconstruct the pca do identify which regions are contributing to each cluster?







    # plot_optimal_kmeans_clusters()


    # for num_clusters in [2,3]:
    #     # num_clusters = 4  # replace with your number

    #     # Create a KMeans object
    #     kmeans = KMeans(n_clusters=num_clusters)

    #     # Fit the KMeans object to your data
    #     kmeans.fit(scores)

    #     # Get the cluster assignments for each data point
    #     clusters = kmeans.predict(scores)

    #     # You can now add these cluster assignments back to your original dataframe:, also add groupping information
        
        
    #     # add these grouppings to the score df
    #     pca_df = pd.DataFrame(scores).assign(
    #         animal_int = df.index.values,
    #     )
    #     pca_df = pca_df.assign(
    #         cluster = clusters,
    #         group = pca_df['animal_int'].map(angroupdf['group']),
    #         groupcohort = pca_df['animal_int'].map(angroupdf['groupcohort']),
    #         sex = pca_df['animal_int'].map(angroupdf['sex']),
    #         strain = pca_df['animal_int'].map(angroupdf['strain']),
    #     )



    #     # plot each animal id by cluster, colored by cluster and styled (markers) by groupcohort or fcgroup
    #     palette = dict(zip(['FC+EXT-cohort2', 'FC+EXT-cohort3', 'FC-cohort2', 'FC-cohort3'], ['blue', 'purple', 'red', 'orange']))
    #     marker_palette = dict(zip(range(num_clusters), ["o", "*", "s", "p",  "D", "^"]))
    #     size_palette = dict(zip(range(num_clusters), [100, 175, 125, 125, 100, 125]))
    #     # marker_palette = dict(zip(range(num_clusters), [f"${i}$" for i in range(8)]))
        
    #     fig,axs = plt.subplots(1, num_components//2, figsize=(16,4))
    #     for pca_index in list(range(0, num_components, 2))[:]:
    #         ax = axs[pca_index//2]
    #         ax.set_xlabel(f'PCA component {pca_index}')
    #         ax.set_ylabel(f'PCA component {pca_index+1}')

            
    #         sp = sns.scatterplot(
    #             data=pca_df, x=pca_index, y=pca_index+1, hue='groupcohort', palette=palette, style='cluster', markers=marker_palette,
    #             ax=ax, size='cluster',sizes=size_palette, edgecolors='k', 
    #             legend=False if pca_index+2 != num_components else True,
    #         )
    #         # add animal id annotations
    #         for rowi, row in pca_df.iterrows():
    #             x,y,an_int = row[pca_index], row[pca_index+1], row['animal_int']
    #             ax.annotate(f'{an_int}', (x,y),)

    #     # create legend
    #     ax.legend(bbox_to_anchor = (1.2, 1.0))

    #     fig.suptitle(f'PCA components: {num_components}, num_clusters:{num_clusters}')
    #     plt.tight_layout()
    #     plt.show()




    #     # plot bla reactivation by animal and assigned cluster
    #     bladf = andf.loc[andf['reg_id']==295]
    #     bladf = bladf.assign(cluster = bladf['animal_int'].map(pca_df.set_index('animal_int')['cluster']))
    #     sns.barplot(data=bladf, x='animal_int', y='reactivation', order=index_order, hue='cluster', dodge=False)


        

    # def generate_adjacency_matricies(corr_df):
    #     ''' 
    #         Generate adjacency matricies for network plotting, as described in https://www.frontiersin.org/articles/10.3389/fnbeh.2022.907707/full
    #             Where the following params were used:
    #                 Correlations were filtered by statistical significance and a false discovery rate of 5%
    #                 To ensure that the thresholding parameters did not bias network analyses, additional adjacency matrices were generated using 
    #                     either more (r > 0.95; α = 0.0005) or less (r > 0.8; α = 0.05) conservative thresholds.
    #     '''
    #     # Define the threshold parameters
    #     false_discovery_rate = 0.05
    #     pval_corr = 0.05
    #     thresholds = [(0.75, 0.05), (0.8, 0.005), (0.85, 0.0005)]

    #     p_values_fc = corr_df.corr(method=lambda x, y: pearsonr(x, y)[1])  # p-values
    #     # Flatten the correlation matrix and p_values
    #     correlations_flat = corr_df.values.flatten()
    #     p_values_flat = p_values_fc.values.flatten()
    #     # Apply a significance filter and a false discovery rate of 5% to the correlations
    #     _, p_values_fdr_bh, _, _ = multipletests(p_values_flat, alpha=false_discovery_rate, method='fdr_bh')
    #     # Reshape the corrected p-values back into the shape of the correlation matrix
    #     p_values_fdr_bh = p_values_fdr_bh.reshape(corr_df.shape)
    #     # Create a mask for significant correlations
    #     mask_significant = p_values_fdr_bh < pval_corr
    #     print(f"% significant: {mask_significant.sum()/mask_significant.size}")
    #     # Apply the mask to the correlation matrix
    #     correlation_matrix_significant = corr_df.where(mask_significant)

    #     # Create a dictionary to store the adjacency matrices
    #     adjacency_matrices = {}

    #     # Generate the adjacency matrices
    #     for r, alpha in thresholds:
    #         mask = (correlation_matrix_significant > r) & (p_values_fdr_bh < alpha)
    #         adjacency_matrix = correlation_matrix_significant.where(mask)
    #         adjacency_matrix = adjacency_matrix.notnull().astype('int')  # Convert to binary (0, 1)
    #         adjacency_matrices[f"r_{r}_alpha_{alpha}"] = adjacency_matrix
        
    #     return adjacency_matrices, correlation_matrix_significant


    # def plot_networks_nx(adjacency_matrices, correlation_matrix_significant): 
    #     # Visualize the correlation matrix as adjacency matrix as network graph
    #     fig, axs = plt.subplots(1, 3, figsize=(30, 5))
    #     for ax, (key, adjacency_matrix) in zip(axs, adjacency_matrices.items()):
    #         # convert correlation matric to adjacency matrix
    #         graph = nx.from_pandas_adjacency(adjacency_matrix)

    #         # Determine edge colors
    #         edge_colors = [correlation_matrix_significant[edge[0]][edge[1]] > 0 for edge in graph.edges()]

    #         # Convert boolean values to colors
    #         edge_colors = ['blue' if color else 'red' for color in edge_colors]

    #         # Determine edge widths
    #         edge_widths = [abs(correlation_matrix_significant[edge[0]][edge[1]]) * 10 for edge in graph.edges()]

    #         # Draw the graph
    #         nx.draw_networkx(graph, ax=ax, with_labels=False, node_size=20, edge_color=edge_colors, width=edge_widths)
        

    #     # calculate significance
    #     significance_stats = {}
    #     for ax, (key, adjacency_matrix) in zip(axs, adjacency_matrices.items()):
    #         n_significant = adjacency_matrix.sum().sum() / 2  # Each edge is counted twice in an adjacency matrix
    #         n_positive = (correlation_matrix_significant > 0).sum().sum() / 2
    #         n_negative = (correlation_matrix_significant < 0).sum().sum() / 2
    #         significance_stats[key] = {"n_significant": n_significant, "n_positive": n_positive, "n_negative": n_negative}
    #         as_str = ', '.join([f'{k}: {v}' for k,v in significance_stats[key].items()])
    #         ax.set_title(f'{key}  {as_str}')

    #     plt.show()
    #     return significance_stats



    # adjacency_matrices_fc, correlation_matrix_significant_fc = generate_adjacency_matricies(corr_fc)
    # adjacency_matrices_ext, correlation_matrix_significant_ext = generate_adjacency_matricies(corr_ext)

    # significance_stats_fc = plot_networks_nx(adjacency_matrices_fc, correlation_matrix_significant_fc)
    # significance_stats_ext = plot_networks_nx(adjacency_matrices_ext, correlation_matrix_significant_ext)



    # print(f"correlation ranges. FC: ({corr_fc.min().min()}, {corr_fc.max().max()}), EXT: ({corr_ext.min().min()}, {corr_ext.max().max()})")
    # corr_fc_scaled, corr_ext_scaled = custom_scale(corr_fc),  custom_scale(corr_ext)
    # print(f"num og corr. above/below 0: {(corr_fc.values.flatten()>0).sum()}, {(corr_fc.values.flatten()<0).sum()}")
    # print(f"num scaled corr. above/below 0: {(corr_fc_scaled.values.flatten()>0).sum()}, {(corr_fc_scaled.values.flatten()<0).sum()}")
    # print(f"old sum > 0: {(corr_fc[corr_fc<0]).sum().sum()}, <0: {(corr_fc[corr_fc>0]).sum().sum()}")
    # print(f"new sum > 0: {(corr_fc_scaled[corr_fc_scaled<0]).sum().sum()}, <0: {(corr_fc_scaled[corr_fc_scaled>0]).sum().sum()}")

    # corr_diff = (corr_fc_scaled+1) - (corr_ext_scaled+1)
    # corr_diff_rescaled = custom_scale(corr_diff)


    # # group ids by parent structure, into one list
    # st_order_ids = [el[1] for el in arhfs.gather_ids_by_st_level(ont_ids, as_dict=False)]
    # set_st_order = []
    # for el in st_order_ids:
    #     if el not in set_st_order:
    #         set_st_order.append(el)
    # st_order_names = arhfs.get_attributes_for_list_of_ids(ont_ids, set_st_order, 'name')

    # ordered_index = [el for el in st_order_names if el in corr_diff_rescaled.index.values]
    # assert all(corr_diff_rescaled.index.values == corr_diff_rescaled.columns.values) # check index == columns labels
    # corr_diff_rescaled = corr_diff_rescaled.reindex(index=ordered_index, columns=ordered_index)


    # sns.heatmap(corr_diff)
    # plt.show()

    # fig,ax = plt.subplots(figsize=(18,20))
    # sns.heatmap(corr_diff_rescaled, ax=ax)
    # # set ticks to label brain regions at st level 5
    # to_label = arhfs.get_attributes_for_list_of_ids(ont_ids, arhfs.gather_ids_by_st_level(ont_ids, as_dict=True)[5], 'name') + ['Basolateral amygdalar nucleus']
    # tick_indicies, tick_labels = [], []
    # for areg_i, areg in enumerate(ordered_index):
    #     if areg in to_label:
    #         tick_indicies.append(areg_i+0.5)
    #         tick_labels.append(areg)
    # # ax.set_xticks(
    # ax.set_xticks(tick_indicies, tick_labels)
    # ax.set_yticks(tick_indicies, tick_labels)
    # plt.show()
