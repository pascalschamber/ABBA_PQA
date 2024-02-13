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
import shutil
import pyclesperanto_prototype as cle
import matplotlib.patches as mpatches
from pathlib import Path
from timeit import default_timer as dt
from numba import jit, types
from numba.typed import List

import concurrent
import concurrent.futures
import multiprocessing

# package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
import utilities.utils_general as ug
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import utilities.utils_plotting as up


# point in polygon functions
##################################################################################################################
@nb.jit(nopython=True)
def nb_process_polygons(singles, multis, centroids):
    """
    # localize centroids to regions, first checking singles (no excluded sub-regions) then multipolygons (have subregions to exclude)
    # input numba dtypes:
        # ListType[ListType[array(float64, 2d, C)]], ListType[ListType[ListType[array(float64, 2d, C)]]], array(float64, 2d, C)
    """
    num_polys = len(singles) # singles and multis must have same length
    num_centroids = len(centroids)
    results = np.full(num_centroids, -1)
    for i in nb.prange(num_centroids):
        centroid_x, centroid_y = centroids[i]
        found = False
        for polyi in range(num_polys): 
            if found: break
            these_single_polys, these_multi_polys = singles[polyi], multis[polyi]
            
            if len(these_single_polys)>0:
                for spi in range(len(these_single_polys)):
                    if nb_point_in_single(these_single_polys[spi], centroid_x, centroid_y):
                        results[i] = polyi
                        found = True
                        break
            if found: break
            if len(these_multi_polys)>0:
                for mpi in range(len(these_multi_polys)):
                    if nb_point_in_multi(these_multi_polys[mpi], centroid_x, centroid_y):
                        results[i] = polyi
                        found = True
                        break
    return results

@nb.jit(nopython=True)
def ray_casting(x, y, centroid_x, centroid_y):
    # determine if point in polygon using ray casting algorithm
    num_vertices = len(x)
    j = num_vertices - 1
    odd_nodes = False
    for i in range(num_vertices):
        if ((y[i] < centroid_y and y[j] >= centroid_y) or (y[j] < centroid_y and y[i] >= centroid_y)) and (x[i] <= centroid_x or x[j] <= centroid_x):
            odd_nodes ^= (x[i] + (centroid_y - y[i]) / (y[j] - y[i]) * (x[j] - x[i]) < centroid_x) # The ^ operator does a binary xor. a ^ b will return a value with only the bits set in a or in b but not both
        j = i
    return odd_nodes

@nb.jit(nopython=True)
def nb_point_in_single(single, centroid_x, centroid_y):
    main_x, main_y = single.T[0], single.T[1]
    if ray_casting(main_x, main_y, centroid_x, centroid_y):
        return True
    return False
    
@nb.jit(nopython=True)
def nb_point_in_multi(multi, centroid_x, centroid_y):
    main_polygon, interiors = multi[0], multi[1:]
    main_x, main_y = main_polygon.T[0], main_polygon.T[1]
    if ray_casting(main_x, main_y, centroid_x, centroid_y):
        in_interior = False
        for interior in interiors:
            interior_x, interior_y = interior.T[0], interior.T[1]
            if ray_casting(interior_x, interior_y, centroid_x, centroid_y):
                in_interior = True
                break
        if not in_interior:
            return True
    return False

def get_empty_single_nb():
    # Create a typed list for the outermost level of single polygons (ListType[array(float64, 2d, C)])
    inner_array_type = types.Array(dtype=types.float64, ndim=2, layout='C')
    return List.empty_list(inner_array_type)
    
def get_empty_multi_nb():
    # Create a typed list for the outermost level of multipolygons (ListType[ListType[array(float64, 2d, C)]])
    inner_array_type = types.Array(dtype=types.float64, ndim=2, layout='C')
    return List.empty_list(types.ListType(inner_array_type))
    


# handling geojson objects
##################################################################################################################
def load_geojson_objects(geojson_path):
    # read in qupath .geojson file and parse detections to get only valid annotations
    with open(geojson_path) as f:
        allobjects = geojson.load(f)
    allfeatures = allobjects['features']
    
    # get all detections that have an 'ID' (regions), except the root which doesn't have an atlas id
    all_objs = [obj for obj in allfeatures if ('measurements' in obj['properties'].keys()) and ('ID' in obj['properties']['measurements'])]
    
    return all_objs


def extract_polyregions(geojson_objs, ont):
    # build region polygon objects, parsing single and multipolygons from qupaths geojson file
    all_polys = []
    for obj_i, anobj in enumerate(geojson_objs):
        print_str = ""

        aPoly = regionPoly(obj_i=obj_i, anobj=anobj, st_level=get_st_lvl_from_regid(anobj['properties']['measurements']['ID'], ont))
        aPoly.region_name = ont.ont_ids[aPoly.obj_id]['name'] if aPoly.obj_id != 997 else 'root'

        coord_arrs, num_poly_coords = get_coordinates(anobj)
        print_str += f"{obj_i} ({anobj['geometry']['type']}) --> n: {num_poly_coords}\n"
        try:
            maybeError = aPoly.unpack_feature_polygons(coord_arrs)
            aPoly.extract_info()
            
            if maybeError is not None: 
                raise(ValueError)
            
        except ValueError:
            print(aPoly.print_str)
            raise ValueError(f'ERROR:\n{print_str}\n{maybeError}')
        
        finally:
            all_polys.append(aPoly)
    return all_polys

def get_st_lvl_from_regid(regid, ont):
    st_lvl = ont.ont_ids[regid]['st_level'] if regid in ont.ont_ids else 0
    return st_lvl

def separate_polytypes(polyObjs):
    """
    # build initial numba format for singles and multipolygons
        ListType[ListType[array(float64, 2d, C)]], ListType[ListType[ListType[array(float64, 2d, C)]]]
    # numba input needs to be consistent, such that:
        # each element of input list is all polys for a given region
        # this element will contain two lists, one for singles and one for multis
            # can be multiple multipolygons so nested again, is empty if none
                # for each element of the inner list the first element is coords to include and rest are excluded
    """
    # get regionPoly objects as dicts and sort by st_lvl
    polyObj_dicts = sorted([apoly.to_dict() for apoly in polyObjs], key=lambda x: x['st_level'], reverse=True) 
    
    # gather numba capatible outer list and info for each poly
    nb_singles, nb_multis, infos = List(), List(), []
    for di, d in enumerate(polyObj_dicts):
        infos.append({k:v for k,v in d.items() if k not in ['singles', 'multis']}) # get info_only (not arrays)
        nb_singles.append(d['singles'])
        nb_multis.append(d['multis'])

    assert len(nb_singles) == len(nb_multis) == len(infos), f"lengths do not match {len(nb_singles)}, {len(nb_multis)}, {len(infos)}"
    return nb_singles, nb_multis, infos

def get_coordinates(aFeature):
    polyType = aFeature['geometry']['type']
    
    if polyType == 'MultiPolygon':
        coord_arrs = [np.array(el) for el in aFeature['geometry']['coordinates']]
    elif polyType == 'Polygon':
        coord_arrs = [np.array(aFeature['geometry']['coordinates'])]
    else: 
        raise ValueError(f"{polyType} is not handled")
    num_poly_coords = len(coord_arrs)
    
    return coord_arrs, num_poly_coords



class regionPoly:
    """
        object to hold a collection of polys so know the following:
            regions to extract or ignore
            area
            numba compatible, so can return list of polys (in order so it can be proc'd in order)
        polygon coords are stored in a dict where keys are interally used poly indices (in order)
            value is a dict: 
                arr: np.ndarray, 
                polytype: main_poly, exteriors, interiors
        there are three types of polys in qupath abba annotatins
            a region ('exteriors')
            a multipolygon with regions to exclude ('main')
            those regions to exclude from this main polygon ('interiors')
        Interface with nuclei localization
            this is the order based on what we can infer from the different types of polygons
                doing it this way avoids potential issue when checking multipolys first where it could be in excluded region
                and also in a later non-multipoly
            check against non-multipolys, if inside can be sure not going to be excluded
            then, check multipolys, if inside and not in excluded areas we're good
    ARGS
        GO_FAST (bool): if true skip superfoulous calculations (debugging)
    """

    def __init__(self, obj_i, anobj, st_level, GO_FAST=True):
        self.valid_polytypes = ['main', 'exteriors', 'interiors']
        self.count_polytypes = None # used for debugging
        self.obj_i = obj_i # this is index of polygon collection in geojson file (i.e. poly_index)
        self.st_level = st_level
        self.region_area = None
        self.region_extent = None # store bounding box coordinates
        self.poly_count = 0
        self.poly_arrays = {} # store polys here
        self.GO_FAST = GO_FAST
        self.ingest_obj(anobj)


    def ingest_obj(self, anobj):
        self.anobj = anobj # might want to not store to conserve memory if possible
        self.region_name = None
        self.obj_id = anobj['properties']['measurements']['ID']  # this is id of region in ontology
        # self.obj_names = ', '.join(anobj['properties']['classification']['names'])
        obj_names = anobj['properties']['classification']['names']
        self.reg_side = str(obj_names[0])
        self.acronym = str(obj_names[1])
        self.geometry_type = anobj['geometry']['type']

    def __str__(self):
        prt_str = ''
        get_attrs = ['obj_id', 'acronym', 'reg_side', '', 'region_name',  '', 'geometry_type','', 'region_area']
        for attr in get_attrs:
            if len(attr) == 0: prt_str+='\n'
            else: prt_str += f"{attr}: {getattr(self, attr)} "
        return prt_str+'\n'
    
    def add_poly(self, poly_arr, polyType):
        if polyType not in self.valid_polytypes: 
            raise ValueError(f'polyType ({polyType}) must be one of {self.valid_polytypes}')
        assert self.poly_count not in self.poly_arrays, f"key ({self.poly_count}) should not already exist"

        # add poly to dict
        self.poly_arrays[self.poly_count] = {'arr':poly_arr, 'polyType':polyType}
        self.poly_count += 1
        
    def unpack_feature_polygons(self, coord_arrs):
        # store exteriors and interiors, where exteriors are regions to include, and interiors are regions to exclude
        # store shapes for debugging
        self.ragged_shapes = []
        error = None
        self.print_str = ""
        self.numba_format = [] # list of dicts, where dict includes indicies to include/ exclude for each coord array
        
        try:
            for arr_i, arr in enumerate(coord_arrs):
                nb_dict = {'include':None, 'exclude':None}
                if arr.ndim == 3: # array of shape e.g. (1, 2, nPoints)
                    if arr.shape[0]>1: raise ValueError(f"{arr.shape} is not handled")
                    for v_idx in range(arr.shape[0]): # but could handle it if implemented here
                        valid_arr = arr[v_idx]
                        assert valid_arr.ndim ==2, f"{valid_arr.shape} is not 2d"
                        if arr_i == 0: # handle case where only a single poly
                            self.add_poly(valid_arr, 'main')
                        else:
                            self.add_poly(valid_arr, 'exteriors')
                        nb_dict['include'] = valid_arr
                        
                elif arr.ndim == 1: # handle ragged arrays 
                    unpacked_arrs = [np.array(el) for el in arr]
                    self.ragged_shapes.append(f"{arr.shape} --> {[a.shape for a in unpacked_arrs]}")
                    
                    # check_all_2dim
                    unpacked_shapes = [el.shape for el in unpacked_arrs]
                    assert all([len(el)==2 for el in unpacked_shapes]), f"{unpacked_shapes} contains non 2d arrays"
                    nb_dict['exclude'] = []
                    # split into main body and interiors
                    for i, el in enumerate(unpacked_arrs):
                        if i == 0:
                            self.add_poly(el, 'main')
                            nb_dict['include'] = el
                        else:
                            self.add_poly(el, 'interiors')
                            nb_dict['exclude'].append(el)
                else: 
                    raise ValueError(f'this should not happen, arr ndim: {arr.ndim}')
                self.numba_format.append(nb_dict)
            

            self.print_str += f"{self.obj_i} ({self.geometry_type}) --> n: {len(coord_arrs)}\n"
            for astr in self.ragged_shapes:
                self.print_str += f"\tragged arr: {astr}\n"
            
        except Exception as e:
            error = e

        finally: # append additional info
            for arr in coord_arrs:
                self.print_str += f'{arr.shape}'
                for ca_i, ca in enumerate(coord_arrs):
                    if ca.ndim == 1:
                        for el in ca:
                            self.print_str += f'\t {np.array(el).shape}'
        
        return error
    
    def prepare_numba_input(self, polygonCollection):
        # where polygonCollection is a list of dict with keys for include and exclude 
        # and include is an array and  exclude is list of coord arrays
        nb_include, nb_exclude = [], []
        for d in polygonCollection:
            exclude = nb.typed.List(d['exclude']) if d['exclude'] is not None else None
            nb_include.append(d['include']), nb_exclude.append(exclude)
        return nb.typed.List(nb_include), nb.typed.List(nb_exclude)

    def extract_info(self):
        self.region_area = self.get_total_area(self.poly_arrays)
        self.region_extent = self.get_region_extent(self.poly_arrays)
        self.all_obj_atlas_coords = self.get_atlas_coords(self.anobj) # NEW 2023_0808, not required
        if not self.GO_FAST:
            self.get_count_polytypes()
    

    def get_count_polytypes(self):
        # count num polys of each type, for plotting/debuging
        self.count_polytypes = dict(zip(self.valid_polytypes, [0]*len(self.valid_polytypes)))
        for pi, p_arr in self.poly_arrays.items():
            polyType = p_arr['polyType']
            self.count_polytypes[polyType]+=1
            
    def get_total_area(self, poly_arrays):
        # get area of regions to include minus areas to exclude
        total_area = 0
        for poly_i, poly_dict in poly_arrays.items():
            area = polygon_area(poly_dict['arr']) 
            area *= -1 if poly_dict['polyType'] == 'interiors' else 1
            total_area += area
        return total_area
    
    def get_region_extent(self, poly_arrays):
        # get bounding box of all polygons comprising this region
        return list(get_polygons_extent([pd['arr'] for pd in poly_arrays.values()]))
    
    def get_atlas_coords(self, geojson_obj):
        # extracts the atlas coords for each region from geojson file if present
        atlas_coords_not_found = 0 # store number of coords not found
        atlas_measurements_keys = ['Atlas_X', 'Atlas_Y', 'Atlas_Z']
        obj_output = []
        measurements = geojson_obj['properties']['measurements']
        for coord_key in atlas_measurements_keys:
            if coord_key not in measurements: # check atlas coords exist
                atlas_coords_not_found += 1
                obj_output.append(None)
            else:
                obj_output.append(measurements[coord_key])
        if not self.GO_FAST:
            if atlas_coords_not_found > 0: print(f'WARN --> atlas coords not extracted (num: {atlas_coords_not_found})', flush=True)
        return obj_output
    
    def to_dict(self):
        # helper function for numba processing, prepares the polygons and extract info needed for centroid df
        polyObj_dict = {
            'poly_index':self.obj_i, 
            'reg_id':self.obj_id, 
            'st_level':self.st_level,
            'region_name':self.region_name,
            'reg_side':self.reg_side,
            'acronym':self.acronym,
            'region_area':self.region_area,
            'singles':[], 'multis':[]}
        for coord_dict in self.numba_format:
            if coord_dict['exclude'] is not None:
                polyObj_dict['multis'].append(List([coord_dict['include']] + [a for a in coord_dict['exclude']]))
            else:
                polyObj_dict['singles'].append(coord_dict['include'])
        
        polyObj_dict['singles'] = get_empty_single_nb() if len(polyObj_dict['singles']) == 0 else List(polyObj_dict['singles'])
        polyObj_dict['multis'] = get_empty_multi_nb() if len(polyObj_dict['multis']) == 0 else List(polyObj_dict['multis'])
        return polyObj_dict
    
    def to_region_df_row(self):
        # helper function to convert object to a dict that is compatible with a pandas dataframe
        # note that previously 'region_polygons' were only the first region, now that there are multiple it cannot be used the same way
            # so new implementations that need these coordinate arrays should access them through .numba_format
        atx, aty, atz = self.all_obj_atlas_coords if len(self.all_obj_atlas_coords)==3 else [np.nan]*3
        row_dict = dict(zip(
            ['poly_index', 'region_ids', 'acronym', 'region_name', 'region_sides', 'region_areas', 'region_extents', 'atlas_x', 'atlas_y', 'atlas_z'],
            [self.obj_i, self.obj_id, self.acronym, self.region_name, self.reg_side, self.region_area, self.region_extent, atx, aty, atz]
        ))
        return row_dict
        
# general utils
##############################################
def polygon_area(region_poly):
    ''' Calculates the area of a complex polygon using the shoelace formula. Calculates signed area, so abs is taken to get the actual area '''
    assert region_poly.shape[1] == 2 and region_poly.ndim==2, f'error --> region poly shape {region_poly.shape}'
    Xs, Ys = region_poly[:,1], region_poly[:,0]
    area = 0.5 * abs(sum(Xs[i]*(Ys[i+1]-Ys[i-1]) for i in range(1, len(Xs)-1)) + Xs[0]*(Ys[1]-Ys[-1]) + Xs[-1]*(Ys[0]-Ys[-2]))
    return area

def pixel_to_mm(pixel_area, pixel_size_in_microns=0.650):
    # Convert pixel_size_in_microns to square millimeters
    pixel_area_in_mm = (pixel_size_in_microns / 1000) ** 2 * pixel_area
    return pixel_area_in_mm

def pixel_to_um(pixel_area, pixel_size_in_microns=0.650):
    # Convert pixel_size_in_microns to square um
    pixel_area_in_um = (pixel_size_in_microns / 1000) ** 2 * pixel_area * 1e6
    return pixel_area_in_um

def get_polygons_extent(poly_list):
    minimum_x, minimum_y, maximum_x, maximum_y = np.inf, np.inf, 0, 0
    
    for arr in poly_list:
        if len(arr) == 0:
            continue
        max_x = np.max(arr[:,0])
        max_y = np.max(arr[:,1])
        min_x = np.min(arr[:,0])
        min_y = np.min(arr[:,1])
        if max_x > maximum_x:
            maximum_x = max_x
        if max_y > maximum_y:
            maximum_y = max_y
        if min_x < minimum_x:
            minimum_x = min_x
        if min_y < minimum_y:
            minimum_y = min_y
    return minimum_x, minimum_y, maximum_x, maximum_y


# plotting helpers
##############################################
def plot_polygons(apoly, plot_points=None, fig_title=None, show_legend=True, region_outlines=None, region_label=False, SAVE_PATH=None, limit_plot=True, fig_size=(20,20), fc_alpha=0.2, pad=200):
    fig,ax = plt.subplots(1, figsize=fig_size)       
    # support for passing a single array of coordinates too
    if isinstance(apoly, regionPoly):
        minimum_x, minimum_y, maximum_x, maximum_y = get_polygons_extent([pd['arr'] for pd in apoly.poly_arrays.values()])
    elif isinstance(apoly, np.ndarray):
        minimum_x, minimum_y, maximum_x, maximum_y = get_polygons_extent([apoly])
        poly_dict = {0:{'arr':apoly, 'polyType':'main'}}
    else: 
        raise ValueError()

    # limit plot area
    x_min, x_max = minimum_x-pad, maximum_x+pad
    y_min, y_max = minimum_y-pad, maximum_y+pad
    if limit_plot:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


    # plot outlines of all regions provided, expects a list of polyRegions
    if region_outlines is not None: # put these down first so current polylist is clear        
        for ap in region_outlines:
            for pi,pd in ap.poly_arrays.items():
                centroid = pd['arr'].mean(axis=0)
                if not(x_min <= centroid[0] <= x_max and y_min <= centroid[1] <= y_max):
                    continue
                ax.add_patch(patches.Polygon(pd['arr'], linewidth=0.5, edgecolor='k', alpha=0.15, facecolor='none'))
                if region_label is False: continue
                # add region label
                c_lbl = ap.anobj['properties']['classification']['names'][1]
                bbox_props = dict(boxstyle="square,pad=0.3", fc="black", ec="black", lw=1, alpha=0.7)
                ax.text(centroid[0], centroid[1], c_lbl, ha='right', va='top', bbox=bbox_props, c='w', fontsize='xx-small')

    # Add the patch to the Axes
    blue_rgba = (0.0, 0.0, 1.0, fc_alpha)    # with opacity
    green_rgba = (0.0, 1.0, 0.0, fc_alpha)  
    red_rgba = (1.0, 0.0, 0.0, fc_alpha)    
    yellow_rgba = (1.0, 1.0, 0.0, 0.5)
    palette2 = {'exteriors':yellow_rgba, 'main': yellow_rgba, 'interiors':(1.0, 0.0, 0.0, 0.5)}
    palette2_fc = {'exteriors':blue_rgba, 'main': green_rgba, 'interiors':red_rgba}
    
    to_add = apoly.poly_arrays if isinstance(apoly, regionPoly) else poly_dict
    for pi, pd in to_add.items():
        arr, ptype = pd['arr'], pd['polyType']
        ec, fc = palette2[ptype], palette2_fc[ptype]
        polygon = patches.Polygon(arr, linewidth=0.5, edgecolor=ec, facecolor=fc)
        ax.add_patch(polygon)

    if plot_points is not None:
        ax.scatter(plot_points[:,0], plot_points[:,1], c='k', s=25, marker='x')
    

    if fig_title is not None: plt.title(fig_title)
    if show_legend:
        # Create a legend using the provided color palettes
        legend_handles= [mpatches.Patch(edgecolor=palette2[label], facecolor=palette2_fc[label], label=label, alpha=0.2) for label in palette2]
        ax.legend(handles=legend_handles, loc='best')
    
    
    plt.gca().invert_yaxis()
    if SAVE_PATH is None:
        plt.show()
    else:
        outname = ug.clean_filename(fig_title, replace_spaces='_') + '.svg'
        fig.savefig(os.path.join(SAVE_PATH, outname), dpi=300, bbox_inches='tight')
        plt.close()


        

# debugging
########################################################################################################
def plot_compare_region_counts(region_count_df, test_region_count_df, stain_col_names, st_lvl_max=10):
    import seaborn as sns
    # plot comparison of shared nuclei per region for two dataframes (for debugging)
    shared_reg_ids = list(set(region_count_df['reg_id'].unique()).intersection(set(test_region_count_df['reg_id'].unique())))
    shared_columns = list(set(region_count_df.columns.to_list()).intersection(set(test_region_count_df.columns.to_list())))
    df_gt = region_count_df[region_count_df['reg_id'].isin(shared_reg_ids)][shared_columns].assign(condition='gt')
    df_gt['st_level'] = df_gt['st_level'].replace('notFound', 0)
    df_test = test_region_count_df[test_region_count_df['reg_id'].isin(shared_reg_ids)][shared_columns].assign(condition='test')
    df_plot = pd.concat([df_gt, df_test], ignore_index=True)
    df_plot['st_level'] = df_plot['st_level'].replace('notFound', 0).astype('int')
    
    for cell_type in stain_col_names:
        fig,ax = fig, ax = plt.subplots(figsize=(20,10))
        bp = sns.swarmplot(    
            data=df_plot[df_plot['st_level']>st_lvl_max], x='region_name', y=cell_type, hue='condition', palette=dict(zip(['gt', 'test'], ['k','r'])), ax=ax,
            dodge=True,
        )
        ax.set_xticks(np.arange(len(ax.get_xticklabels())), ax.get_xticklabels(), rotation=-45, ha='left')
        plt.show()

def get_info_index(poly_list, info_LoD, poly_index):
    # lookup polygon by poly_index 
    inds, polys, infos = [], [], []
    for di, d in enumerate(info_LoD):
        if d['poly_index'] == poly_index:
            inds.append(di)
            polys.append(poly_list[di])
            infos.append(d)
    return inds, polys, infos

def plot_coordinates_inside_polygon(rpdf, nb_singles, infos, regionPoly_list, centroids, get_polyi=0, get_colocal_id=0 ):
    # plot nuclei inside a polygon (for debugging)
    # NOTE: only shows points that were assigned the lowest structural level since rpdf only contains these 
    print(len(rpdf[rpdf['poly_index']==get_polyi]))
    get_inds, get_polyarrs, get_info  = get_info_index(nb_singles, infos, get_polyi)
    apoly = [el for el in regionPoly_list if el.obj_i == get_polyi][0]
    get_centroids_df = rpdf[(rpdf['poly_index']==get_polyi) & (rpdf['colocal_id']==get_colocal_id)]
    centroid_coords = centroids[get_centroids_df['centroid_i'].values]
    plot_polygons(apoly, plot_points=centroid_coords, limit_plot=True) 

def plot_unassigned_coordinates(rpdf, centroids, regionPoly_list, get_polyi=0):
    # plot coordinates of nuclei that were not assigned to any region, can show a region too
    # where process_polygons_df is the output of process_polygons function after info has been added back
    get_unassigned_centroids_df = rpdf[(pd.isnull(rpdf['poly_index']))]
    unassigned_centroids = centroids[get_unassigned_centroids_df['centroid_i'].values]
    plot_polygons(regionPoly_list[get_polyi], plot_points=unassigned_centroids, limit_plot=False) 

def print_poly_list(poly_list):
    # print info for a list of regionPolys
    for pi, apoly in enumerate(poly_list):
        print(apoly.region_name)
        print(pi, f'n: {len(apoly.poly_arrays)}')
        
        if bool(1):
            for k,v in apoly.anobj['properties'].items():
                print(f'\t{k}: {v}')
        total_area = 0
        for poly_i, poly_dict in apoly.poly_arrays.items():
            ptype = poly_dict['polyType']
            coords = poly_dict['arr']
            area = polygon_area(coords)
            area *= -1 if ptype == 'interiors' else 1
            total_area += area
            print(poly_i, ptype, coords.shape, f'area: {area}')
        print(f"total area: {pixel_to_um(total_area)}\n")

def inspect_highest_num_polys(all_polys, nmin=0, nmax=5, PLOT=True, SAVE_PATH=None):
    # for each datum get the top most poly regions
    for geojson_path, apolylist in all_polys.items():
        print(geojson_path, len(apolylist))
        numpolys = {}    
        for pi, apoly in enumerate(apolylist):
            npolys = len(apoly.poly_arrays)
            numpolys[pi] = npolys

        most_ps = sorted(numpolys.items(), key=lambda item: item[1])[::-1][nmin:nmax]
        most_ps_idxs = [el[0] for el in most_ps]
        for pidx in most_ps_idxs:
            print(f'datum obj idx: {pidx}')
            print_poly_list(apolylist[pidx:pidx+1])
            if PLOT:
                apoly = apolylist[pidx]
                fig_title = f"{Path(geojson_path).stem}\n[{pidx}] {apoly.obj_names+ ', ' + str(apoly.obj_id)} - num main/exts/ints: {apoly.count_polytypes['main']}/{apoly.count_polytypes['exteriors']}/{apoly.count_polytypes['interiors']} - {apoly.geometry_type} - area: {int(apoly.total_area)}"
                plot_polygons(apoly, fig_title=fig_title, region_outlines=apolylist, SAVE_PATH=SAVE_PATH)





