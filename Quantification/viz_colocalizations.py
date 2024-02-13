import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tifffile import imread
from timeit import default_timer as dt
import ast
import matplotlib.patches as mpatches
import scipy
from skimage import morphology

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
import utilities.utils_image_processing as uip
import utilities.utils_plotting as up
import utilities.utils_general as ug
from utilities.utils_general import verify_outputdir
from utilities.utils_data_management import AnimalsContainer
import utilities.utils_atlas_region_helper_functions as arhfs
import core_regionPoly as rp


"""
##################################################################################################
    DESCRIPTION
        plotting tool to visualize nuclei colocalizations
        two modes
            - view individual nuclei and the image channels they are colocal with
            - view a subset of regions with nuclei outlined by colocal id
##################################################################################################
"""

def filter_label_img(segmented_image, labels_to_keep):
    # filter object labels from a segmented label image
    mask = np.zeros(segmented_image.shape, dtype=bool)

    # Iterate over each label you wish to keep and update the mask
    for label in labels_to_keep:
        mask = mask | (segmented_image == label)

    # Apply the mask to the segmented image
    # Set pixels not in `labels_to_keep` to 0 (or any other background value you prefer)
    filtered_image = np.where(mask, segmented_image, 0)
    return filtered_image
    
def mask_to_outlines(binary_mask):
    # extract object outlines from a label mask
    sobel_h, sobel_v = scipy.ndimage.sobel(binary_mask, axis=0), scipy.ndimage.sobel(binary_mask, axis=1)
    magnitude = np.where(np.sqrt(sobel_h**2 + sobel_v**2) != 0, 1, 0).astype(int)
    outline = morphology.skeletonize(magnitude)
    return outline

def show_channels(img, chs=[0,1,2], clip_max=None, draw_box=None, show_lbl_is_only=None, extent=None, axs=None, interpolation = None):
    # show individual channels from an intensity and segmented image
    assert img.ndim == 3

    # normalize, or apply label color map
    if img.dtype == np.int32: # labels
        cmaps = [up.get_label_colormap() for _ in range(img.shape[-1])]
        interpolation = 'nearest'
    elif img.dtype == np.uint16: # images
        if clip_max is None: clip_max = img.max() # equates to min/max normalization
        img = uip.convert_16bit_image(img, NORM=True, CLIP=(0,clip_max))

    if img.dtype == np.uint8:
        rgb = ['red', 'green', 'blue']
        cmaps = [up.generate_custom_colormap(rgb[c], img) for c in chs]

    
    for i in range(img.shape[-1]):
        if axs is None:
            fig, ax = plt.subplots()
        else:
            ax=axs[i]
        if show_lbl_is_only is not None:
            img[...,i] = np.where(img[...,i]==show_lbl_is_only[i], 1, 0)
        
        ax.imshow(img[...,i], cmap=cmaps[i], extent=extent, interpolation=interpolation)

        if draw_box is not None:
            rect = mpatches.Rectangle(
                (draw_box[0], draw_box[1]), draw_box[3]-draw_box[1], draw_box[2]-draw_box[0],
                linewidth=1, edgecolor='w', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

def plot_colocalization(rpdf_view, centroid_i=63222, nuc_view_pad=500):
    # plot individual colocal nuclei
    rpdf_view = rpdf_view.loc[centroid_i, :].to_dict()
    possible_cols = ['ch0_intersecting_label', 'label', 'ch2_intersecting_label']
    show_lbl_is = [int(rpdf_view[col]) for col in possible_cols if ((col in rpdf_view) and (not pd.isnull(rpdf_view[col])))]

    img_bbox = ast.literal_eval(rpdf_view['bbox'])
    view_bbox = np.array(img_bbox) + np.array([-nuc_view_pad, -nuc_view_pad, nuc_view_pad, nuc_view_pad])
    nuc_bbox = np.array(img_bbox) - np.array([view_bbox[0], view_bbox[1], view_bbox[0], view_bbox[1]])
    x1, y1, x2, y2 = view_bbox

    fig, axs = plt.subplots(3,2, figsize=(10,15))
    axs = axs.flatten()
    show_channels(fs_img[x1:x2, y1:y2,  :], axs=[axs[0],axs[2],axs[4]], extent=None, draw_box=nuc_bbox)
    show_channels(nuc_img[x1:x2, y1:y2,  :], axs=[axs[1],axs[3],axs[5]], extent=None, draw_box=nuc_bbox)#, show_lbl_is_only=show_lbl_is)
    plt.show()

def parse_reg_ids(GET_REGION_ID, GET_CHILD_REGIONS, rpdf):
        child_ids = None
        if GET_REGION_ID is not None:
            child_ids = [GET_REGION_ID] + arhfs.get_all_children(ont.ont_ids[GET_REGION_ID]['children']) if GET_CHILD_REGIONS else [GET_REGION_ID]
            # filter rpdf to only contain children
            rpdf = rpdf[rpdf['reg_id'].isin(child_ids)]
            assert len(rpdf) != 0, f"{child_ids} not found in rpdf"
        return rpdf, child_ids

def parse_side(GET_REG_SIDE, rpdf):
    get_side = None
    if GET_REG_SIDE is not None:
        assert GET_REG_SIDE in ['First', 'Left', 'Right'], f"{GET_REG_SIDE} must be one of ('First', 'Left', 'Right')"
        if GET_REG_SIDE == 'First':
            get_side = rpdf.loc[rpdf.index[0], 'reg_side']
        else:
            get_side = GET_REG_SIDE
        rpdf[rpdf['reg_side'] == get_side]
    return rpdf, get_side

def parse_regionpolys(child_ids, get_side, regionPoly_list):
    # parse region polys
    region_poly_objs = []
    if child_ids is not None and get_side is not None:
        for poly_obj in regionPoly_list:
            # check regid and side 
            for reg_id in child_ids:
                if poly_obj.obj_id == reg_id and poly_obj.reg_side == get_side:
                    region_poly_objs.append(poly_obj)
                    break

    # print info about each region being plotting
    for p in region_poly_objs:
        print(p)
    
    return region_poly_objs

def parse_regions_extent(region_poly_objs):
    # get extent of all child regions
    extents, poly_arrays = [], []
    for p in region_poly_objs:
        poly_arrays.append(p.poly_arrays)
        for pdict in p.poly_arrays.values():
            extents.append(pdict['arr'])
    return extents, poly_arrays

# PARAMS
##################################################################################################
GET_ANIMALS = ['TEL15']
GET_AN_I = 0
GET_DATUM_I = 21
GET_REG_SIDE = 'First'
GET_REGION_ID = 295
GET_CHILD_REGIONS = bool(1)
PLOT_NUC_IS = [0, 5]
SHOW_COLOCAL_ID = 3
read_img_kwargs = {'flip_gr_ch':lambda an_id: True if (an_id > 29 and an_id < 50) else False} 
cmaps = ['red', 'green', 'blue', 'magenta'][:3]


# setup 
##################################################################################################
ont = arhfs.Ontology()
ac = AnimalsContainer()
ac.init_animals()
animals = ac.get_animals(GET_ANIMALS)
an = animals[GET_AN_I]
datums = an.get_valid_datums(['fullsize_paths', 'quant_dir_paths', 'geojson_regions_dir_paths'])
datum = datums[GET_DATUM_I]

# read data
rpdf = pd.read_csv(datum.rpdf_paths)
region_df = pd.read_csv(datum.region_df_paths)
geojson_objs = rp.load_geojson_objects(datum.geojson_regions_dir_paths)
regionPoly_list = rp.extract_polyregions(geojson_objs, ont) 
# read images
d_read_img_kwargs = {k:v if not callable(v) else v(an.animal_id_to_int(an.animal_id)) for k,v in read_img_kwargs.items()} if read_img_kwargs else {}
fs_img = uip.read_img(datum.fullsize_paths, **d_read_img_kwargs)
nuc_img = imread(datum.quant_dir_paths)



# show fullsize images
##################################################################################################
if bool(0):
    plt.imshow(np.stack([
        uip.convert_16bit_image(fs_img[...,i], NORM=True, CLIP=(0,fs_img[...,i].max())) for i in range(fs_img.shape[-1])], -1)
        );plt.show()
    plt.imshow(nuc_img[...,0], cmap=up.lbl_cmap(), interpolation='nearest');plt.show()


# show colocal nuclei
##################################################################################################
if bool(0):
    rpdf_clc = rpdf[rpdf['colocal_id']==SHOW_COLOCAL_ID]
    valid_centroid_is = rpdf_clc['centroid_i'].values
    print(f"num valid_centroid_is: {len(valid_centroid_is)}")
    for ci in range(PLOT_NUC_IS[0], PLOT_NUC_IS[-1]): 
        plot_colocalization(rpdf_clc, centroid_i=valid_centroid_is[ci], nuc_view_pad=100)


# show regions with nuclei detections and colocalizations overlayed
##################################################################################################
if bool(0):
    # PARAMS
    ##############################
    colocal_ids = [0,1,2,3,4]
    img_chs = [2,0,1,1,1]
    hide_chs = []
    mask_colors = [(0, 0, 255, 255), (255, 0, 0, 255),(0, 255, 0, 255), (255, 255, 0, 255), (255, 255, 255, 255)]
    show_colocal_ids = [0,1,2,3]


    # prep imgs for display
    ##############################
    comp_img = up.create_composite_image_with_colormaps(fs_img.astype(float), colormaps=cmaps)

    # parse ids, side, and regions polys to get
    rpdf, child_ids = parse_reg_ids(GET_REGION_ID, GET_CHILD_REGIONS, rpdf)
    rpdf, get_side = parse_side(GET_REG_SIDE, rpdf)
    region_poly_objs = parse_regionpolys(child_ids, get_side, regionPoly_list)

    # get extent of all child regions and get regionpoly arrays
    extents, poly_arrays = parse_regions_extent(region_poly_objs)
    xmin, ymin, xmax,  ymax = np.rint(rp.get_polygons_extent(extents)).astype(int)

    # format the intensity image to 8bit for display, add alpha channel
    int_img = comp_img.copy()
    image_combined = (int_img[ymin:ymax, xmin:xmax,  :] * 255).astype('uint8')
    image_combined = np.concatenate([image_combined,np.ones_like(image_combined[...,:1])*255], axis=-1)
    nuc_crop = nuc_img[ymin:ymax, xmin:xmax,   :].copy()

    # hide channels in intensity image if desired
    for ch_i in hide_chs:
        image_combined[...,ch_i] = 0

    # get outlines of nuclei, adding them onto intensity image
    for clc_i, clc_id in enumerate(colocal_ids[:]):
        print(clc_id)
        if clc_id not in show_colocal_ids:
            continue
        img_ch = img_chs[clc_i]
        clc_labels = rpdf[rpdf['colocal_id']==clc_id]['label'].values
        print(len(clc_labels))
        filt_lbl_img = filter_label_img(nuc_crop[..., img_ch], clc_labels)
        mask = mask_to_outlines(filt_lbl_img)
        y_coords, x_coords = np.where(mask>0)
        image_combined[y_coords, x_coords, :] = mask_colors[clc_i]


    # show image with nuc outlines
    fig,ax = plt.subplots(figsize=(20,20))
    ax.imshow(image_combined)

    # overlay region outlines
    base_outlines = (0.5, 0.5, 0.5, 0.8) 
    blank_rgba_fc = (1.0, 1.0, 1.0, 0.0)
    palette_outlines = {'exteriors':base_outlines, 'main': base_outlines, 'interiors':(1.0, 0.0, 0.0, 1.0)}
    palette_fc = {'exteriors':blank_rgba_fc, 'main': blank_rgba_fc, 'interiors':(1.0, 0.0, 0.0, 0.2)}
    for el in poly_arrays:
        for pi, pdict in el.items():
            print(pdict['polyType'])
            arr, ptype = pdict['arr'], pdict['polyType']
            arr[:, 0] -= xmin
            arr[:, 1] -= ymin
            ec, fc = palette_outlines[ptype], palette_fc[ptype]
            polygon = mpatches.Polygon(arr, linewidth=0.5, edgecolor=ec, facecolor=fc)
            ax.add_patch(polygon)





