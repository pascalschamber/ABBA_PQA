import numpy as np
try:
    import cv2
except ImportError:
    pass
import matplotlib.pyplot as plt
import math
from typing import Tuple

import matplotlib
import utils_image_processing as u2

def lbl_cmap():
    # alias for label colormap
    return get_label_colormap()

def get_label_colormap():
    ''' colormap for a label image from pyclesperantos implementation but with ability to get px labels'''
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    import matplotlib
    
    rs = RandomState(MT19937(SeedSequence(3)))
    lut = rs.rand(65537, 3)
    lut[0, :] = 0
    # these are the first four colours from matplotlib's default
    lut[1] = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765]
    lut[2] = [1.0, 0.4980392156862745, 0.054901960784313725]
    lut[3] = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313]
    lut[4] = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392]
    
    cmap = matplotlib.colors.ListedColormap(lut)
    return cmap


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    image_cmap: str,
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5, 
    resize: Tuple[int, int] = (1024, 1024)
    ) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay
    Params:
        image: Training image. should be normalized before hand
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image. (x,y,3)
        
    """
    if image.ndim != 3:
        image = np.stack([image]*3, 0)
    elif image.ndim == 3:# convert ch last to ch first
        if image.shape[-1] == 1: # e.g. a 3dim with only 1 ch
            image = np.stack([image[..., 0]]*3, -1)
        if image.shape[-1] == 3: 
            image = np.moveaxis(image, -1, 0)
        else: raise ValueError(image.shape)
    mask_color = np.asarray(mask_color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=mask_color)
    image_overlay = masked.filled()
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    image_combined = np.moveaxis(image_combined, 0, -1)
    return image_combined

def plot_image_grid(images, masks=False, mask_alpha=0.0005, mask_color=(255,0,0), n_cols=3, size_per_dim=8, labels=False, cmap=None, titles=None, outpath=False, noshow=False):
    ''' plot a x,y grid of images '''
    # coax images into a 2d array if not already
    cmap = get_label_colormap() if labels else cmap
    n_imgs = len(images)
    n_rows = math.ceil(n_imgs/n_cols)
    fig,axs = plt.subplots(n_rows, n_cols, figsize=(size_per_dim*n_cols, size_per_dim*n_rows))
    for img_i in range(n_imgs):
        # img_x = img_i // n_rows
        # img_y = img_i // n_cols
        img_x = img_i // n_cols
        img_y = img_i % n_cols
        # print(img_x, img_y)
        
        ax = axs[img_x, img_y]
        if not masks:
            ax.imshow(images[img_i], cmap=cmap)
        else:
            ax.imshow(overlay(images[img_i], masks[img_i], color=mask_color, alpha=mask_alpha))
        try:
            t = titles[img_i]
        except:
            t = None
        ax.set_title(t, fontsize='x-large')
        ax.axis('off')
    fig.tight_layout()

    if outpath:        fig.savefig(outpath, dpi=300, bbox_inches='tight')
    
    if noshow:        plt.close()
    else:        plt.show()


def show_channels(img, chs=None, clip_max=None):
    print(img.shape)
    assert img.ndim == 3

    if chs == None:
        chs = [0,1,2]
        
    # normalize, or apply label color map
    if img.dtype == np.int32: # labels
        cmaps = [get_label_colormap()]
    elif img.dtype == np.uint16: # images
        if clip_max is None: clip_max = img.max() # equates to min/max normalization
        img = u2.convert_16bit_image(img, NORM=True, CLIP=(0,clip_max))

    if img.dtype == np.uint8:
        rgb = ['red', 'green', 'blue']
        cmaps = [generate_custom_colormap(rgb[c], img) for c in chs]

    for i in range(img.shape[-1]):
        plt.imshow(img[...,i], cmap=cmaps[i])
        plt.show()

            
            
def generate_custom_colormap(color_name, image):
    # create a custom color map from black to specific color, where px at max value appear white
    color_dict = {'red':0, 'green': 1, 'blue': 2}
    
    if color_name not in color_dict.keys():
        raise ValueError("color_name must be 'red', 'green', or 'blue'")
    
    try: # handle integer 
        dtype_max, dtype_min = np.iinfo(image.dtype).max, np.iinfo(image.dtype).min
    except ValueError: # and float dtypes
        dtype_max, dtype_min = np.finfo(image.dtype).max, np.finfo(image.dtype).min

    n = dtype_max-dtype_min  # Number of color steps

    # Generate list of RGBA tuples ranging from black to the specified color
    ch_color = color_dict[color_name]
    color_list = [[0, 0, 0, 1]] + [[i/n if j==ch_color else 0 for j in range(3)] +[1] for i in range(1, n-1)] + [[255,255,255,1]]

    # Create a color map from this list
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('black_to_' + color_name, color_list)
    
    return cmap


def apply_colormap(image, cmap, NORM=False):
    if NORM: # Normalize image to range 0-1
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Apply colormap (this converts image to RGBA)
    image_colored = cmap(image)
    
    return image_colored


def plot_image_hist(img):
    assert img.ndim == 3
    for i in range(img.shape[-1]):
        ch = img[..., i]        
        plt.hist(ch.ravel(), bins=255)
        plt.yscale('log')
        plt.show()

def show(img, ax=None, cmap=None, alpha=1.0, interpolation=None, def_title=None, figsize=(10,10)):
    img = np.array(img)

    # handle different image dimensions
    ####################################
    if img.ndim == 2: # handle 2d
        pass
    elif (img.ndim==3 and img.shape[-1]==3): # handle 2d 3ch
        pass
    else: # if 3d
        img = u2.create_MIP(np.array(img))[0]

    # handle custom colormaps
    if img.dtype.type == np.uint16:
        img = u2.convert_16bit_image(img)
    elif img.dtype == np.int32:
        cmap = get_label_colormap()
        interpolation = 'nearest'

    
    ax_flag = False if ax is None else True
    if not ax_flag: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap=cmap, interpolation=interpolation, alpha=alpha)
    ax.set_title(str(def_title))
    if not ax_flag: plt.show()
    else: return ax
