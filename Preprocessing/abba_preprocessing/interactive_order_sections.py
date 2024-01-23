import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import os
from pathlib import Path
import math
import skimage
import pandas as pd
from tkinter import Tk, Canvas, Frame, Label, Entry, Button, filedialog, StringVar#, PhotoImage
import ctypes
import re
import pyclesperanto_prototype as cle
import time
import sys

# interactive view imports
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from pylab import get_current_fig_manager


# Try importing utilities and TensorFlow-related packages
def add_parent_directory_to_path():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

try:
    # package imports
    add_parent_directory_to_path()
    import utilities.utils_image_processing as u2
except ImportError:
    print('Package utils not loaded.')

try:
    import tensorflow as tf
    from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
except ImportError:
    print('TensorFlow not loaded.')




'''
#################################################################################################
    Main
#################################################################################################
'''

def main():
    # create UI
    ui = UI(
        images_directory=r'I:\processed_czi_images_cohort-5', 
        channel_folder='resized_unedited',
        animal_name_base='TEL', 
        animal_id = 83, 
        base_dir_filter = '.png',
        use_dual_monitors=bool(1),
        init_image_order=bool(1),
        model_path = os.path.join(os.path.dirname(__file__), 'models', '2023_0526_015844_densenet201_attn_bs32_e1200_best_model.h5'),
        crop=bool(0),
        image_channels_display = (0,2),
        # visualization UI params
        set_n_rows = 4, # default = 4
        set_h_spacing=0, # default = 0
        set_v_spacing=0, # default = 0
        set_h_offset=0, # default = 0
        set_v_offset=30, # default = 30
        override_img_px_size = False, # or provide a tuple (x,y)
        overide_monitor_size = False,  # or provide a tuple (x,y)
        overide_screen_area_scaling = (0.9,0.77), # or provide a tuple (x,y)
    )

    # end main loop
    ui.root.mainloop()
    plt.close('all')


'''
#################################################################################################
    Interactive image display
#################################################################################################
'''

def show(image, def_title='none', size=(15, 15), cmap=None, SAVE_PATH=False):
    import matplotlib.pyplot as plt
    
    if isinstance(image, tuple): 
        raise ValueError (f'show cannot handle inputs of type tuple, ensure array is passed')
    
    # read a 3d array or colored 3d array
    if image.ndim == 3 or (image.ndim ==4 and image.shape[-1] == 3):
        if image.ndim == 3 and image.shape[-1] == 3: # for 3 dim image where last dim is RGB
            pass
        # image = create_MIP(image)[0]
    elif image.ndim == 2:
        pass
    else:
        raise ValueError (f'cannot interpret image of shape {image.shape}')
    
    # plot image
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(image, cmap=cmap)
    if def_title != 'none':
        ax.set(title=(str(def_title)))
    
    # save image
    if SAVE_PATH:
        fig.savefig(SAVE_PATH, bbox_inches='tight', dpi=300)

    plt.show()

'''
#################################################################################################
    Interactive image display
#################################################################################################
'''
def interactive_display(state_obj):
    ax_list = []
    img_dirs_i = 0
    for x_row in range(state_obj.n_rows):
        for y_row in range(state_obj.n_cols):
            if img_dirs_i == len(state_obj.img_dirs):
                break
            
            # base figure
            ax = plot_image(state_obj, img_dirs_i)
            
            # get interactive figure manager, configure display params relative to image screen
            current_manager = config_window(state_obj, img_dirs_i, x_row)
            
            # store all managers
            ax_list.append({'manager':current_manager, 'image_path':state_obj.current_img_path})

            # increment image
            img_dirs_i+=1
            # interactive stuff
            zp = make_interactive_ax(ax)

    state_obj.set_ax_list(ax_list)

class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print (event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

def make_interactive_ax(an_ax):
    zp = ZoomPan()
    figZoom = zp.zoom_factory(an_ax, base_scale = 1.1)
    figPan = zp.pan_factory(an_ax)
    return zp

def clear_axis(an_ax):
    an_ax.set_frame_on(False)
    an_ax.get_yaxis().set_ticks([])
    an_ax.get_yaxis().set_ticklabels([])
    an_ax.get_xaxis().set_ticks([])
    an_ax.get_xaxis().set_ticklabels([])
    an_ax.set_xlabel('')
    an_ax.set_ylabel('')


def plot_image(state_obj, img_dirs_i):
    # get the path to the image
    state_obj.current_img_dirs_i = img_dirs_i
    img_path = state_obj.img_dirs[img_dirs_i]
    state_obj.current_img_path = img_path

    # determine whether to load images from paths or use already generated image
    if not state_obj.init_image_order: # original
        img = skimage.io.imread(img_path)
        img = process_imgs(img)
    else:
        img = extract_model_predictions(state_obj.sorted_prediciton_dict, key='image')[img_dirs_i]
    # set channels if specified
    img = set_image_channels_display(state_obj, img) 
    # scale_px_intensity = 1.1
    # img = np.where(img * scale_px_intensity > 255, 255, img * scale_px_intensity)
    # adaptive histogram equalization
    img = skimage.exposure.equalize_adapthist(img, clip_limit=0.1)
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img)
    state_obj.current_img = img
    
    # format axis
    clear_axis(ax)
    ax.set_title('_'.join(el for el in Path(img_path).stem.split('_')[:2]))
    ax.margins(x=0., y=0.)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    return ax

def set_image_channels_display(state_obj, img):
    if state_obj.image_channels_display != '*':
        img_array = []
        assert img.shape[-1] == 3
        for i in range(3):
            if i in state_obj.image_channels_display:
                ch_img = img[...,i]
            else:
                ch_img = np.zeros_like(img[...,i])
            img_array.append(ch_img)
        img = np.stack(img_array, axis=-1)
    return img

def process_imgs(img):
    gray = skimage.color.rgb2gray(img)
    labeled = skimage.measure.label(gray>skimage.filters.threshold_otsu(gray))
    rp = skimage.measure.regionprops(labeled)
    largest_label = np.argmax(np.bincount(labeled.flat)[1:])+1
    pad = 20
    b1,b2,b3,b4 = rp[largest_label-1].bbox
    b1, b2, b3, b4 = max(b1-pad, 0), max(b2-pad, 0), min(b3+pad,img.shape[0]), min(b4+pad, img.shape[1])
    bin = np.where(labeled==largest_label, 1, 0)
    filled = skimage.morphology.remove_small_holes(bin, area_threshold=1024)
    threshed = np.where(np.stack([filled]*3, -1)==1, img, 0)
    cropped = threshed[b1:b3, b2:b4, :]
    return cropped

def config_window (state_obj, img_dirs_i, x_row, title=None, geometry=None):
    '''set window extent and name'''
    
    # calculate image size
    state_obj.final_img_px_size_x, state_obj.final_img_px_size_y = get_img_px_sizes(state_obj)
    
    # set properties for this image
    title = Path(state_obj.current_img_path).stem
    geometry = get_tiled_frame_geometry(
        state_obj,
        img_dirs_i, state_obj.n_cols, x_row, 
        state_obj.final_img_px_size_x, 
        state_obj.final_img_px_size_y, 
        state_obj.h_spacing, state_obj.v_spacing, state_obj.v_offset
    )
    # !!! geometry is correct, it just doesn't get set at right position when crossing monitors

    # set window extent
    thismanager = get_current_fig_manager()
    thismanager.window.setGeometry(*geometry)
    thismanager.set_window_title(title)
    
    
    # return the manager to track position
    return thismanager

def get_tiled_frame_geometry(state_obj, row_i, n_cols, row_x, wfs_x, wfs_y, h_spacing, v_spacing, v_offset):
    '''return a tup of (x-pos, y-pos, width, height)'''
    dual_monitor_x_offset = 0 #!!! if state_obj.use_dual_monitors == False else -1 * get_primary_monitor_screensize(state_obj)[0]
    geometry=(
        (wfs_x+h_spacing)*(row_i%n_cols) + dual_monitor_x_offset + state_obj.h_offset,
        row_x*(wfs_y+v_spacing)+v_offset,
        wfs_x,
        wfs_y
    )
    return geometry

def check_if_overide_screen_area_scaling(state_obj):
    if state_obj.overide_screen_area_scaling != False:
        assert isinstance(state_obj.overide_screen_area_scaling, tuple)
        x_scale, y_scale = state_obj.overide_screen_area_scaling
        if state_obj.current_img_dirs_i == 0:
            print(f'overriding screen_area_scaling to {state_obj.overide_screen_area_scaling}')
        return x_scale, y_scale
    return False

def get_monitor_area(state_obj):
    ''' calculate the maximum x and y coordinates of useable screen space to display images '''
    monitor_size = get_primary_monitor_screensize(state_obj)
    VERT_MONITOR = True if monitor_size[0] < monitor_size[1] else False # detect portrait single monitor, 3/9/23 used to be "if monitor_size == (1440, 2560)"

    # if state_obj.overide_screen_area_scaling != False:
    #     assert isinstance(state_obj.overide_screen_area_scaling, tuple)
    #     x_scale, y_scale = state_obj.overide_screen_area_scaling
    #     if state_obj.current_img_dirs_i == 0:
    #         print(f'overriding screen_area_scaling to {state_obj.overide_screen_area_scaling}')
    #     max_x, max_y = (monitor_size[0]*x_scale)-state_obj.h_offset, (monitor_size[1]*y_scale)-state_obj.v_offset
    check_scaling = check_if_overide_screen_area_scaling(state_obj)

    if state_obj.use_dual_monitors:
        monitor_size = (monitor_size[0]*2, monitor_size[1])
        max_x, max_y = monitor_size    
        if check_scaling != False: 
            x_scale, y_scale = check_scaling
        else:
            x_scale, y_scale = (0.9, 0.75)
        max_x, max_y = int(max_x*x_scale)-state_obj.h_offset, int(max_y*y_scale)-state_obj.v_offset # convert to actual, is different depending on whether using dual monitors
    elif VERT_MONITOR:
        max_x, max_y = monitor_size
        if check_scaling != False: 
            x_scale, y_scale = check_scaling
        else:
            x_scale, y_scale = (0.8, 0.75)
        max_x, max_y = int(max_x*x_scale)-state_obj.h_offset, int(max_y*y_scale)-state_obj.v_offset
    else:
        max_x, max_y = monitor_size
        if check_scaling != False: 
            x_scale, y_scale = check_scaling
        else:
            x_scale, y_scale = (0.8, 0.75)
        # convert to actual, from tests this is max size of 1 monitor thismanager.window.setGeometry(1,30,int(2560*0.8),int(1440*0.75))
        max_x, max_y = int(max_x*x_scale)-state_obj.h_offset, int(max_y*y_scale)-state_obj.v_offset
    
    state_obj.monitor_size = monitor_size
    state_obj.max_x, state_obj.max_y = max_x, max_y
    if state_obj.current_img_dirs_i == 0:
        print(f'get_monitor_area() returned a screen size of ({max_x, max_y})')

    return max_x, max_y

def get_img_px_sizes(state_obj):
    ''' calculate the maximum optimal size of each image that can fit on the display '''
    max_x, max_y = get_monitor_area(state_obj)
    num_imgs = len(state_obj.img_dirs)
    n_rows = state_obj.n_rows #if not VERT_MONITOR else state_obj.n_rows+1
    n_cols = math.ceil(num_imgs/n_rows)
    px_x, px_y = max_x//n_cols, max_y//n_rows
    if state_obj.override_img_px_size !=False:
        assert isinstance(state_obj.override_img_px_size, tuple)
        assert len(state_obj.override_img_px_size) == 2
        px_x, px_y = state_obj.override_img_px_size
    state_obj.n_rows = n_rows
    state_obj.n_cols = n_cols
    return (px_x, px_y)

# DEPRECATED 02/19/2023
# def get_img_px_sizes (state_obj):
#     ''' set single img size based on screen resolution '''
#     n_rows, n_cols, h_spacing, v_spacing = state_obj.n_rows, state_obj.n_cols, state_obj.h_spacing, state_obj.v_spacing
#     mx, my = get_primary_monitor_screensize(state_obj)
#     if state_obj.use_dual_monitors:
#         mx, my = mx*2, my*1
#     img_px_x = int(mx/n_cols)
#     img_px_y = int(my/(n_rows+1))
#     final_img_px_size_x = img_px_x - h_spacing #- ((n_cols-1) * h_spacing)
#     final_img_px_size_y = img_px_y - v_spacing
#     return (final_img_px_size_x, final_img_px_size_y)


def get_primary_monitor_screensize(state_obj):
    '''returns a tuple of width, height for the primary monitor, e.g. (2050,1100) '''
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    if state_obj.overide_monitor_size != False:
        assert isinstance(state_obj.overide_monitor_size, tuple)
        assert len(state_obj.overide_monitor_size) == 2
        if state_obj.current_img_dirs_i == 0:
            print(f'overriding detected screen size of {screensize} to {state_obj.overide_monitor_size}')
        screensize = state_obj.overide_monitor_size

    #!!! overwriting for testing dimension mismatch, gets resolution correctly but in display appears rending is smaller
    # return (2050,1100)
    return (screensize)

def get_row_y_thresholds(state_obj):#final_img_px_size_y, n_rows):
    '''scale the criteria for determining which row an image is in for smaller screens'''
    return [(state_obj.final_img_px_size_y+state_obj.h_spacing+state_obj.v_offset)*iii for iii in range(state_obj.n_rows)]


'''
#################################################################################################
    State variables class 
#################################################################################################
'''
class StateVars:
    def __init__(self, base_dir, UI, **kwargs):
        self.base_dir = base_dir
        self.UI = UI # maintain a reference to the current UI to get animal ID, etc...
        self.ax_list = None
        self.overide_monitor_size = False
        self.overide_screen_area_scaling = False
        self.image_channels_display = '*'
        self.this_start_time = time.time()

        [setattr(self, k, v) for k,v in kwargs.items()]
        self.get_n_cols()

    def get_n_cols(self):
        self.n_cols = math.ceil(len(self.img_dirs)/self.n_rows)

    def set_ax_list(self, ax_list):
        self.ax_list = ax_list
    # def update_state(self):
    #     self.ax_list, self.n_rows, self.base_dir, self.img_dirs = ax_list, n_rows, base_dir, img_dirs

    def print_ax_list(self):
        # self.update_state()
        print('~'*60)
        for i, ax_dict in enumerate(self.ax_list):
            if i%self.n_cols == 0: print()
            manager = ax_dict['manager']
            print(Path(ax_dict['image_path']).stem, '--->', manager.window.x() , manager.window.y())
        print(f'displaying on screen {len(self.ax_list)} of {len(self.img_dirs)} total images')
        print('~'*60)

    def print_state(self):
        # self.update_state()
        # self.print_ax_list()
        print_str = '\n' + '`'*60 + '\n'
        ordered_fps = self.get_window_order()
        for row_i, img_row in self.row_dict.items():
            sorted_row = sorted(img_row)
            for el in sorted_row:
                img_path = Path(img_row[el]['p']).stem
                pattern = '.*_(\d+\-\d+)_.*'
                match = re.match(pattern, img_path)
                if match:
                    print_str += match.groups(1)[0] + ' || '
                else:
                    print_str += f'{img_path} || '
            print_str += ('\n' + '`'*60 + '\n')
        print(print_str)
        print(f'displaying on screen {len(self.ax_list)} of {len(self.img_dirs)} total images')
        # print(self.display_grid(self.monitor_size, [ax_dict['manager'].window.geometry().getCoords() for ax_dict in self.ax_list] ))
    
    def get_window_order(self):
        pos_dicts = [{'x':ax_dict['manager'].window.x() , 'y':ax_dict['manager'].window.y(), 'p':ax_dict['image_path']} for ax_dict in self.ax_list]
        pos_df = pd.DataFrame(pos_dicts)
        # print(pos_df.head())

        # sort by coordinates
        ordered_fps = []
        row_dict_groups = dict(zip(range(self.n_rows), get_row_y_thresholds(self)))
        row_dict = dict(zip(range(self.n_rows), [{},{},{},{}]))
        # break up into rows
        for row_i, row in pos_df.iterrows():
            xx,yy = row['x'], row['y']
            for rn, thresh in row_dict_groups.items():
                if int(yy) in range(thresh-130, thresh + 130):
                    row_dict[rn][xx] = {'y':yy, 'p':row['p']}
        self.row_dict = row_dict

        for row_i, img_row in row_dict.items():
            sorted_row = sorted(img_row)
            for el in sorted_row:
                img_path = Path(img_row[el]['p']).stem
                ordered_fps.append(img_path)
        return ordered_fps
    
    def display_grid(self, screen_size, rectangles):
        # testing a function to help check order, not currently in a useful state
        # Scaling factors
        width_scale = 50
        height_scale = 50

        # Initialize the display grid
        grid_width = screen_size[0] // width_scale
        grid_height = screen_size[1] // height_scale
        display_grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

        # Fill in the square edges
        for index, (x, y, width, height) in enumerate(rectangles):
            start_x = x // width_scale
            start_y = y // height_scale
            end_x = (x + width) // width_scale
            end_y = (y + height) // height_scale

            for i in range(start_y, end_y):
                for j in range(start_x, end_x):
                    if i == start_y or i == end_y - 1 or j == start_x or j == end_x - 1:
                        try:
                            display_grid[i][j] = 'X'
                        except IndexError:
                            pass

        # Convert the grid to a string and return
        return '\n'.join([''.join(row) for row in display_grid])

    def get_position_dicts(self):
        # self.update_state()
        # once all images have been rearranged run this
        ordered_fps = self.get_window_order()
        self.ordered_fps = ordered_fps
        print(ordered_fps)

        out_df = pd.DataFrame(columns=['ImageFolder', 'Order', 'rotate', 'h_flip', 'remove'])
        out_df['Order'] = ordered_fps
        out_df['ImageFolder'] = Path(self.base_dir).parent.name
        
        assert len(ordered_fps) == len(self.img_dirs)
        print('checking num images match expected and ordered\n', f'expected: {len(self.img_dirs)}', f'num_windows: {len(ordered_fps)}')
        return out_df
    
    def save_state_to_xlsx(self):
        outdir = self.base_dir
        out_df = self.get_position_dicts()
        df_out_path = os.path.join(outdir, f'{self.UI.animal_name_base}{self.UI.animal_id}_ImageOrder.xlsx')
        out_df.to_excel(df_out_path, index=False)
        print('saved as:', df_out_path)



def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def crop_imgs(imgs, resize=True):
    cropped_imgs = []
    for img in imgs:
        gray = img[...,0]
        labeled = skimage.measure.label(gray>skimage.filters.threshold_otsu(gray))
        rp = skimage.measure.regionprops(labeled)
        largest_label = np.argmax(np.bincount(labeled.flat)[1:])+1
        pad = 20
        b1,b2,b3,b4 = rp[largest_label-1].bbox
        b1, b2, b3, b4 = max(b1-pad, 0), max(b2-pad, 0), min(b3+pad,img.shape[0]), min(b4+pad, img.shape[1])
        bin = np.where(labeled==largest_label, 1, 0)
        filled = skimage.morphology.remove_small_holes(bin, area_threshold=1024)
        threshed = np.where(np.stack([filled]*3, -1)==1, img, 0)
        cropped = threshed[b1:b3, b2:b4, :]
        # expand dims
        resized = skimage.transform.resize(cropped, (256, 256)) if resize else cropped
        # cropped = np.stack([cropped]*3, axis=-1)
        cropped_imgs.append(resized)
    cropped_imgs = np.stack(cropped_imgs, axis=0) if resize else cropped_imgs
    return cropped_imgs

def rescale_image(image, rescale_factor=0.1):
    ''' rescale all dimensions by same factor '''
    assert image.ndim == 2
    img = cle.push(image)
    resampled = cle.create(
        [int(img.shape[0] * rescale_factor), 
        int(img.shape[1] * rescale_factor), 
        ])

    cle.scale(
        img, resampled, 
        factor_x=rescale_factor, factor_y=rescale_factor,
        centered=False)

    return np.array(resampled)

def convert_16bit_image(image):
    assert image.ndim == 3
    array = []
    for i in range(3):
        ch_min, ch_max = image[...,i].min(), image[...,i].max()
        ch = ((image[...,i]-ch_min)/(ch_max - ch_min))*255
        array.append(ch)
    return np.stack(array, axis=-1).astype('uint8')

def load_imgs(image_paths, convert_grayscale=False, rescale=False):
    # if rescale is True uses cle to downsample by 10x, e.g. for fullsize images
    # Define the desired size of the images
    desired_size = (256, 256)

    # Load each image in the list and resize it
    og_imgs = []
    resized_images = []
    for image_path in image_paths:
        image = skimage.io.imread(image_path)
        og_imgs.append(image)
        image = convert_16bit_image(image) if image.dtype == np.uint16 else image
        image = np.stack([rescale_image(image[...,i]) for i in range(3)],-1) if rescale else image
        image = skimage.color.rgb2gray(image) if convert_grayscale else image
        image = skimage.transform.resize(image, desired_size)
        image = np.stack([image]*3, axis=-1) if convert_grayscale else image
        resized_images.append(image)

    # Stack the resized images into a numpy array
    image_array = np.stack(resized_images, axis=0)

    # normalize
    image_array = normalize(image_array)

    return image_array, og_imgs



def extract_model_predictions(sorted_dict, key='path'):
    ''' returns a list of "key" sorted by prediciton, where key can be pred, path, or image '''
    return [v[key] for v in sorted_dict.values()]

def predict_image_order(model_path, image_paths, crop=True, p=False):
    print('generating model predictions')
    # load pre-trained model and make predictions
    model = tf.keras.models.load_model(model_path, compile=False)
    testX, og_imgs = load_imgs(image_paths, convert_grayscale=False)
    testXc, og_imgs = (crop_imgs(testX), crop_imgs(og_imgs, resize=False)) if crop else (testX, og_imgs)
    testXf = preprocess_input_densenet(testXc)
    predictions = model.predict(testXf)

    # sort the images and paths by prediction
    dict_output = dict(zip(range(len(image_paths)), predictions))
    dict_output_sorted = {k: v for k, v in sorted(dict_output.items(), key=lambda item: item[1])}
    sorted_dict = {}
    for index, pred in dict_output_sorted.items():
        sorted_dict[index] = {
            'pred':pred,
            'path':image_paths[index],
            'image':og_imgs[index],
        }
    sorted_paths = extract_model_predictions(sorted_dict, key='path')

    if p:
        for i in range(len(image_paths)):
            print(i, predictions[i], '--->', image_paths[i])
    print('prediction complete.')

    return sorted_paths, sorted_dict


def make_state_object(
        UI, input_base_dir, animal_index_i, channel_folder, animal_name_base, base_dir_filter, 
        model_path = None,
        use_dual_monitors=False, 
        init_image_order=True, crop=False, 
        set_n_rows = 4, set_h_spacing=0, set_v_spacing=0, set_h_offset=-1, set_v_offset=30, override_img_px_size=False,
        overide_monitor_size = False, overide_screen_area_scaling = False,
        image_channels_display = '*',
        
        ):
    ''' 
        Description
        ``````````````````````````
        get img paths from base dir and create state var 
        NOTE: may need to reset order specified in base dir depending on file directory construction
    
    '''
    base_dir = os.path.join(
        input_base_dir, channel_folder, f'{animal_name_base}{animal_index_i}'
    )
    if not os.path.exists(base_dir):
        raise ValueError(f'could not find base directory {base_dir}')
    img_dirs, sorted_prediciton_dict = sorted([os.path.join(base_dir, el) for el in os.listdir(base_dir) if base_dir_filter in el]), None
    if init_image_order:
        if not os.path.exists(model_path):
            raise ValueError(f"model path not found, looking for: {model_path}")
        img_dirs, sorted_prediciton_dict = predict_image_order(model_path, img_dirs, crop=crop)

    state_obj = StateVars(
        base_dir,
        UI,
        h_spacing = set_h_spacing, # 10
        v_spacing = set_v_spacing, # 50 
        h_offset = set_h_offset,
        v_offset = set_v_offset,
        override_img_px_size = override_img_px_size,
        n_rows = set_n_rows, # use 4 for horizontal monitor and 8 for vertical monitor (typically)
        img_dirs = img_dirs,
        use_dual_monitors=use_dual_monitors,
        init_image_order=init_image_order,
        sorted_prediciton_dict=sorted_prediciton_dict,
        crop=crop,
        overide_monitor_size=overide_monitor_size,
        overide_screen_area_scaling=overide_screen_area_scaling,
        image_channels_display=image_channels_display
    )
    return state_obj
    

'''
#################################################################################################
    UI for on screen buttons 
#################################################################################################
'''  

class UI:
    def __init__(self, **kwargs):
        '''
            images_directory=r'D:\ReijmersLab\TEL\slides\byAnimal', 
            animal_id=15, 
            channel_folder='resized',
            animal_name_base='TEL', 
            base_dir_filter = '.png',
            use_dual_monitors=bool(0),
            init_image_order=bool(1),
            crop=bool(0),
        '''
        HEIGHT = 150
        WIDTH = 500
        root = Tk()
        root.title('Order Sections')
        canvas = Canvas(root, height=HEIGHT, width=WIDTH)
        canvas.pack() 
        frame = Frame(root, bg='#34ebde', bd=5)
        frame.place(relx=0.025, rely=0.05, relwidth=0.95, relheight=0.90)
        frame_label = Label(frame,
                            text='Controls',
                            anchor = 'center', bg='white'
                            )
        frame_label.place(relx = 0, rely = 0, relwidth=1, relheight=0.2)
        
        self.root, self.canvas, self.frame = root, canvas, frame

        # add the args that used to be passed to interactive order session as attributes
        self.model_path = None
        self.state_obj = None
        self.images_directory = None
        self.animal_id = None
        self.channel_folder = None
        self.animal_name_base = None
        self.base_dir_filter = None
        self.use_dual_monitors = False
        self.init_image_order = True
        self.crop = False
        self.set_n_rows = 4
        self.set_h_spacing=0
        self.set_v_spacing=0
        self.set_h_offset=-1
        self.set_v_offset=30
        self.overide_monitor_size = False
        self.overide_screen_area_scaling = False
        self.override_img_px_size = False
        self.image_channels_display = '*'
        for k,v in kwargs.items():
            setattr(self, k, v)

        self.verify_image_channel_display()
        self.interactive_order_sections()
        self.build_UI()

        

    def build_UI(self):
        self.make_directory_button()
        self.make_check_screen_button()
        self.make_save_order_button()
        self.make_close_window_button()
        self.make_get_next_directory_button()

    def interactive_order_sections(self):
        # TODO update state should simply increment animal_index_i which should be an attribute of ui
        global state_obj
        state_obj = make_state_object(
            self,
            self.images_directory, self.animal_id, self.channel_folder, self.animal_name_base, self.base_dir_filter, 
            model_path = self.model_path,
            use_dual_monitors=self.use_dual_monitors, 
            init_image_order=self.init_image_order, 
            crop=self.crop,
            set_n_rows=self.set_n_rows, set_h_spacing=self.set_h_spacing, set_v_spacing=self.set_v_spacing, 
            set_h_offset=self.set_h_offset, set_v_offset=self.set_v_offset,
            override_img_px_size = self.override_img_px_size,
            overide_monitor_size=self.overide_monitor_size, overide_screen_area_scaling=self.overide_screen_area_scaling,
            image_channels_display = self.image_channels_display
        )
        self.update_state(state_obj)
        # tile images along screen
        interactive_display(self.state_obj)
        # sanity checks
        self.state_obj.print_ax_list() 

    def update_state(self, new_state_object):
        self.state_obj = new_state_object
    
    def verify_image_channel_display(self):
        if self.image_channels_display == '*':
            return
        # if not all infer which channels to display
        assert isinstance(self.image_channels_display, tuple)
        return
    

    # buttons
    ###############################################
    # get directory
    def make_directory_button(self):
        label_entry1 = Label(self.frame, 
                        text='images_directory: ', 
                        bg='#34ebde', anchor = 'center')
        label_entry1.place(relx=0, rely=0.25, relwidth=0.25, relheight= 0.2) 

        def prompt_directory():
            self.root.filename = str(filedialog.askopenfilename())
            dir_label = Label(self.frame, text=self.root.filename, bg='#34ebde', anchor = 'w')
            dir_label.place(relx=0.25, rely=0.25, relwidth=0.55, relheight= 0.2)

        self.root.filename=self.images_directory # current file name
        dir_label = Label(self.frame, text=self.root.filename, bg='#34ebde', anchor = 'w')
        dir_label.place(relx=0.25, rely=0.25, relwidth=0.55, relheight= 0.2)

        directory_button = Button(
                                self.frame, text='Browse', bg='white',
                                command=lambda : prompt_directory()
                                )
        directory_button.place(relx=0.85, rely=0.25, relwidth=0.15, relheight= 0.2)

    # button to check order matches screen
    ########################################
    def make_check_screen_button(self):       
        check_screen_button = Button(
            self.frame, text='check order', bg='white', 
            command=lambda : self.state_obj.print_state())
        check_screen_button.place(relx=0.00, rely=0.5, relwidth=0.2, relheight= 0.2)

    # button to save order to excel
    ########################################
    def make_save_order_button(self):
        save_order_button = Button(
            self.frame, text='save order', bg='white', 
            command=lambda : self.state_obj.save_state_to_xlsx())
        save_order_button.place(relx=0.25, rely=0.5, relwidth=0.2, relheight= 0.2)

    

    # button to get next directory
    ########################################
    def make_get_next_directory_button(self):
        def get_next_directory():
            print(f'{self.animal_id} finished in {time.time()-self.state_obj.this_start_time}')
            plt.close('all')
            self.animal_id += 1
            self.interactive_order_sections()

        get_next_directory_button = Button(
            self.frame, text='get next directory', bg='white', 
            command=lambda : get_next_directory())
        get_next_directory_button.place(relx=0.50, rely=0.5, relwidth=0.30, relheight= 0.2)

    # button to close all images
    ########################################
    def make_close_window_button(self):
        close_windows_button = Button(
            self.frame, text='close all', bg='white', 
            command=lambda : plt.close('all'))
        close_windows_button.place(relx=0.0, rely=0.75, relwidth=0.15, relheight= 0.2)

'''
#################################################################################################
MAIN
#################################################################################################
'''

if __name__ == '__main__':
    main()
    






    



        














