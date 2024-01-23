import sys
from pathlib import Path
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
import math
# import pyclesperanto_prototype as cle
from glob import glob
from scipy import stats
import pandas as pd
import time
# from numba import jit
import concurrent.futures
import queue
import timeit
# from matplotlib.lines import Line2D
from tifffile import imread, imwrite
import cv2
import shutil
import random
import json
from datetime import datetime

# prediction imports
from stardist.models import StarDist2D
from stardist import gputools_available
from csbdeep.data import Normalizer, normalize_mi_ma
from csbdeep.utils.tf import keras_import
keras = keras_import()
from csbdeep.utils import Path, normalize
import tensorflow as tf
try:
    print(tf.config.list_physical_devices('GPU'), flush=True)
except AttributeError:
    print('no GPU detected', flush=True)

# package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utilities.utils_data_management import AnimalsContainer
from utilities.utils_plotting import plot_image_grid, get_label_colormap, overlay
import utilities.utils_image_processing as uip
import utilities.utils_general as ug
np.random.seed(0)

'''
    Description
    ~~~~~~~~~~~
        Generate label image of nuclei in each channel for a given image
        ENV: csbdeep - so gpu is supported
    
    Notes
    ~~~~~~~~~~~
        notes - timings
        8/22/23
            all cohorts ~ 2000 disps, took 

        # multiprocess did 330 disps in about 6.5 hours 1.18 min/per (72s), then 350 in 8 hours (1.37 min/per)
        # serially, did 117 in 195 minutes 1.6 min/per (96s)
        6/7/23 - took 13.5 hours for cohort 2
        
'''

##########################################################################################################################
class Processor:
    def __init__(self, model, max_prefetch=1, max_workers=1, intensity_info_path='fs-image-intensities_all-cohorts.json', pred_n_tiles=(2,2)):
        # load, process, and save, with process bottleneck
        self.model = model
        self.intensity_info_path = intensity_info_path
        self.pred_n_tiles = pred_n_tiles
        self.load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        self.load_queue = queue.Queue(maxsize=max_prefetch)
        self.process_queue = queue.Queue(maxsize=1)
        self.save_queue = queue.Queue(maxsize=max_prefetch)

    def predict(self, normed_img, intensity_info, outpath, t0):
        predst=timeit.default_timer()
        label_image = np.stack(
            [self.model.predict_instances(normed_img[...,i], axes='YX', normalizer=None, n_tiles=self.pred_n_tiles)[0] for i in range(3)], -1)
        print(f'predict elapsed time: {timeit.default_timer()-predst}', flush=True)
        return (label_image, intensity_info, outpath, t0)
    
    def save_results(self, label_image, intensity_info, outpath, t0):
        tSave = timeit.default_timer()

        imwrite(outpath, label_image)
        for i in range(3):
            intensity_info[i]['img_max_val'] = float(label_image[...,i].max())

        disp = Dispatcher()
        disp.write_img_intensity_info({Path(outpath).stem : intensity_info}, self.intensity_info_path) # collect image intensity information

        print(f'{intensity_info}\n {Path(outpath).stem} saved in {timeit.default_timer()-tSave}, img elapsed time: {timeit.default_timer()-t0}\n', flush=True)      
        return 0
    
    def process(self, disps):
        # Start the pipeline stages
        self.load_executor.submit(self.loader, disps)
        self.process_executor.submit(self.processor)
        self.save_executor.submit(self.saver)

    def shutdown(self):
        # Shutdown the pipeline stages
        self.load_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        self.save_executor.shutdown(wait=True)

    def loader(self, disps):
        for disp in disps:
            loaded = disp.initialize()
            self.process_queue.put(loaded)
        self.process_queue.put(None)  # Signal that loading is done

    def processor(self):
        while True:
            loaded = self.process_queue.get()
            if loaded is None:
                break  # Loading is done
            result = self.predict(*loaded)
            self.save_queue.put(result)
        self.save_queue.put(None)  # Signal that processing is done

    def saver(self):
        while True:
            result = self.save_queue.get()
            if result is None:
                break  # Processing is done
            self.save_results(*result)


class Dispatcher:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def initialize(self):
        # calls below functions, idea is to have everything ready to predict right after
        t0 = timeit.default_timer()
        img = self.load_image()
        normed_img, intensity_info = self.normalize(img)
        print(f'{self.disp_i} --> {Path(self.p).stem} initialized in {timeit.default_timer() - t0}', flush=True)
        return (normed_img, intensity_info, self.outpath, t0)

    def load_image(self):
        img = uip.read_img(self.p, **self.read_img_kwargs)
        return img
    
    def normalize(self, img):
        intensity_info = {}
        normed_img = np.zeros_like(img, dtype=np.float32)
        for i in range(3):

            # generate normalizer
            nmin,nmax = self.norm[i]['nmin'], self.norm[i]['nmax']
            miFull, maFull = np.percentile(img[...,i], [nmin,nmax])
            normalizerFull = uip.MyNormalizer(miFull, maFull)

            # normalize image before hand
            normed_img[...,i] = normalizerFull.before(img[...,i], 'YX')

            # store intensity info
            intensity_info[i] = {
                'min':float(img[...,i].min()), 'mean':float(img[...,i].mean()), 'max':float(img[...,i].max()),
                'nmin':float(nmin), 'nmax':float(nmax),
            }
        
        return normed_img, intensity_info
    
    def save_results(self, label_image, intensity_info, outpath, t0):
        tSave = timeit.default_timer()

        imwrite(outpath, label_image)

        for i in range(3):
            intensity_info[i]['img_max_val'] = float(label_image[...,i].max())
        self.write_img_intensity_info({Path(outpath).stem : intensity_info}, self.intensity_info_path) # collect image intensity information

        print(f'{intensity_info}\n saved in {timeit.default_timer()-tSave}, img elapsed time: {timeit.default_timer()-t0}\n', flush=True)
        return 0
    
    def write_img_intensity_info(self, image_data_json, intensity_info_path):
        """ save info about min, mean and max values for each channel of each fullsize image """

        # update values if already exists (e.g. from a partial run)
        if os.path.exists(intensity_info_path):
            with open(intensity_info_path, 'r') as f:
                og_image_data_json = json.load(f)
            for k,v in image_data_json.items():
                og_image_data_json[k]=v
            image_data_json = og_image_data_json

        with open(intensity_info_path, 'w') as f:
            json.dump(image_data_json, f)
        
        return image_data_json
        
##########################################################################################################################
def get_dispatchers(
        animals, norm_dict=None, 
        read_img_kwargs={}, intensity_info_path='fs-image-intensities_all-cohorts.json', 
        pred_n_tiles=(2,2),
        CLEAN=False, SKIP_ALREADY_COMPLETED=False, SAMPLE=None, SHUFFLE=False
    ):
    """ get disp objects containing filepaths to process

        ARGS
        - SKIP_ALREADY_COMPLETED (bool) --> do not re-generate/overwrite predictions for existing images
    """
    if CLEAN: 
        ac.clean_animal_dir(animals, 'quant')
    
    print(norm_dict, flush=True)
    dispatchers, disp_i = [], 0
    for an in animals:
        # get paths to fullsize image for predictions
        fs_paths = an.get_valid_datums(['fullsize_paths'], warn=True, SAMPLE=SAMPLE, SHUFFLE=SHUFFLE)
        if len(fs_paths) == 0: raise ValueError('no valid datums found')

        # create directory in animals folder to hold stardist nuclei predictions
        outdir = ug.verify_outputdir(os.path.join(an.base_dir, 'quant'))

        # for each valid datum create a dispatcher object for processing
        for d in fs_paths:
            p = d.fullsize_paths
            pred_out_path = os.path.join(outdir, Path(p).stem[:-4] + '_nuclei.tif')
            
            if norm_dict is None:
                norm = {0:{'nmin':1, 'nmax':99.8}, 1:{'nmin':1, 'nmax':99.8}, 2:{'nmin':1, 'nmax':99.8}}
            else: 
                norm = norm_dict[an.cohort['cohort_name']]
            
            if SKIP_ALREADY_COMPLETED and os.path.exists(pred_out_path): 
                continue

            an_id = an.animal_id_to_int(an.animal_id)
            d_read_img_kwargs = {k:v if not callable(v) else v(an_id) for k,v in read_img_kwargs.items()}

            dispatchers.append(Dispatcher(
                disp_i = disp_i,
                p = p,
                animal_id = an.animal_id,
                cohort = an.cohort['cohort_name'],
                norm = norm,
                read_img_kwargs = d_read_img_kwargs,
                pred_n_tiles=pred_n_tiles,
                outpath = pred_out_path,
                intensity_info_path = intensity_info_path,
            ))
            disp_i += 1
    print(f"st time: {datetime.now().strftime('%H:%M:%S')}\nnum dispatchers: {len(dispatchers)}", flush=True)
    return dispatchers



##########################################################################################################################
# # conda activate csbdeep
# # python "C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\stardist_predict_2023_0513.py"



if __name__ == '__main__':

    tRun = time.time()
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    ac = AnimalsContainer()
    ac.init_animals(failOnError=False)


    # PARAMS
    #######################################################
    NORM_DICT = {
        'cohort2':{0:{'nmin':20, 'nmax':99.8}, 1:{'nmin':20, 'nmax':99.8}, 2:{'nmin':20, 'nmax':99.8}},
        'cohort3':{0:{'nmin':30, 'nmax':99.8}, 1:{'nmin':30, 'nmax':99.8}, 2:{'nmin':20, 'nmax':99.8}},
        'cohort4':{0:{'nmin':30, 'nmax':99.8}, 1:{'nmin':30, 'nmax':99.8}, 2:{'nmin':20, 'nmax':99.8}},
    }
    CLEAN = bool(0)
    MULTIPROCESS = bool(0)
    SKIP_ALREADY_COMPLETED = bool(0)
    PRED_N_TILES = (2,2)
    READ_IMG_KWARGS = {'flip_gr_ch':lambda an_id: True if (an_id > 29 and an_id < 50) else False}
    IMG_INTENSITY_OUTPATH = os.path.join(r'D:\ReijmersLab\TEL\slides\quant_data\fs_img_intensities', 
                                   '2024_0123_fs-image-intensities_all-cohorts.json')
    disp_start_i = None
    disp_end_i = 1
    animals = ac.get_animals('cohort2') + ac.get_animals('cohort3') + ac.get_animals('cohort4')

    #######################################################

    disps = get_dispatchers(
        animals, 
        norm_dict=NORM_DICT, 
        read_img_kwargs=READ_IMG_KWARGS,
        intensity_info_path=IMG_INTENSITY_OUTPATH,
        pred_n_tiles=PRED_N_TILES,
        CLEAN=CLEAN, 
        SKIP_ALREADY_COMPLETED=SKIP_ALREADY_COMPLETED,
    ) [disp_start_i:disp_end_i]
    print(f"processing num disps: {len(disps)}", flush=True)


    
    if MULTIPROCESS:    
        PROC = Processor(model, max_prefetch=1, max_workers=1, intensity_info_path=IMG_INTENSITY_OUTPATH, pred_n_tiles=PRED_N_TILES)
        PROC.process(disps)
        PROC.shutdown()
    
    else: # without preloading, slower but less memory instensive
        for disp in disps:
            # load and normalize image
            normed_img, intensity_info, outpath, t0 = disp.initialize()

            # predict
            label_image = np.stack(
                [model.predict_instances(normed_img[...,i], axes='YX', normalizer=None, n_tiles=disp.pred_n_tiles)[0] for i in range(3)],
                -1)

            # save pred image and intensity info
            disp.save_results(label_image, intensity_info, outpath, t0)
            
    print(f'completed in: {time.time()-tRun}', flush=True)
    












