
import numpy as np
from PIL import Image
import json
from pathlib import Path
import os
import scipy.ndimage as ndi
import tifffile
from csbdeep.data import Normalizer, normalize_mi_ma
from csbdeep.utils import normalize
import timeit
import matplotlib.pyplot as plt

import generate_image_pyramids
import pyramid_upgrade

from get_animal_files import AnimalsContainer
import utils_image_processing as u2
import utils_plotting as up
import utils_general as ug


def test_normalization():
    for an in animals[:]:
        fs_paths = [d.fullsize_paths for d in an.get_valid_datums(['fullsize_paths'])]
        for img_i in [12,35]:
            fsimg = tifffile.imread(fs_paths[img_i])
            print(f'{Path(fs_paths[img_i]).name} --> {img_i} {fsimg.shape}')
            for i in range(3):
                print(f'\t{i} --> {fsimg[i].max()}, mean: {fsimg[i].mean()}')

            # nmin,nmax = 3, 99.2
            # for i in range(3):
            #     chimg = (fsimg[i]/2**16).astype(np.float32)
            #     miFull, maFull = np.percentile(chimg, [nmin,nmax])
            #     fsnorm = normalize(chimg, pmin=nmin, pmax=nmax, clip=True)
                
            #     # u2.print_array_info(chimg)
            #     # print(miFull, maFull)
            #     # u2.print_array_info(fsnorm)
            #     # print()

            #     fsnorm = fsnorm/fsnorm.max()
            #     plt.imshow(fsnorm)
            #     plt.show()


def flt28bit(arr):
    if arr.max() > 1.0:
        arr /= arr.max()
    return np.rint(arr*255).astype('uint8')

def make_ch_first(img_arr):
    if img_arr.shape[0] != 3:
        img_arr = set_chs_first(img_arr)
    assert img_arr.shape[0] == 3, u2.print_array_info(img_arr)
    return img_arr
def set_chs_last(arr):
    return np.moveaxis(arr, 0, -1)
def set_chs_first(arr):
    return np.moveaxis(arr, -1, 0)


def process_img(img_arr, img_json):
    flip, rotate_val = img_json['flip'], img_json['rotate_val']
    print(flip, rotate_val)
    
    # Transpose array to have channels first
    img_arr = make_ch_first(img_arr)
    
    # If the image needs to be flipped, flip it
    if flip:
        img_arr = np.flip(img_arr, axis=2)

    if rotate_val != 0:
        img_arr = ndi.rotate(img_arr, rotate_val, axes=(2,1), reshape=False, order=0, prefilter=False)
        

    return img_arr

def load_image_labels(animal_obj):
    rs_dir = animal_obj.resized
    json_file = os.path.join(rs_dir, 'preABBA_image_transforms.json')
    assert os.path.exists(json_file)
    with open(json_file, 'r') as f:
        image_labels = json.load(f)
    return image_labels


def flip_images(animal_obj, test=False, start_i=None, SHOW=False, SAVE=False):
    img_i_count = 0 # for testing
    t0 = timeit.default_timer()
    base_outdir = Path(animal_obj.fullsize).parent.parent
    resized_outdir, fs_outdir, fsbit8_outdir = [
        ug.verify_outputdir(os.path.join(base_outdir, odn, animal_obj.animal_id))
        for odn in ['resized2', 'fullsize2', 'fullsize_8bit']]
    
    # Load image labels
    image_labels = load_image_labels(animal_obj)


    # iterate through datums, use resized path to index into json for transforms
    for d in animal_obj.get_valid_datums(['fullsize_paths', 'resized_paths'])[start_i:]:
        if img_i_count == 1 and test: break # for testing
        timg = timeit.default_timer()

        # get paths and transforms
        rs_path, fs_path = d.resized_paths, d.fullsize_paths
        img_json = image_labels[rs_path]
        print(rs_path, fs_path)

        # Convert resized image
        rs_img = np.array(Image.open(rs_path)) 
        rs_arr = process_img(rs_img, img_json)
        u2.print_array_info(rs_img)
        u2.print_array_info(rs_arr)
        
        # convert fullsize image
        fs_img = tifffile.imread(fs_path)
        fs_arr = process_img(fs_img, img_json)
        u2.print_array_info(fs_img)
        u2.print_array_info(fs_arr)

        # create 8bit version of fs img
        nmin,nmax = 3, 99.2
        bit8 = np.stack([flt28bit(normalize(fs_arr[i], pmin=nmin, pmax=nmax, clip=True)) for i in range(3)], 0)
        u2.print_array_info(bit8)

        
        # show image
        if SHOW:
            plt.imshow(rs_img);plt.show()
            plt.imshow(set_chs_last(rs_arr));plt.show()
            # plt.imshow(set_chs_last(fs_img));plt.show()
            # plt.imshow(set_chs_last(fs_arr));plt.show()
            # plt.imshow(set_chs_last(bit8));plt.show()
            plt.imshow(rs_img[600:800, 400:600,:]);plt.show()
            plt.imshow(set_chs_last(rs_arr[:,600:800, 400:600]));plt.show()
            plt.imshow(set_chs_last(fs_img[:,6000:8000, 4000:6000]));plt.show()
            plt.imshow(set_chs_last(fs_arr[:,6000:8000, 4000:6000]));plt.show()
            plt.imshow(set_chs_last(bit8[:,6000:8000, 4000:6000]));plt.show()
        
        # save the resulting image
        rs_outpath = os.path.join(resized_outdir, Path(rs_path).name)
        fs_outpath = os.path.join(fs_outdir, Path(fs_path).name)
        fs_8bit_outpath = os.path.join(fsbit8_outdir, Path(fs_path).name)
        print(rs_outpath, fs_outpath, fs_8bit_outpath)
        u2.print_array_info(rs_arr)
        u2.print_array_info(fs_arr)
        u2.print_array_info(bit8)
        
        if SAVE:
            # save the resized image as .png
            Image.fromarray(set_chs_last(rs_arr)).save(rs_outpath)

            # save the fullsize image as a tiff
            tifffile.imwrite(fs_outpath, fs_arr)

            # save a version of the fullsize image rescaled to 8bit using mima normalization as ome.tif
            generate_image_pyramids.main([bit8[i] for i in range(3)], fs_8bit_outpath)
            pyramid_upgrade.main(fs_8bit_outpath, ['zif', 'gfp', 'dapi'])

        img_i_count +=1
        print(f'image completed in {timeit.default_timer()-timg}')
    print(f"{animal_obj.animal_id} completed in {timeit.default_timer()-t0}.")



##########################################################################################################################
# # conda activate stardist
# # python "C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\czi_images\2023_0718_test_preprocess_flipLR_FLIPPER.py"



ac = AnimalsContainer()
ac.init_animals()
animals = ac.get_animals('cohort4')


for an in animals[9:]: 
    flip_images(an, test=bool(0), start_i=None, SHOW=bool(0), SAVE=bool(1))


