import sys
import os
from aicsimageio import AICSImage 
import numpy as np
import pyclesperanto_prototype as cle
import cv2
from datetime import datetime
import pandas as pd
import re
import scipy.ndimage as ndi

# package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "."))) # add current dir to sys.path
import utilities.utils_image_processing as u2
from utilities.utils_plotting import show
import utilities.utils_plotting as up
from auto_crop import CropProcessor, BaseProcess
import generate_image_pyramids
import pyramid_upgrade


def get_image_path(data_dir, fn):
    ''' get path to image, format to remove spaces '''
    image_path = os.path.join(data_dir, fn)
    reformatted_fn = fn.replace(' ', '_') # remove spaces from filename
    if fn != reformatted_fn: # if there are spaces and this has not been done before, rename the file
        os.rename(image_path, os.path.join(data_dir,reformatted_fn))
    fn = reformatted_fn
    image_path = os.path.join(data_dir, fn)
    return image_path, fn

def load_czi_image(image_path):
    ''' load czi image and extract scene ids (number of images) '''
    img = AICSImage(image_path) # get AICSImage object
    scene_ids = img.scenes
    print('dims', img.dims, 'shape', img.shape, '\nscene_ids:', scene_ids)
    return img, scene_ids

def czi_scene_to_array(czi_img, scene_i, czi_fmt_str, czi_fmt_timepoint, czi_fmt_slice, ch_last=True, rotation=None, bgr2rgb=False):
    ''' extract a single image from a czi file, and convert to numpy array '''
    czi_img.set_scene(scene_i)
    arr = czi_img.get_image_data(czi_fmt_str, T=czi_fmt_timepoint)[get_slice_from_string(czi_fmt_slice)]
    if rotation is not None:
        arr = ndi.rotate(arr, rotation, axes=(2,1), reshape=False, order=0, prefilter=False)
    if bgr2rgb:
        arr = np.stack([arr[i] for i in [2,1,0]], 0)
    arr = np.moveaxis(arr, 0, -1) if ch_last else arr # reshape so channels last
    
    print(arr.shape)
    return arr

def get_slice_from_string(slice_str):
    """ Slice the numpy array based on a slice string (e.g. ":, 0, :, :"), returning tuple that can index into numpy array """
    # Split the slice string and remove spaces
    slices = [s.strip() for s in slice_str.split(',')]
    def get_slice_parts(s):
        # Convert the slice strings to actual slice objects or integers
        return slice(*list(map(lambda x: int(x) if x else None, s.split(':')))) if ':' in s else int(s)
    return tuple([get_slice_parts(s) for s in slices])


def verify_output_path(outdir, subdir_name, animal_id, fn):
    output_directory = os.path.join(outdir, subdir_name, animal_id)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return os.path.join(output_directory, fn)



##########################################################################################################################
# SECOND PASS FUNCTIONS 

def extract_new_image_file_name(this_img_id, fn_pattern, order_base_dir):
    match = re.match(fn_pattern, this_img_id)
    if not match:
        raise ValueError (this_img_id)
    animal_id, g2,g3,g4 = match.groups()
    slide_n = get_slide_number(order_base_dir, animal_id, f'{animal_id}_s000_{g2}-{g4}')
    new_fn = f'{animal_id}_{slide_n}_{g2}-{g4}.ome.tif'
    return new_fn, animal_id

def get_slide_number(order_base_dir, animal_i, img_stem):
    try:
        df_path = get_image_order_file(order_base_dir, animal_i)
        if not os.path.exists(df_path):
            index = 0
        else:
            adf = pd.read_excel(df_path)
            index = adf[adf['Order'] == img_stem].index[0]
        s_str = 's' + str(index).zfill(3)
    except IndexError:
        raise IndexError(f"cant find img name in order df.\n\tcurrent name:({order_base_dir, animal_i, img_stem})\n\tpresent names:{adf['Order'].values}")
    return s_str

def get_image_order_file(order_base_dir, animal_i):
    df_path = os.path.join(order_base_dir, animal_i, f'{animal_i}_ImageOrder.xlsx')
    return df_path

def maybe_copy_image_order(order_base_dir, animal_i, resized_path):
    # if making new resized images, on first one copy over image order
    new_df_path = get_image_order_file(os.path.dirname(resized_path), animal_i)
    if not os.path.exists(new_df_path):
        df_path = get_image_order_file(order_base_dir, animal_i)
        if not os.path.exists(df_path):
            raise ValueError(f"source df does not exist, tried to find: {df_path}")
        adf = pd.read_excel(df_path)
        adf.to_excel(new_df_path, index=False)



##########################################################################################################################
# MISC use cases
def split_an_image():
    # used once to split an image that had two sections in it. basically just running the main script but manually defining how to split it
    # TEL73_1_out_of_3
    image_path = 'I:\\raw_czi_images\\TEL_slides_cohort-5\\TEL73_1_out_of_3.czi'
    czi_img, scene_ids = load_czi_image(image_path)# get AICSImage object
    arr = czi_scene_to_array(czi_img, 8, czi_fmt_str, czi_fmt_timepoint, czi_fmt_slice_str, ch_last=CH_LAST, rotation=180, bgr2rgb=True) # get a scene as numpy array
    arr1, arr2 = arr[:11250, ...], arr[11250:, ...]
    for el in [['TEL73_1_out_of_3.czi_8', arr1, 8], ['TEL73_1_out_of_3.czi_99', arr2, 99]]:
        this_img_id, arr, scene_i = el
        new_fn, animal_id = extract_new_image_file_name(this_img_id, fn_pattern, order_base_dir)
        processor = CropProcessor(
                image_path, scene_i, new_fn, animal_id, False, 
                IS_EXCEPTION=True, EXCEPTION_NOTE=None, 
                SECOND_PASS=SECOND_PASS, ATTEMPT_ROTATION=ATTEMPT_ROTATION, SHOW_IMAGES=bool(0), SEGMENTATION_RESULT_DIR=segmentation_result_dir,
                SHOW_SEGMENTATION_RESULT_PLOT=bool(0), SAVE_SEGMENTATION_RESULT_PLOT=bool(0),
                DETECTION_CHANNEL=DETECTION_CHANNEL, RESCALE_FACTOR=RESCALE_FACTOR,
        )
        resized_out = processor.process(arr)

        fullsize_outpath = verify_output_path(outdir, 'fullsize', animal_id, new_fn)
        resized_outpath = verify_output_path(outdir, 'resized', animal_id, new_fn.replace('.ome.tif', '.png'))
        
        # upscale the processing parameters to the fullsize image
        fullsize_out = processor.apply_transforms(arr, scale=1/RESCALE_FACTOR)
        # get cropped resized image without masking background
        resized_out = processor.apply_transforms(processor.resized, scale=1)
        resized_out = u2.convert_16bit_image(resized_out)
        show(resized_out)
        # save all the images
        if SAVE_AND_UPGRADE:
            # save resized
            cv2.imwrite(resized_outpath, resized_out)
            # generate image pyramids
            generate_image_pyramids.main([fullsize_out[...,i] for i in range(3)], fullsize_outpath)
            pyramid_upgrade.main(fullsize_outpath, ['zif', 'gfp', 'dapi'])


def print_keys_from(d, keys, prefix=''):
    for k in keys:
        print(f"{prefix}{k}: {d.get(k)}")

def print_czi_metadata(czi_filepath):
    c = AICSImage(czi_filepath)
    md = c.ome_metadata.dict()
    images_metadata = md['images']
    print(f"{czi_filepath}\n\tnum images:{len(images_metadata)}")
    for imd in images_metadata[:1]:
        print_keys_from(imd['pixels'], ['dimension_order', 'size_c', 'size_t'])
        print_keys_from(imd['pixels'], ['physical_size_x', 'physical_size_y', 'physical_size_z'])
        for ch in imd['pixels']['channels']:
            print(ch['id'])
            print_keys_from(ch, ['fluor', 'name', 'excitation_wavelength', 'emission_wavelength'], prefix='\t')
        


    
    

'''
##########################################################################################################################
    SUMMARY
        ENV: stardist

        STAGE 1: SECOND_PASS=bool(0)
            1) extract each individual image 
            2) create resized version of image and save as .png
            3) use resized images to order sections (using auto_order_sections program)

        STAGE 2: SECOND_PASS=bool(1)
            1) extract each individual fullsize image
            2) assign s_number created in auto_order_sections
            3) convert each fullsize image to OME-TIFF format and save (generate_image_pyramids and pyramid_upgrade)

    DESCRIPTION
        - This is the first step in preparing czi images for use in ABBA. Czi files store multiple images in same file.
        - To prepare them we need to extract each image seperately, prepare fullsized and resized versions of the image, and convert fullsize to OME-Tiff.
            - resized is used to faciliate viewing, optionally use in QuickNii, and to align/rotate the sections before creating qupath/abba project.
            - OME-TIFF format supports storing multi-resolution images (image pyramids) in the same file. Which faciliates alignments in ABBA.
            - there is also a function to semi-autonomously crop the images so just the brain section is visible, some images contain pieces of other sections
                that can mess up the automated registrations in abba.
        - This is done in two stages. Each has a designated mode controlled by setting first pass to true or false.
            1) extract just the resized images and use section ordering program (auto_order_sections) to derive the anterior to posterior order 
                of the sections and assign a 's' number to each image so that all the other data we generate from a given image can be tied together.
                - if using semi-autonomous crop function need to proofread the images, as ~5% will fail b/c it crops part of section we want to keep.
            2) coming back to this script, we then incorporate the s number into the filename and create the fullsize OME-TIFF formated images.


##########################################################################################################################
'''

'''
##########################################################################################################################
    PARAMS
        data_dir (str) --> path to directory with all .czi files
        outdir (str) -->  where to save processed files, will create two folders: fullsize, resized
        fn_pattern = used to parse animal id and index in .czi file from the .czi file name
            '(TEL\d\d)_(\d)_out_of_(\d)\.czi_(\d+)' # example: TEL20_2_out_of_3.czi_7 --> parse groups= TEL20, 2, 3, 7 --> output fn= 'TEL20_2-7...'
        order_base_dir (str) --> directory with resized images used to set order ('sXXX' number), will be directory with folders for each animal
        czi_umPerPixel (float) -->  from czi acquizition, need to know to create ome.tiffs
        exceptions_df_path (str, None) -->  used with AUTOCROP, after manually reviewing tell which to skip b/c autocrop failed
        segmentation_result_dir (str, None) --> if using AUTOCROP this is where to save the figures for manual review
        json_outpath (str, None) --> configured automatically, if using AUTOCROP this is where to save the json file containing all the crop parameters for each image
        
        -- arguments --
            SECOND_PASS (bool) --> if True, assumes exceptions have been filled out and will start to save images to respective folders
            SAVE_AND_UPGRADE (bool) --> if True, create the OME.TIFF image pyramids and save final version of the resized image after the first pass exceptions
            RESCALE_FACTOR (float) --> how much to downsample the resized images
            czi_fmt_str (str) --> default "CZYX", format of .czi image array
            czi_fmt_timepoint (int) --> default 0, timepoint to extract, use 0 if no timepoints
            czi_fmt_slice_str (str) --> default ":, 0, :, :", determines how to slice .czi image, must match czi_fmt_str, assumes not a Z stack
            CH_LAST = (bool) --> whether to format the output image so channels are last dimension, this is the convention that image j expects
            CZI_ROTATION = (float, None) --> rotate .czi images while reading in, used if images are upside down for example
            BGR2RGB = (bool) --> convert channel order from BGR (default for czi) to RGB
            AUTO_CROP (bool) --> whether to use the automated cropping function which attempts to detect the brain section, optional
            DETECTION_CHANNEL (int) --> channel to use for cropping/rotation
            ATTEMPT_ROTATION (bool) --> whether to use the automated rotation function which attempts to rotate the section so it is perfectly horizontal
            SAVE_SEGMENTATION_RESULT_PLOT (bool) --> if using AUTOCROP this generates figures for manual review
            SHOW_SEGMENTATION_RESULT_PLOT (bool) --> if using AUTOCROP shows the figures as they are made
            SHOW_IMAGES (bool) --> if using AUTOCROP shows more detailed diagnostic images while AUTOCROP is running
        

    NOTES
        in the past czi px scaling is 0.650um, used when creating ome-tiffs, but this can change.
 
##########################################################################################################################
'''
if __name__ == '__main__':
    ##########################################################################################################################
    # PARAMETERS
    ##########################################################################################################################
    data_dir = r'I:\raw_czi_images\TEL_slides_cohort-5'
    outdir = r'I:\processed_czi_images_cohort-5'
    order_base_dir = r'I:\processed_czi_images_cohort-5\resized_unedited'
    exceptions_df_path = r'I:\raw_czi_images\cohort5_segmentation_results\segmentation_result_exclusions_cohort-5.xlsx'
    segmentation_result_dir = r'I:\raw_czi_images\cohort5_segmentation_results'

    # each image derived from the czi file is parsed so that each image has a unique ID
    fn_pattern = '(TEL\d\d)_(\d)_out_of_(\d)\.czi_(\d+)' 

    # after generating resized image and defining order, apply s number to image filenames
    SECOND_PASS = bool(0) # if True, assumes exceptions have been filled out and will start to save images to respective folders
    SAVE_AND_UPGRADE = bool(0) # if True, create the OME.TIFF image pyramids and save final version of the resized image after the first pass exceptions

    # parameters for making resized images and extracted images from the .czi file
    RESCALE_FACTOR = 1/10
    czi_fmt_str = "CZYX"
    czi_fmt_timepoint = 0
    czi_fmt_slice_str = ":, 0, :, :"
    CH_LAST = True
    CZI_ROTATION = None # optionally, provide an initial rotation to apply to the image loaded from the czi file
    BGR2RGB = True 

    # AUTOCROP args
    AUTO_CROP = bool(0) # if not using, then will just return images in ABBA compatible format
    DETECTION_CHANNEL = 0 # channel to use for cropping/rotation
    ATTEMPT_ROTATION = bool(0) # if True try to correct rotation, else just use raw image
    SAVE_SEGMENTATION_RESULT_PLOT = bool(0)
    SHOW_SEGMENTATION_RESULT_PLOT = bool(0) 
    SHOW_IMAGES = bool(0) # show diagnostic images while AUTOCROP is running
    
    # if cropping/rotating need to define where to save json files containing crop and rotation applied
    json_outpath = os.path.join(
        segmentation_result_dir,
        f"{os.path.basename(segmentation_result_dir)}_2ndpass-{SECOND_PASS}.json"
    )

    

    
    ##########################################################################################################################
    # MAIN
    ##########################################################################################################################
    
    cle.select_device() # enable GPU (if present)
    baseProc = BaseProcess(json_outpath, exceptions_df_path) # init object that manages storing results of Processor and checking if images are exceptions

    all_fns = sorted([el for el in os.listdir(data_dir) if el.endswith('.czi')])
    fn_indicies = dict(zip(range(len(all_fns)), all_fns))
            
    for fn in all_fns[:1]:
        # load czi image
        t1_czi = datetime.now()
        image_path, fn = get_image_path(data_dir, fn) # get path to image, format to remove spaces
        czi_img, scene_ids = load_czi_image(image_path)# get AICSImage object
        
        #########################################################################################
        # iterate through individual images in czi file
        for scene_i in list(range(len(scene_ids)))[:1]:
            this_img_id = f'{fn}_{scene_i}'
            print('starting processing of:', this_img_id)

            #########################################################################################
            # check if manually declared exception for this image exists
            IS_EXCEPTION, EXCEPTION_NOTE = baseProc.get_exception(this_img_id)

            #########################################################################################
            # get a scene from czi as numpy array serves as the fullsize image
            t1_img = datetime.now()
            arr = czi_scene_to_array(
                czi_img, scene_i, czi_fmt_str, czi_fmt_timepoint, czi_fmt_slice_str, 
                ch_last=CH_LAST, rotation=CZI_ROTATION, bgr2rgb=BGR2RGB,
            ) 
            t_czi2arr = datetime.now() - t1_img
                        
            #########################################################################################
            # generate the new image file name
            new_fn, animal_id = extract_new_image_file_name(this_img_id, fn_pattern, order_base_dir)
            
            #########################################################################################
            # resize, then maybe crop and rotate
            processor = CropProcessor(
                fn, scene_i, new_fn, animal_id, AUTO_CROP, 
                IS_EXCEPTION=IS_EXCEPTION, EXCEPTION_NOTE=EXCEPTION_NOTE, 
                SECOND_PASS=SECOND_PASS, ATTEMPT_ROTATION=ATTEMPT_ROTATION, SEGMENTATION_RESULT_DIR=segmentation_result_dir,
                DETECTION_CHANNEL=DETECTION_CHANNEL, RESCALE_FACTOR=RESCALE_FACTOR,
                SHOW_SEGMENTATION_RESULT_PLOT=SHOW_SEGMENTATION_RESULT_PLOT, SAVE_SEGMENTATION_RESULT_PLOT=SAVE_SEGMENTATION_RESULT_PLOT, SHOW_IMAGES=SHOW_IMAGES,
                
            )
            resized_out = processor.process(arr) #; show(resized_out)
            baseProc.append_to_json_file(this_img_id, processor.get_detection_results_dict())

            #########################################################################################
            # first pass, just to generate resized images so can order them
            if not SECOND_PASS:
                resized_outpath = verify_output_path(outdir, 'resized_processed', animal_id, new_fn.replace('.ome.tif', '.png'))
                cv2.imwrite(resized_outpath, resized_out)

            #########################################################################################
            if SECOND_PASS:
                # prepare output directories 
                fullsize_outpath = verify_output_path(outdir, 'fullsize', animal_id, new_fn)
                resized_outpath = verify_output_path(outdir, 'resized', animal_id, new_fn.replace('.ome.tif', '.png'))
                
                # upscale the processing parameters to the fullsize image
                fullsize_out = processor.apply_transforms(arr, scale=1/RESCALE_FACTOR)
                # get cropped resized image without masking background
                resized_out = processor.apply_transforms(processor.resized, scale=1)
                resized_out = u2.convert_16bit_image(resized_out)
                
                # save all the images
                if SAVE_AND_UPGRADE:
                    # save resized
                    cv2.imwrite(resized_outpath, resized_out)
                    maybe_copy_image_order(order_base_dir, animal_id, resized_outpath) # if making new resized images, on first one copy over image order
                    # generate image pyramids
                    generate_image_pyramids.main([fullsize_out[...,i] for i in range(3)], fullsize_outpath)
                    pyramid_upgrade.main(fullsize_outpath, ['zif', 'gfp', 'dapi'])

            print(f'{this_img_id} took: {datetime.now() - t1_img}, t_czi2arr: {t_czi2arr}, processor time: {processor.processing_time}')
        print(f'{fn} total took: {datetime.now() - t1_czi}')

    




    

