import cv2
import skimage
import skimage.measure
import skimage.morphology
import skimage.filters
import skimage.transform
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from datetime import datetime
import scipy.ndimage as ndi
import pyclesperanto_prototype as cle

import utilities.utils_image_processing as u2
import utilities.utils_plotting as up


class BaseProcess:
    def __init__(self, json_path, exceptions_df_path):
        self.json_path = json_path
        self.exceptions_df_path = exceptions_df_path
        self.exdf = None
        self.exception_ids = None
        
        # init exceptions
        self.load_exceptions()

    def append_to_json_file(self, key, new_data):
        """
        Append a dictionary to a JSON file.

        Args:
        - new_data (dict): The dictionary to append to the file.
        """
        # Check if the file already exists
        try:
            with open(self.json_path, 'r') as file:
                data = json.load(file)
                if not isinstance(data, dict):
                    raise ValueError("JSON root is not a dict")
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        # Append new data
        data[key] = new_data

        # Write back to file
        with open(self.json_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    def load_exceptions(self):
        if self.exceptions_df_path is not None:
            assert os.path.exists(self.exceptions_df_path), f'file does not exist: {self.exceptions_df_path}'
            self.exdf = pd.read_excel(self.exceptions_df_path)
            self.exdf['exception_ids'] = self.exdf['fn'] + '.czi_' + self.exdf.scene_i.astype('str')
            self.exception_ids = self.exdf.exception_ids.to_list()
        return self.exception_ids
    
    def get_exception(self, this_img_id):
        # check for declared exceptions, i.e. revert to no autocrop, no rotation, or both
        IS_EXCEPTION, EXCEPTION_NOTE = False, None
        if this_img_id in self.exception_ids:
            IS_EXCEPTION, EXCEPTION_NOTE = True, self.exdf[self.exdf['exception_ids'] == this_img_id]['note'].values[0]
        return IS_EXCEPTION, EXCEPTION_NOTE


class CropProcessor:

    def __init__(self, fn, scene_i, new_fn, animal_id, AUTOCROP,
                 IS_EXCEPTION = False, EXCEPTION_NOTE=None, SECOND_PASS=True, ATTEMPT_ROTATION=True, SHOW_IMAGES=False,
                 SHOW_SEGMENTATION_RESULT_PLOT=True, SEGMENTATION_RESULT_DIR='', SAVE_SEGMENTATION_RESULT_PLOT=True,
                 DETECTION_CHANNEL=0, RESCALE_FACTOR=1/10, detection_failed_area_thresh_factor=0.3,
                 area_threshold_holes=1000, binary_erosion_radius=3, binary_dilation_radius=4, remove_holes_area_threshold=16000,
                 edge_px_thresh=800, border_px_thresh=100, check_edge_erosion_radius=24, check_edge_remove_small_holes_thresh=2000,
                 process_edge_objects_sigma_spot_detection = 5, process_edge_objects_sigma_outline = 1, process_edge_objects_errosion_kernel_size = 16, 
                 process_edge_objects_voronoi_rescale_factor = 0.1

                 
        ):
        self.fn = fn
        self.scene_i = scene_i
        self.this_img_id = f'{self.fn}_{self.scene_i}'
        self.new_fn = new_fn
        self.animal_id = animal_id
        self.AUTOCROP = AUTOCROP
        self.IS_EXCEPTION = IS_EXCEPTION
        self.EXCEPTION_NOTE = EXCEPTION_NOTE
        self.SECOND_PASS = SECOND_PASS
        self.ATTEMPT_ROTATION = ATTEMPT_ROTATION
        self.SHOW_IMAGES = SHOW_IMAGES
        self.SHOW_SEGMENTATION_RESULT_PLOT = SHOW_SEGMENTATION_RESULT_PLOT
        self.SEGMENTATION_RESULT_DIR = SEGMENTATION_RESULT_DIR
        self.SAVE_SEGMENTATION_RESULT_PLOT = SAVE_SEGMENTATION_RESULT_PLOT
        self.DETECTION_CHANNEL = DETECTION_CHANNEL
        self.RESCALE_FACTOR = RESCALE_FACTOR
        self.detection_failed_area_thresh_factor = detection_failed_area_thresh_factor
        self.area_threshold_holes = area_threshold_holes
        self.binary_erosion_radius = binary_erosion_radius
        self.binary_dilation_radius = binary_dilation_radius
        self.remove_holes_area_threshold = remove_holes_area_threshold
        self.edge_px_thresh = edge_px_thresh
        self.border_px_thresh = border_px_thresh
        self.check_edge_erosion_radius = check_edge_erosion_radius
        self.check_edge_remove_small_holes_thresh = check_edge_remove_small_holes_thresh
        self.process_edge_objects_sigma_spot_detection = process_edge_objects_sigma_spot_detection
        self.process_edge_objects_sigma_outline = process_edge_objects_sigma_outline
        self.process_edge_objects_errosion_kernel_size = process_edge_objects_errosion_kernel_size
        self.process_edge_objects_voronoi_rescale_factor = process_edge_objects_voronoi_rescale_factor

        # other initializations that maybe set during execution
        self.resized = None
        self.DETECTION_FAILED = False
        self.EXCEPTION_RAISED_DETECTION = False
        self.EXCEPTION_RAISED_ROTATION = False
        self.ON_EDGE = False
        self.ABORT_ROTATION = None
        self.box = None
        self.rotated_bbox = None
        self.theta = None
        # track elapsed time
        self.t0 = datetime.now()


    #################################################################################################################
    #################################################################################################################
    def get_detection_results_dict(self):
        attributes = ['DETECTION_FAILED', 'IS_EXCEPTION', 'EXCEPTION_NOTE', 'ON_EDGE', 'processing_time', 'rotated_bbox', 'box', 'theta']
        arr2list = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        return {k:arr2list(getattr(self, k)) for k in attributes}
    
    def show_images(self, imgs, masks=False, mask_alpha=0.0005, mask_color=(255,0,0), n_cols=3, size_per_dim=8, labels=False, cmap=None, titles=None, outpath=False, noshow=False):
        if self.SHOW_IMAGES: 
            up.plot_image_grid(imgs, masks=masks, mask_alpha=mask_alpha, mask_color=mask_color, n_cols=n_cols, size_per_dim=size_per_dim, labels=labels, cmap=cmap, titles=titles, outpath=outpath, noshow=noshow)
        return None
    
    def check_elapsed_time(self):
        return datetime.now() - self.t0
    
    def norm_uint8(self, img):
        # normalize all channels to range 0,1 then convert to 8 bit
        return np.array(np.stack([u2.normalize_01(img[...,i])*255 for i in range(3)], -1), dtype=np.uint8)

    def check_exceptions(self):
        if self.IS_EXCEPTION:
            if self.EXCEPTION_NOTE == 'bad rotation':
                self.EXCEPTION_RAISED_ROTATION = True
            else:
                self.EXCEPTION_RAISED_DETECTION = True
                self.EXCEPTION_RAISED_ROTATION = True
        if self.EXCEPTION_RAISED_DETECTION:
            self.DETECTION_FAILED = True
    
    def check_detection_failed(self, bin_img, filled):
        # detect if detection failed to assign a label to a significant portion of the section
        detection_failed_area_thresh = self.detection_failed_area_thresh_factor * bin_img.size
        self.DETECTION_FAILED = True if np.count_nonzero(filled) < detection_failed_area_thresh else False
        return self.DETECTION_FAILED
    
    #################################################################################################################


    def process(self, fullsize_arr):
        ''' 
        #################################################################################################################
        main function 
        #################################################################################################################
        '''
        # resize each channel in the image and store as attribute
        self.resized = np.stack([self.rescale_image(fullsize_arr[...,i], rescale_factor=self.RESCALE_FACTOR) for i in range(3)], axis=-1) 
        
        if not self.AUTOCROP: 
            result = self.norm_uint8(self.resized)
        else:
            # else
            self.check_exceptions()
            # continue processing using the helper methods...
            filled, bin_img, labeled, gray, n_px = self.detect(self.resized)
            watershed_result=None

            # maybe process edge objects
            if self.DETECTION_FAILED:
                result = self.resized
            elif self.ON_EDGE:
                result, watershed_result = self.process_edge_objects(filled, labeled, gray)
            else:
                result = self.process_non_edge_objects(bin_img, n_px)
                
            # maybe rotate
            result, self.rotated_bbox, self.theta, self.box = self.auto_rotate(result)
            result = self.norm_uint8(result)

            # maybe show results
            if not self.SECOND_PASS and (self.SHOW_SEGMENTATION_RESULT_PLOT or self.SAVE_SEGMENTATION_RESULT_PLOT):
                self.show_segmentation_results(result, filled, bin_img, labeled, gray, watershed_result)
            
            # maybe crop result
            result = self.crop_image(result, self.rotated_bbox)

            
        self.processing_time = str(self.check_elapsed_time())
        return result
    
    def crop_image(self, img, bbox):
        # maybe crop result
        # crop an image given the rotated_bbox --> coordinates for min/max x and y (e.g. top-left and bottom-right corners of a rectangle)
        if bbox is not None:
            minr, minc, maxr, maxc = np.array(bbox)
            img = img[minr:maxr, minc:maxc, :]
        return img
    
    
    def apply_transforms(self, arr, scale):
        # apply detected transforms to image, scaling transforms by given factor
        # rotate
        if self.theta is not None and self.theta != 0:
            print('applying rotation')
            arr = ndi.rotate(arr, self.theta, axes=(2,1), reshape=False, order=0, prefilter=False)
        # crop rotated image
        if self.rotated_bbox is not None:
            minrf, mincf, maxrf, maxcf = np.intp(np.array(self.rotated_bbox)*(scale))
            arr = arr[minrf:maxrf, mincf:maxcf, :]
        return arr

    def auto_rotate(self, result):
        if self.DETECTION_FAILED:
            return self.resized, None, None, None
        
        box = self.get_object_bounding_box(result)
        lymin, rymin, lymax, rymax = self.find_lr_points(result, box)
        

        if self.ABORT_ROTATION or self.EXCEPTION_RAISED_ROTATION or (not self.ATTEMPT_ROTATION):
            theta = 0.0
            print('skipping rotation correction.')
        else:
            theta = self.calculate_rotation(lymin, rymin, lymax, rymax)
        # self.visualize_detected_bbox(result, box, lymin, rymin, lymax, rymax)

        result, rotated_bbox = self.get_rotated_bbox(result, theta)
        return result, rotated_bbox, theta, box


    
    def get_object_bounding_box(self, result):
        # calculate the bounding box of the result
        contours,_ = cv2.findContours(result.astype('uint8'), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_TREE
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt) # returns (center(x, y), (width, height), angle of rotation)
        box = cv2.boxPoints(rect) # takes as input the Box2D structure and returns the 4 corner points
        box = np.intp(box) # pts 0 and 3 are mins
        return box
    
    def plot_rotated_bbox(self, result, box):
        # plot result of detection with overlaid bounding box
        image = np.stack([result.astype('uint8')]*3, axis=-1)
        image = np.where(image>0, self.resized, 0)
        cv2.drawContours(image, [box], 0, (36,255,12), 3)
        plt.imshow(image)
        plt.show()

    def bbox_to_corners(self, bbox, mode):
        # either convert bbox of minr, minc, maxr, maxc to corners, where these are top left and bottom right points OR
        # convert bbox of miny, minx, height, width to corners it represents, assumes standard image indexing where origin is top-left
        if mode == 'tlbr': # these are top left and bottom right points
            miny, minx, xbr, ybr = bbox
            height, width =  xbr-miny, ybr-minx, 
        elif mode == 'xyhw':
            miny, minx, height, width = bbox
        else: raise ValueError(mode)

        return np.array([
            [minx, miny], 
            [minx + width, miny], 
            [minx + width, miny + height], 
            [minx, miny + height]
        ])


    def box_corners_to_plot_lines(self, box):
        # convert array consisting of 4 corners of a rectange, to an array that is plottable in matplotlib
        # where plot_lines[:,0] is pairs of x coords, plot_lines[:,1] is pairs of y coords
        plot_lines = np.zeros([box.shape[0],2,2])
        for i in range(box.shape[0]):
            next_i = (i+1) % box.shape[0]
            plot_lines[i] = np.stack([box[i], box[next_i]]).T
        return plot_lines
    
    def visualize_detected_bbox(self, img, box):
        # # visualize the detected bounding box
        plot_lines = self.box_corners_to_plot_lines(box)
        fig,ax = plt.subplots(1,1)
        ax.imshow(img)
        ax.plot(plot_lines[:,0], plot_lines[:,1], c='red')
        plt.show()

    def find_lr_points(self, result, box):
        ''' find the points on left half and right half
            returns an attempt to label l/r t/b points 
            if an error is raised the last return element signals to abort rotation '''
        try:
            # find min and max y val of each side, lymin = y,x pos of that point
            left_points = np.where(box[:,0] < result.shape[1]//2)[0]
            left_boxes = box[left_points[0]], box[left_points[1]]
            lymin = left_boxes[np.argmax(np.array(left_boxes)[:,0])]
            lymax = left_boxes[np.argmin(np.array(left_boxes)[:,0])]
            right_points = np.where(box[:,0] > result.shape[1]//2)[0]
            right_boxes = box[right_points[0]], box[right_points[1]]
            rymin = right_boxes[np.argmax(np.array(right_boxes)[:,0])]
            rymax = right_boxes[np.argmin(np.array(right_boxes)[:,0])]
            # print(lymin, rymin, lymax, rymax)
            self.ABORT_ROTATION = False
            return lymin, rymin, lymax, rymax
        except IndexError: # think this is caused by what appears to be a heavily rotated box but is actually not a rotated object so this is good to keep as a check where we shuold not be doing rotation anyways
            self.ABORT_ROTATION = True
            return None, None, None, None

    def calculate_rotation(self, lymin, rymin, lymax, rymax):
        # find whether min or max pair has lower dY
        delta_mins, delta_maxs = rymin[1] - lymin[1], rymax[1] - lymax[1]
        # set whether to use the greater or lower dY                 |-- here
        el1, el2, delta_y = (rymin, lymin, delta_mins) if delta_mins < delta_maxs else (rymax, lymax, delta_maxs)
        delta_x = el1[0] - el2[0]
        theta = math.degrees(math.atan(delta_y/delta_x))
        return theta
    
    def get_rotated_bbox(self, result, theta):
        # extract the grayscale image
        extracted_label = [np.where(result==1, self.resized[...,i], 0) for i in range(3)]
        #  correct rotation of cropped piece to make it flat
        rotated_cropped = [skimage.transform.rotate(el, theta) for el in extracted_label]
        # merge channels
        result = np.stack(rotated_cropped, axis=-1)
        
        # get the rotated bbox
        bin_merged = np.where(result>0, 1, 0)
        bin_merged_sum = sum([bin_merged[...,i] for i in range(3)])
        binary_merged = np.where(bin_merged_sum>0, 1, 0)
        binmerge_label = skimage.measure.label(binary_merged)
        binmerge_rp = skimage.measure.regionprops(binmerge_label)
        if len (binmerge_rp) == 1:
            rotated_bbox = binmerge_rp[0].bbox # expected normal case only 1 label
        elif len(binmerge_rp) == 0:
            raise ValueError('this should not happen, no region labels detected')
        else: 
            # e.g. merging all channels leads to some small objects floating around
            # get index of biggest label by area, and indicies of others
            biggest_rp_i = np.argmax(np.array([rp.area for rp in binmerge_rp]))
            other_rps_is = [i for i in range(len(binmerge_rp)) if i != biggest_rp_i]
            biggest_area = binmerge_rp[biggest_rp_i].area
            other_areas = [binmerge_rp[i].area for i in other_rps_is]

            # ensure it is much bigger than all the rest
            for num in other_areas:
                if num > biggest_area * 0.15: # raise error if significant portion of biggest area is contained in one of the others
                    raise ValueError('significant portion of largest object is going to be excluded')
            
            rotated_bbox = binmerge_rp[biggest_rp_i].bbox
        return result, rotated_bbox



    
    def detect(self, img):
        if self.DETECTION_FAILED:
            print('!!!! detection failed !!!!')
            return img, None, None, None, None
        # get channel to perform detection on, GFP (auto-fluorescence works best)
        gray = self.convert_u16_to_u8(img)[..., self.DETECTION_CHANNEL]
        labeled = skimage.measure.label(self.get_thresholded_image(gray))
        bin_img, rp, largest_label = self.get_region(labeled, get_largest=True)

        try:
            filled = self.fill_holes(bin_img)
            labeled = skimage.measure.label(filled)
            bin_img, rp, largest_label = self.get_region(labeled, get_largest=True)
        except:
            filled = np.zeros_like(gray)
        
        n_px, labeled, filled, bin_img, rp, largest_label = self.check_on_edge(bin_img, gray, labeled, filled, rp, largest_label)
        self.show_images([gray, labeled, filled, bin_img], titles=['gray', 'labeled', 'filled', 'bin_img'])

        return filled, bin_img, labeled, gray, n_px


    def check_on_edge(self, bin_img, gray, labeled, filled, rp, largest_label):
        self.ON_EDGE, n_px = self.count_border_pixels(bin_img)
        if n_px > self.edge_px_thresh: # i.e edge objects were overblown
            labeled = skimage.measure.label(self.get_thresholded_image(gray))
            bin_img, rp, largest_label = self.get_region(labeled, get_largest=True)
            remove_tiny_holes = skimage.morphology.remove_small_holes(bin_img, area_threshold=self.area_threshold_holes)
            remove_thin_features = skimage.morphology.binary_erosion(remove_tiny_holes, footprint=skimage.morphology.disk(self.check_edge_erosion_radius))
            filled = skimage.morphology.remove_small_holes(remove_thin_features, area_threshold=self.check_edge_remove_small_holes_thresh)
            
            labeled = skimage.measure.label(filled)
            bin_img, rp, largest_label = self.get_region(labeled, get_largest=True)
            self.ON_EDGE, n_px = self.count_border_pixels(bin_img)
        # self.show_images([bin_img], labels=True, titles=['check_on_edge_bin_img'])
        return n_px, labeled, filled, bin_img, rp, largest_label
    
    def count_border_pixels(self, image, border_px_thresh = 100):
        ''' returns true if border_px exceeds threshold '''
        # detect number of border pixels
        border_mask = np.zeros_like(image)
        border_mask[1:-1, 1:-1] = 1
        border_px = image - border_mask
        border_px = np.where(border_px>0, 1, 0)
        print('n border px', np.count_nonzero(border_px))
        if np.count_nonzero(border_px) < border_px_thresh:
            return False, np.count_nonzero(border_px)
        else:
            return True, np.count_nonzero(border_px)
    
    def process_non_edge_objects(self, bin_image, n_px):
        # ... [Contents of process_non_edge_objects function]
        if n_px>60:
            result = skimage.morphology.binary_dilation(bin_image, footprint=skimage.morphology.disk(24))
        else:
            gray = cv2.cvtColor(u2.convert_16bit_image(self.resized), cv2.COLOR_BGR2GRAY)
            labeled = skimage.measure.label(gray>skimage.filters.threshold_otsu(gray))
            bin, rp, largest_label = self.get_region(labeled, get_largest=True)
            filled = skimage.morphology.remove_small_holes(bin, area_threshold=16000)
            result = skimage.morphology.binary_dilation(filled, footprint=skimage.morphology.disk(4))
        return result
    


    #... other helper methods...

    def get_thresholded_image(self, gray_image):
        threshold = skimage.filters.threshold_otsu(gray_image)
        return gray_image > threshold
    
    def get_region(self, image, get_largest=True):
        region_props = skimage.measure.regionprops(image)
        min_max_func = np.argmax if get_largest else np.argmin # else get smallest
        label_of_interest = min_max_func(np.bincount(image.flat)[1:]) + 1 
        binary_image = np.where(image==label_of_interest, 1, 0)
        return binary_image, region_props, label_of_interest

    def fill_holes(self, bin_img):
        holes_removed = skimage.morphology.remove_small_holes(bin_img, area_threshold=self.area_threshold_holes)
        eroded = skimage.morphology.binary_erosion(holes_removed, footprint=skimage.morphology.disk(self.binary_erosion_radius))
        dilated = skimage.morphology.binary_dilation(eroded, footprint=skimage.morphology.disk(self.binary_dilation_radius))
        large_holes_removed = skimage.morphology.remove_small_holes(dilated, area_threshold=self.remove_holes_area_threshold)
        self.show_images([bin_img, holes_removed, eroded, large_holes_removed], titles=['bin_img', 'holes_removed', 'eroded', 'large_holes_removed'])
        return large_holes_removed

    def detect_edge_labels(self, image):
        eld = {} # edge label dict
        region_props = skimage.measure.regionprops(image)
        for rp in region_props:
            b1,b2,b3,b4 = rp.bbox #y1,x1,y2,x2
            maxY, maxX = image.shape[0], image.shape[1]
            is_on_edge = True if (b1 == 0 or b2 == 0 or b3 == maxY or b4 == maxX) else False
            eld[rp.label] = is_on_edge
        return list(eld.values())[0] if len(eld) == 1 else eld
    
    def pad_rescaled_image(self, result, input_shape):
        # pad result so og shape is compatible
        x_shape_diff, y_shape_diff = np.array(input_shape) - np.array(result.shape)
        l_pad = x_shape_diff//2
        r_pad = x_shape_diff - l_pad
        t_pad = y_shape_diff//2
        b_pad = y_shape_diff - t_pad
        print(l_pad, r_pad, t_pad, b_pad)
        padded_result = np.zeros(input_shape)
        padded_result[l_pad:padded_result.shape[0]-r_pad, t_pad:padded_result.shape[1]-b_pad] = np.where(result==1, 1, 0)
        return padded_result
    
    def watershed_segmentation(self, keep, labeled, gray):
        # perform watershed seg, using label detected by voronoi as seed and background as og not labeled
        markers = np.zeros_like(keep)
        markers = np.where(self.rescale_image(labeled, self.process_edge_objects_voronoi_rescale_factor) == 0, 1, 0) # background
        markers = np.where(keep == 1, 2, markers) # sure foreground
        # cle.imshow(markers, labels=True)
        elevation_map = skimage.filters.sobel(self.rescale_image(gray, self.process_edge_objects_voronoi_rescale_factor))
        watershed_seg = skimage.segmentation.watershed(elevation_map, markers, connectivity=3, compactness=1)
        # cle.imshow(watershed_seg, labels=True)
        watershed_result = np.where(watershed_seg!=1, 1, 0)
        # show(keep2)

        result1 = self.rescale_image(watershed_result, 1/self.process_edge_objects_voronoi_rescale_factor)
        # result = cv2.GaussianBlur(result1, (9,9), 1, 1, cv2.BORDER_TRANSPARENT)
        result = skimage.filters.gaussian(result1, sigma=9, preserve_range=True)
        result = np.where(result>0, 1, 0)
        # show(result)
        final_result = skimage.morphology.binary_erosion(result, footprint=skimage.morphology.disk(32))#np.array(([0,1,0],[1,1,1],[0,1,0])))
        # show(result)
        return final_result, watershed_result


    def process_edge_objects(self, filled, labeled, gray):
        # try to remove large objects that are touching the edge of the image
        print('Attempting to remove edge objects')
        input_arr = filled
        input_shape = input_arr.shape

        largest_masked = np.where(np.array(input_arr)==1, 1, 0).astype('uint8')
        erosion_kernel = np.ones((self.process_edge_objects_errosion_kernel_size, self.process_edge_objects_errosion_kernel_size))
        eroded = cv2.erode(largest_masked, kernel=erosion_kernel)

        input_gpu = cle.push(eroded)
        eroded_rescaled = self.rescale_image(input_gpu, rescale_factor=self.process_edge_objects_voronoi_rescale_factor)
        # cle.imshow(eroded_rescaled)
        segmented = cle.voronoi_otsu_labeling(eroded_rescaled, spot_sigma=self.process_edge_objects_sigma_spot_detection, outline_sigma=self.process_edge_objects_sigma_outline)
        # cle.imshow(segmented, labels=True)
        if segmented.max() > 1:
            # remove any labels that touch the edge
            voronoi_edge_dict = self.detect_edge_labels(segmented)
            to_keep = [k for (k,v) in voronoi_edge_dict.items() if v == False]
            if to_keep == []: # all objects touch edge
                to_keep = list(voronoi_edge_dict.keys())
                # raise ValueError(voronoi_edge_dict)
            keep = np.zeros_like(segmented)
            for label_i in to_keep:
                keep = np.where(segmented==label_i, 1, keep)
        elif segmented.max() == 1.0:
            keep, diff_rp, diff_largest = self.get_region(np.array(segmented), get_largest=True)
        else:
            raise ValueError(print(self.this_img_id), up.show(np.array(segmented)))

        # apply watershed
        result, watershed_result = self.watershed_segmentation(keep, labeled, gray)
        result = self.pad_rescaled_image(result, input_shape)

        # prevent result from exceeding the size initiall passed in through label
        result = np.where(input_arr == 0, 0, result)
        
        # ensure only one label (largest) is returned
        result_labeled = skimage.measure.label(result)
        result, _, _ = self.get_region(result_labeled, get_largest=True)
        # add a little dilation/rounding
        result = skimage.morphology.binary_dilation(result, footprint=skimage.morphology.disk(4))

        self.show_images([labeled, eroded, segmented, keep, watershed_result, result], labels=True, titles=['labeled', 'eroded', 'segmented', 'keep', 'watershed_result', 'result'])
        return result, watershed_result




    def show_segmentation_results(self, result, filled, bin_img, labeled, gray, watershed_result):
        # create summary figure of segmentation results
        fig,axs = plt.subplots(5,3, figsize=(20,20))
        fig.suptitle(self.fn + ' - ' + str(self.scene_i))
        axs[0][0].imshow(self.resized)
        axs[0][1].imshow(gray)
        axs[0][2].imshow(labeled)
        axs[1][0].imshow(filled)
        
        if self.ON_EDGE and not self.DETECTION_FAILED:
            # axs[1][1].imshow(eroded)
            # axs[1][2].imshow(segmented)
            # axs[2][0].imshow(keep)
            # axs[2][1].imshow(markers)
            axs[2][2].imshow(watershed_result)
            axs[3][0].imshow(result)
        if not self.DETECTION_FAILED:
            pass
            # imagerb = np.stack([result.astype('uint8')]*3, axis=-1)
            # imagerb = np.where(imagerb>0, self.resized, 0)
            # # cv2.drawContours(imagerb, [box], 0, (36,255,12), 3)
            # axs[3][1].imshow(imagerb)        
            # axs[3][2].imshow(np.where(result>0,1,0))
        axs[4][0].imshow(self.resized)
        axs[4][1].imshow(result)
        init_gray_diff = np.where(result>0, 0, self.resized)
        axs[4][2].imshow(init_gray_diff)
        if self.box is not None:
            plot_lines = self.box_corners_to_plot_lines(self.box)
            axs[4][1].plot(plot_lines[:,0], plot_lines[:,1], c='red')
        if self.rotated_bbox is not None:
            plot_lines = self.box_corners_to_plot_lines(self.bbox_to_corners(self.rotated_bbox, mode='tlbr'))
            axs[4][2].plot(plot_lines[:,0], plot_lines[:,1], c='red')
        
        axs[0][0].set_title('resized')
        axs[0][1].set_title('gray')
        axs[0][2].set_title('labeled')
        axs[1][0].set_title('filled')
        axs[1][1].set_title('eroded')
        axs[1][2].set_title('segmented')
        axs[2][0].set_title('keep')
        axs[2][1].set_title('markers')
        axs[2][2].set_title('watershed_result')
        axs[3][0].set_title('result')
        axs[3][1].set_title('result with bbox')
        axs[3][2].set_title('merged as mask')
        axs[4][0].set_title('resized')
        axs[4][1].set_title('segmentation result')
        axs[4][2].set_title('init gray minus result mask')
        
    
        if self.SAVE_SEGMENTATION_RESULT_PLOT: 
            fig_outpath = os.path.join(self.SEGMENTATION_RESULT_DIR, f'{self.fn}_{self.scene_i}.png')
            plt.savefig(fig_outpath, dpi=150, bbox_inches='tight')
        if self.SHOW_SEGMENTATION_RESULT_PLOT: plt.show()
        else: plt.close()

    






    ###################################################################################################
    def rescale_image(self, image, rescale_factor=0.1):
        ''' rescale all dimensions by same factor '''
        assert image.ndim == 2
        image_dtype = image.dtype
        img = cle.push(image)
        resampled = cle.create(
            [int(img.shape[0] * rescale_factor), 
            int(img.shape[1] * rescale_factor), 
            ])

        cle.scale(
            img, resampled, 
            factor_x=rescale_factor, factor_y=rescale_factor,
            centered=False)

        return np.array(resampled, dtype=image_dtype)


    def convert_u16_to_u8(self, aimg, max_intensity_range=True):
        '''convert 16-bit image to 8-bit for saving as .png'''
        # return ((aimg-aimg.min())/(aimg.max()-aimg.min())*255).astype('uint8')
        # better to use cv2 to handle rounding errors
        if max_intensity_range:
            return cv2.normalize(aimg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else: # use local max
            normed = (aimg/2**16) * 255
            arrs = []
            for i in range(3):
                norm_max = int(aimg[...,i].max()/2**16 * 255)
                arrs.append(cv2.normalize(normed[...,i], None, 0, norm_max, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
            return np.stack(arrs, axis=-1)
    


# Example usage:
# resized_image = None  # Some image input
# EXCEPTION_RAISED_DETECTION = False
# resized, result, bbox = process_image(resized_image, EXCEPTION_RAISED_DETECTION)

# To use the class:
# processor = CropProcessor(fn, scene_i, resized, exception_ids, exdf)
# processor.process()