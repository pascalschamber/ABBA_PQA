EXCEPTION_RAISED_DETECTION = False
EXCEPTION_RAISED_ROTATION = False


if SECOND_PASS: # check if img is in exceptions
    if this_img_id in exception_ids:
        # get note
        note = exdf[exdf['exception_ids'] == this_img_id]['note'].values[0]
        if note == 'bad rotation':
            EXCEPTION_RAISED_ROTATION = True
        else:
            EXCEPTION_RAISED_DETECTION = True
            EXCEPTION_RAISED_ROTATION = True


if EXCEPTION_RAISED_DETECTION:
    DETECTION_FAILED = True

else:
    # get the largest label
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # doesn' work as well
    gray = resized[...,1] # works best apparently but might also be due to my adjustment using this result
    labeled = skimage.measure.label(gray>skimage.filters.threshold_otsu(gray))
    bin, rp, largest_label = get_region(labeled, get_largest=True)
    try:
        remove_tiny_holes = skimage.morphology.remove_small_holes(bin, area_threshold=1000)
        remove_thin_features = skimage.morphology.binary_erosion(remove_tiny_holes, footprint=skimage.morphology.disk(3))
        filled = skimage.morphology.binary_dilation(remove_thin_features, footprint=skimage.morphology.disk(4)) # hesitent as ti may throw everything else off
        filled = skimage.morphology.remove_small_holes(filled, area_threshold=16000)
        labeled = skimage.measure.label(filled)
        bin, rp, largest_label = get_region(labeled, get_largest=True)
    except:
        filled = np.zeros_like(gray)
    
    if bool(0):
        cle.imshow(labeled, labels=True)
        show(bin)
        show(remove_tiny_holes)
        show(remove_thin_features)
        show(filled)
    
    # detect if detection failed to assign a label to a significant portion of the section
    detection_failed_area_thresh = 0.3 * bin.size
    DETECTION_FAILED = True if np.count_nonzero(filled) < detection_failed_area_thresh else False

    # detect number of border pixels
    ON_EDGE, n_px = count_border_pixels(bin)
    if n_px > 800: # i.e edge objects were overblown
        labeled = skimage.measure.label(gray>skimage.filters.threshold_otsu(gray))
        bin, rp, largest_label = get_region(labeled, get_largest=True)
        remove_tiny_holes = skimage.morphology.remove_small_holes(bin, area_threshold=1000)
        remove_thin_features = skimage.morphology.binary_erosion(remove_tiny_holes, footprint=skimage.morphology.disk(24))
        filled = skimage.morphology.remove_small_holes(remove_thin_features, area_threshold=2000)
        
        labeled = skimage.measure.label(filled)
        bin, rp, largest_label = get_region(labeled, get_largest=True)
        ON_EDGE, n_px = count_border_pixels(bin)
        # show(bin)
#########################################################################################


# handle if on edge
if DETECTION_FAILED:
    print('!!!! detection failed !!!!')
    
elif ON_EDGE:
    print('Attempting to remove edge objects')
    sigma_spot_detection = 5
    sigma_outline = 1
    err_n = 16
    input_arr = filled
    input_shape = input_arr.shape

    largest_masked = np.where(np.array(input_arr)==1, 1, 0).astype('uint8')
    erosion_kernel = np.ones((err_n,err_n))
    eroded = cv2.erode(largest_masked, kernel=erosion_kernel)

    voronoi_rescale_factor = 0.1
    input_gpu = cle.push(eroded)
    eroded_rescaled = rescale_image(input_gpu, rescale_factor=voronoi_rescale_factor)
    # cle.imshow(eroded_rescaled)
    segmented = cle.voronoi_otsu_labeling(eroded_rescaled, spot_sigma=sigma_spot_detection, outline_sigma=sigma_outline)
    # cle.imshow(segmented, labels=True)
    if segmented.max() > 1:
        # remove any labels that touch the edge
        voronoi_edge_dict = detect_edge_labels(segmented)
        to_keep = [k for (k,v) in voronoi_edge_dict.items() if v == False]
        if to_keep == []: # all objects touch edge
            to_keep = list(voronoi_edge_dict.keys())
            # raise ValueError(voronoi_edge_dict)
        keep = np.zeros_like(segmented)
        for label_i in to_keep:
            keep = np.where(segmented==label_i, 1, keep)
    elif segmented.max() == 1.0:
        keep, diff_rp, diff_largest = get_region(np.array(segmented), get_largest=True)
    else:
        raise ValueError(print(image_path, scene_i), show(np.array(segmented)))

    # perform watershed seg, using label detected by voronoi as seed and background as og not labeled
    markers = np.zeros_like(keep)
    markers = np.where(rescale_image(labeled, voronoi_rescale_factor) == 0, 1, 0) # background
    markers = np.where(keep == 1, 2, markers) # sure foreground
    # cle.imshow(markers, labels=True)
    elevation_map = sobel(rescale_image(gray, voronoi_rescale_factor))
    watershed_seg = watershed(elevation_map, markers, connectivity=3, compactness=1)
    # cle.imshow(watershed_seg, labels=True)
    watershed_result = np.where(watershed_seg!=1, 1, 0)
    # show(keep2)

    result1 = rescale_image(watershed_result, 1/voronoi_rescale_factor)
    # result = cv2.GaussianBlur(result1, (9,9), 1, 1, cv2.BORDER_TRANSPARENT)
    result = skimage.filters.gaussian(result1, sigma=9, preserve_range=True)
    result = np.where(result>0, 1, 0)
    # show(result)
    result = skimage.morphology.binary_erosion(result, footprint=skimage.morphology.disk(32))#np.array(([0,1,0],[1,1,1],[0,1,0])))
    # show(result)

    
    result = pad_rescaled_image(result, input_shape)

    # prevent result from exceeding the size initiall passed in through label
    result = np.where(input_arr == 0, 0, result)
    
    # ensure only one label (largest) is returned
    result_labeled = skimage.measure.label(result)
    result, _, _ = get_region(result_labeled, get_largest=True)
    # add a little dilation/rounding
    result = skimage.morphology.binary_dilation(result, footprint=skimage.morphology.disk(4))

    if bool(0):
        fig, axs = plt.subplots(2,3, figsize=(15, 15))
        cle.imshow(labeled, labels=True, plot=axs[0][0])
        cle.imshow(eroded, labels=True, plot=axs[0][1])
        cle.imshow(segmented, labels=True, plot=axs[0][2])
        cle.imshow(keep, labels=True, plot=axs[1][0])
        cle.imshow(watershed_result, labels=True, plot=axs[1][1])
        cle.imshow(result, labels=True, plot=axs[1][2])
else:
    # if no objects on the edge 
    if n_px>60:
        result = skimage.morphology.binary_dilation(bin, footprint=skimage.morphology.disk(24))
    else:
        gray = cv2.cvtColor(normalize_all_channels(resized), cv2.COLOR_BGR2GRAY)
        labeled = skimage.measure.label(gray>skimage.filters.threshold_otsu(gray))
        bin, rp, largest_label = get_region(labeled, get_largest=True)
        filled = skimage.morphology.remove_small_holes(bin, area_threshold=16000)
        filled = skimage.morphology.binary_dilation(filled, footprint=skimage.morphology.disk(4))
        result = filled
    # cle.imshow(labeled, labels=True)
    # cle.imshow(result)

if DETECTION_FAILED:
    result = None
    merged = resized
    bbox=None
    n_px = None
    
    ABORT_ROTATION = None
    box=None
    rotated_bbox=None
    theta=None

else:

    # calculate the bounding box of the result
    contours,_ = cv2.findContours(result.astype('uint8'), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_TREE
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt) # returns (center(x, y), (width, height), angle of rotation)
    box = cv2.boxPoints(rect) # takes as input the Box2D structure and returns the 4 corner points
    box = np.intp(box) # pts 0 and 3 are mins
    
    def plot_rotated_bbox():
        # plot result of detection with overlaid bounding box
        image = np.stack([result.astype('uint8')]*3, axis=-1)
        image = np.where(image>0, resized, 0)
        cv2.drawContours(image, [box], 0, (36,255,12), 3)
        plt.imshow(image)
        plt.show()


    def find_lr_points(result, box):
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
            return lymin, rymin, lymax, rymax, False
        except IndexError: # think this is caused by what appears to be a heavily rotated box but is actually not a rotated object so this is good to keep as a check where we shuold not be doing rotation anyways
            return None, None, None, None, True
    
    lymin, rymin, lymax, rymax, ABORT_ROTATION = find_lr_points(result, box)
    
    

    if ABORT_ROTATION or EXCEPTION_RAISED_ROTATION or (not ATTEMPT_ROTATION): # if args set to skip trying to fix rotation 
        theta = 0.0
        print('skipping rotation correction.')
    else:
        # find whether min or max pair has lower dY
        delta_mins, delta_maxs = rymin[1] - lymin[1], rymax[1] - lymax[1]
        # set whether to use the greater or lower dY                 |-- here
        el1, el2, delta_y = (rymin, lymin, delta_mins) if delta_mins < delta_maxs else (rymax, lymax, delta_maxs)
        delta_x = el1[0] - el2[0]
        theta = math.degrees(math.atan(delta_y/delta_x))
        print(theta)
    

    if bool(0):
        # # visualize the detected bounding box
        fig,axs = plt.subplots(1,1)
        axs.imshow(result)
        axs.scatter(box[:,0], box[:,1])
        axs.plot([lymin[0],rymin[0]], [lymin[1],rymin[1]])
        axs.plot([lymax[0],rymax[0]], [lymax[1],rymax[1]])
        plt.show()


    # show the difference between input and result mask before rotation
    init_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    init_gray_diff = np.where(result>0, 0, init_gray)

    # extract the grayscale image
    extracted_label = [np.where(result==1, resized[...,i], 0) for i in range(3)]
    #  correct rotation of cropped piece to make it flat
    rotated_cropped = [skimage.transform.rotate(el, theta) for el in extracted_label]
    # merge channels
    merged = np.stack(rotated_cropped, axis=-1)
    # get the rotated bbox
    bin_merged = np.where(merged>0, 1, 0)
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