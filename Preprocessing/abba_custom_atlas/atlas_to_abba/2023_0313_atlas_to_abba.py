from aicsimageio import AICSImage 
import os
import matplotlib.pyplot as plt
import skimage
from skimage.filters import sobel
from skimage.segmentation import watershed
import numpy as np
import pyclesperanto_prototype as cle
import cv2
import math
import matplotlib.patches as patches
import json
from datetime import datetime
import pandas as pd
import tifffile
import generate_image_pyramids
import pyramid_upgrade



atlas_dir = r'D:\ReijmersLab\TEL\ccf2017_atlas_images'

img_nissel_path = os.path.join(atlas_dir, 'ccf2017_nissel.tif')
img_labels_path = os.path.join(atlas_dir, 'ccf2017_labels_bla-subdivisions_20230302.tif')
img_nissel_outpath = os.path.join(atlas_dir, 'ccf2017_nissel.ome.tif')
img_labels_outpath = os.path.join(atlas_dir, 'ccf2017_labels_bla-subdivisions_20230313.ome.tif')

img_nissel = tifffile.imread(img_nissel_path).astype('uint16')
# img_nissel = np.expand_dims(img_nissel, -1)
img_labels = tifffile.imread(img_labels_path)
img_labels = np.where(img_labels<0, 0, img_labels)
# img_labels = np.expand_dims(img_labels, -1)
img_labels = img_labels.astype('uint16')
print(img_nissel.shape, img_nissel.min(), img_nissel.max(), img_labels.shape, img_labels.min(), img_labels.max())
print(img_nissel.dtype, img_labels.dtype)



generate_image_pyramids.main([img_nissel], img_nissel_outpath)
generate_image_pyramids.main([img_labels], img_labels_outpath)
pyramid_upgrade.main(img_nissel_outpath, ['nissel'])
pyramid_upgrade.main(img_labels_outpath, ['labels'])