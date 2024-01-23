
from pathlib import Path
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt

# import cv2   #!!! DO NOT USE CV2 FOR non-8bit IMAGES, does not read channels correctly

def normalize_01(image):
    return (image-np.min(image))/(np.max(image)-np.min(image)) 


def convert_16bit_image(image, NORM=True, CLIP=None):
    # if NORM, normalize to img min/max, else use max possible value for 16 bit image
    # if CLIP, set min, max directly, useful for standardizing image display
    ndims = image.ndim
    if ndims==2:
        image = np.expand_dims(image,-1)
    assert image.ndim == 3
    assert image.shape[-1] < image.shape[0] and image.shape[-1] < image.shape[1] # assert chs last
    array = []
    for i in range(image.shape[-1]):
        if CLIP is None:
            ch_min, ch_max = (image[...,i].min(), image[...,i].max()) if NORM else (0, 2**16)
        else: # set max px value
            ch_min, ch_max = CLIP[0], CLIP[1]

        ch = ((image[...,i]-ch_min)/(ch_max - ch_min))*255
        ch = np.clip(ch, 0, 255)

        array.append(ch)
    if ndims ==2:
        return array[0].astype('uint8')
    else:
        return np.stack(array, axis=-1).astype('uint8')
    # def convert_16bit_image(image): # old version 10/24/23
    # assert image.ndim == 3
    # array = []
    # for i in range(3):
    #     ch_min, ch_max = image[...,i].min(), image[...,i].max()
    #     ch = ((image[...,i]-ch_min)/(ch_max - ch_min))*255
    #     array.append(ch)
    #     # if bool(0):
    #     #     plt.hist(ch.ravel(), bins=25)
    #     #     plt.yscale('log')
    #     #     plt.show()
    # return np.stack(array, axis=-1).astype('uint8')

def to_binary(img):
    return np.where(img>0, 1, 0)


def read_img(img_path):
    animal_id = int(Path(img_path).stem[Path(img_path).stem.rfind('_TEL')+4:Path(img_path).stem.rfind('_TEL')+6])
    img = imread(img_path)
    flip_gr_ch = True if animal_id > 29 else False
    if flip_gr_ch:
        img = np.stack([img[1],img[0],img[2]], 0)
    img = np.moveaxis(img, 0, -1)
    print(Path(img_path).stem)
    print(img.shape)
    return img, animal_id

def crop_img(bbox, img, pad, chs):
    if isinstance(pad, int): pad = [pad, pad]
    elif isinstance(pad, list) and len(pad) == 2:
        assert all([isinstance(el, int) for el in pad]), pad
    xmax, ymax, _ = img.shape
    x,y,X,Y = np.array(bbox) + np.array([-pad[0], -pad[1], pad[0], pad[1]])
    x,y,X,Y = max(x, 0), max(y,0), min(X, xmax), min(Y, ymax)
    crop = img[x:X,y:Y, chs]
    return crop

def print_array_info(array):
    print('shape = ', array.shape)
    print('min/max = ', array.min(), array.max())
    print('dtype = ', array.dtype)