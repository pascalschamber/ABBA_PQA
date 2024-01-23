from pathlib import Path
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt

# import cv2   #!!! DO NOT USE CV2 FOR non-8bit IMAGES, does not read channels correctly
def to_binary(img):
    return np.where(img>0, 1, 0)

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

try:
    from csbdeep.data import Normalizer, normalize_mi_ma
    class MyNormalizer(Normalizer):
        def __init__(self, mi, ma):
                self.mi, self.ma = mi, ma
        def before(self, x, axes):
            return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)
        def after(*args, **kwargs):
            assert False
        @property
        def do_after(self):
            return False
except ImportError:
    pass
    

def read_img(img_path, flip_gr_ch=False):
    img = imread(img_path)
    if flip_gr_ch:
        print('flipping gr_ch')
        img = np.stack([img[1],img[0],img[2]], 0)
    img = np.moveaxis(img, 0, -1)
    return img

# def read_img(img_path, flip_gr_ch=False):
#     animal_id = int(Path(img_path).stem[Path(img_path).stem.rfind('_TEL')+4:Path(img_path).stem.rfind('_TEL')+6]) # remove
#     img = imread(img_path)
#     flip_gr_ch = True if (animal_id > 29 and animal_id < 50) else False # TODO reimplement as an argumet when function is called
#     if flip_gr_ch:
#         print('flipping gr_ch # TODO reimplement as an argumet when function is called')
#         img = np.stack([img[1],img[0],img[2]], 0)
#     img = np.moveaxis(img, 0, -1)
#     return img, animal_id

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


def read_czi_image(czi_image_path, czi_scene_i=None, czi_fmt_str="CZYX", czi_fmt_timepoint=0, czi_fmt_slice_str=":, 0, :, :"):
    # returns tiles, not for loading whole images shape will be off
    import aicsimageio
    czi_img = aicsimageio.readers.czi_reader.CziReader(czi_image_path)
    for attribute in ['ome_metadata', 'mosaic_tile_dims', 'chunk_dims', 'scenes', 'channel_names']:
        print(f'\t{attribute}: {getattr(czi_img, attribute)}')
    if czi_scene_i is not None:
        czi_img.set_scene(czi_scene_i)
        return czi_img.get_image_data(czi_fmt_str, T=czi_fmt_timepoint)[get_slice_from_string(czi_fmt_slice_str)]
    return czi_img

def get_slice_from_string(slice_str):
    """ Slice the numpy array based on a slice string (e.g. ":, 0, :, :"), returning tuple that can index into numpy array """
    # Split the slice string and remove spaces
    slices = [s.strip() for s in slice_str.split(',')]
    def get_slice_parts(s):
        # Convert the slice strings to actual slice objects or integers
        return slice(*list(map(lambda x: int(x) if x else None, s.split(':')))) if ':' in s else int(s)
    return tuple([get_slice_parts(s) for s in slices])

def read_ometiff_pyramids(img_path, level=0):
    return imread(img_path, series=0, level=level)

def get_ometiff_metadata(ometiff_path):
    import tifffile
    with tifffile.TiffFile(ometiff_path, is_ome=True) as tif:
        metadata = tif.ome_metadata
    return metadata