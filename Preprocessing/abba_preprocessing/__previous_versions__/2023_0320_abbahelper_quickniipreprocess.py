import os
from pathlib import Path
import re
import skimage 
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

def get_dir_contents(adir, filter_str=''):
    return sorted([Path(os.path.join(adir, c)) for c in os.listdir(adir) if c.endswith(filter_str)])

def get_packet(sub_c):
    ''' create a dict that will be used to create elements for each image in the XML file '''
    fb_pattern = '_s(\d\d\d)'
    nr_match = re.search(fb_pattern, sub_c.stem)
    if not nr_match:
        raise ValueError(sub_c.stem)
    
    sindex = nr_match.groups(0)[0]
    
    img = skimage.io.imread(sub_c)

    packet = {
        'filename' : sub_c.name,
        'nr' : str(int(sindex)),
        'width':str(img.shape[1]),
        'height':str(img.shape[0]),
    }
    return packet

def file_builder(packets, sub_directory):
    ''' build the packets into an XML file '''
    fn = 'filebuilderoutput.xml'
    out_path = os.path.join(sub_directory, fn)

    data = ET.Element('series')
    data.set('name', fn[:-4])

    sorted_packets = sorted(packets, key=lambda k: (int(k['nr'])))
    for packet in sorted_packets:
        elem = ET.SubElement(data, 'slice')
        for k,v in packet.items():
            elem.set(k, v)
    
    b_xml = ET.tostring(data)

    with open(out_path, "wb") as f:
        f.write(b_xml)

def match_ofn(paths, match_to):
    for i, img_path in enumerate(paths):
        fn = img_path.stem
        if match_to == fn:
            return i
    raise ValueError(match_to)

'''
#################################################################################################

The purpose of this program is to create a the xml file needed to open images in quicknii
this is done on the resized images folder that was used to define image ordering
this happens after ordering, image order is infered from the image order .xlsx file
the point being to use quicknii to help create alignments in abba
such as getting the X and Y rotation that defines the section orientation

5/26/23 - added code to apply order to fullsize and resized images

#################################################################################################
'''

# inside resized dir is folders for each animal, inside that is the images and image order .xlsx
base_dir = r'D:\ReijmersLab\TEL\slides\processed_czi_images_cohort-4'
resized_dir = os.path.join(base_dir, 'resized')
fullsize_dir =  os.path.join(base_dir, 'fullsize')

animal_dirs_resized = get_dir_contents(resized_dir)
animal_dirs_fullsize = get_dir_contents(fullsize_dir)
assert len(animal_dirs_fullsize) > 0
assert [p.stem for p in animal_dirs_resized] == [p.stem for p in animal_dirs_fullsize]


for animal_dir_resized, animal_dir_fullsize in zip(animal_dirs_resized[1:], animal_dirs_fullsize[1:]):
    # get the xlsx file 
    xlsx_fn = get_dir_contents(animal_dir_resized, filter_str='.xlsx')[0]
    df = pd.read_excel(xlsx_fn)
    order = df['Order']
    # get the image paths
    img_paths = get_dir_contents(animal_dir_resized, filter_str='.png')
    fs_img_paths = get_dir_contents(animal_dir_fullsize, filter_str='.ome.tif')
    assert len(img_paths) > 0
    assert len(fs_img_paths) > 0
    assert [p.stem for p in img_paths] == [p.stem[:-4] for p in fs_img_paths]

    ordered_paths, ordered_paths_fs = [], []
    new_fns, new_fns_fs = [], []
    packets = []
    # iterate through order matching to fn
    for i in range(len(order)):

        # match to img_paths
        matched_index = match_ofn(img_paths, order[i])
        matched, matched_fs = img_paths[matched_index], fs_img_paths[matched_index]
        ordered_paths.append(matched)

        # get the new fn
        slide_i, slide_i_fs = f'{i*4}'.zfill(3), f'{i}'.zfill(3)
        s_ind = matched.stem.rfind('_s')+2
        czi_id = matched.stem[s_ind+3:] # e.g. _3-7
        new_fn = os.path.join(matched.parent, f'{matched.stem[:s_ind]}{slide_i}{czi_id}.png')
        new_fn_fs = os.path.join(matched_fs.parent, f'{matched_fs.stem[:s_ind]}{slide_i_fs}{czi_id}.ome.tif')
        new_fns.append(new_fn)
        new_fns_fs.append(new_fn_fs)

        # rename the file
        os.rename(matched, new_fn)
        os.rename(matched_fs, new_fn_fs)

        # get the packet
        packet = get_packet(Path(new_fn))
        packets.append(packet)

    file_builder(packets, animal_dir_resized)
    

    
# create empty folders for qupath projects on a new drive
if bool(0):
    abba_base_dir = r'H:\ABBA_projects_cohort-4'
    for animal_dir in animal_dirs_resized:
        
        adir = os.path.join(abba_base_dir, animal_dir.stem)
        print(adir)
        os.mkdir(adir)




