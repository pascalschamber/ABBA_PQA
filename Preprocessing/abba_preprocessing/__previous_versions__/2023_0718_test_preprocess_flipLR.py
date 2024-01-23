import tkinter as tk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import json
from pathlib import Path
import os
import scipy.ndimage as ndi
import numpy as np

from get_animal_files import AnimalsContainer


class ImageProcessor:
    '''
        Description
        ~~~~~~~~~~~~~
            - goal is to prepare the images for ABBA by flipping and rotating before hand instead of in abba since it is cumbersome.
            - the results are saved in a json file that is then passed to another program that applies the transforms
                to the resized image, fullsize image, and creates an ome.tiff as 8bit for use in ABBA.
    '''
    def __init__(self, image_paths, st_index=0):
        self.image_paths = image_paths
        self.num_images = len(self.image_paths)
        self.output_path = os.path.join(str(Path(image_paths[0]).parent), 'preABBA_image_transforms.json')
        self.index = st_index

        self.image_labels = {}
        self.flip = False
        self.rotate_val = 0

        # load previous if exists
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                self.image_labels = json.load(f)
            print(f'loaded previous labels:\n{self.image_labels}\n')
        else:
            # init the flip tracking
            for p in self.image_paths:
                self.image_labels[p] = {'flip':False, 'rotate_val':0}

        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)  # Make the window go to fullscreen mode

        # Get screen width and height
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, width=self.screen_width, height=self.screen_height)
        self.canvas.pack()

        # bindings
        self.root.bind('<Escape>', self.escape)
        self.root.bind('<space>', self.keep_image)
        self.root.bind('q', self.previous_image)
        self.root.bind('f', self.flip_image)
        self.root.bind('w', self.rotate_image_L5)
        self.root.bind('e', self.rotate_image_L)
        self.root.bind('r', self.rotate_image_R)
        self.root.bind('t', self.rotate_image_R5)

        # grid overlay params
        self.grid_spacing = 150
        self.grid_opacity = 0.1
        self.font = ImageFont.load_default()
        self.text_pos = (50, 50)
        self.text_color = (255, 255, 255)
        
        self.show_image()
        self.root.mainloop()


    


    def show_image(self):
        img = Image.open(self.image_paths[self.index])
        if self.flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.rotate_val != 0:
            img = self.rotate(img, self.rotate_val)
        img = img.resize((self.screen_width, self.screen_height), Image.Resampling.LANCZOS)  # Resize the image to screen size
        img = self.make_grid_overlay(img)
        img = self.draw_text_on_image(img, f'{self.index} / {self.num_images-1}\n(f={self.flip}, r={self.rotate_val})')

        tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        self.canvas.image = tk_img
    
    def rotate(self, PILimage, rotate_val):
        arr = ndi.rotate(np.array(PILimage), rotate_val, axes=(1,0), reshape=False, order=3, prefilter=True)
        return Image.fromarray(arr)
    
    def make_grid_overlay(self, PILimg):
        # Create an identical array
        image_np = np.array(PILimg)
        height, width, _ = image_np.shape # Get image size
        overlay = np.copy(image_np) 

        # Create grid
        grid_size = self.grid_spacing
        for i in range(0, width, grid_size):
            overlay[:, i, :] = [255, 255, 255]
        for i in range(0, height, grid_size):
            overlay[i, :, :] = [255, 255, 255]

        # Combine original image and overlay
        image_new = ((1.0 - self.grid_opacity) * image_np + self.grid_opacity * overlay).astype(np.uint8)
        return Image.fromarray(image_new)
    
    def draw_text_on_image(self, img, text):
        # Create ImageDraw object
        draw = ImageDraw.Draw(img)
        # Draw text on image
        draw.text(self.text_pos, str(text), font=self.font, fill=self.text_color) # text color in RGB
        return img


    def get_index_flip(self, idx):
        return self.image_labels[self.image_paths[idx]]['flip']
    def get_index_rotation(self, idx):
        return self.image_labels[self.image_paths[idx]]['rotate_val']
    def set_index_flip(self):
        self.image_labels[self.image_paths[self.index]]['flip'] = self.flip
    def set_index_rotation(self):
        self.image_labels[self.image_paths[self.index]]['rotate_val'] = self.rotate_val
    
    
    def keep_image(self, event):
        # save current values
        self.set_index_flip()
        self.set_index_rotation()

        # check for exit condition, when reach end of list
        if self.index + 1 >= self.num_images:
            self.shutdown()
        else:
            # get previous image index
            self.index += 1
            self.flip = self.get_index_flip(self.index)
            self.rotate_val = self.get_index_rotation(self.index)
            self.show_image()
    
            
    def previous_image(self, event):
        # save current values
        self.set_index_flip()
        self.set_index_rotation()

        # get previous image index
        self.index -=1 if self.index > 0 else 0
        self.flip = self.get_index_flip(self.index)
        self.rotate_val = self.get_index_rotation(self.index)
        self.show_image()

    def flip_image(self, event):
        self.flip = True if self.flip is False else False        
        self.show_image()
    
    
    def rotate_image_L(self, event):
        self.rotate_val += 1
        self.show_image()
    def rotate_image_R(self, event):
        self.rotate_val -= 1
        self.show_image()
    def rotate_image_L5(self, event):
        self.rotate_val += 5
        self.show_image()
    def rotate_image_R5(self, event):
        self.rotate_val -= 5
        self.show_image()



    
    
    def save_labels(self):
        with open(self.output_path, 'w') as f:
            json.dump(self.image_labels, f)
        print(f'labels saved.\n{self.output_path}\n')

    def shutdown(self):
        self.save_labels()
        self.root.destroy()
        self.root.quit()

    def escape(self, event):
        self.shutdown()
        
# TODO
# see if i can wrap rotation into one function, where event key dictates rotation angle


ac = AnimalsContainer()
ac.init_animals()
animals = ac.get_animals('cohort4')

# note
# check out this fs img to see if mask is applied correctly: 58-s31
# last finished 8 --> TEL58 07/19/23

# 8123 - TEL60 may have some pieces flipped wrong
 
an_i = 18
st_index = 0 
for an in animals[an_i:an_i+1]:
    resized_image_paths = [d.resized_paths for d in an.get_valid_datums(['resized_paths'])]
    print(an_i, '-->', an.animal_id, len(resized_image_paths))
    
    if bool(1):
        app = ImageProcessor(resized_image_paths, st_index=st_index) 

