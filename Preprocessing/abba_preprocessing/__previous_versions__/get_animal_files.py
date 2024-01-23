import os
from pathlib import Path
import re
import traceback
import shutil


class AnimalsContainer:
    def __init__(self, quiet=True):
        ''' if quiet suppress printing of warnings for alignment mismatches '''
        self.quiet = quiet

        # define the base dirs
        # self.abba_projects_by_animal_dir = r'D:\ReijmersLab\TEL\slides\ABBA_projects\byAnimal' # replaced with one for each cohort
        self.fullsize_base_dir = r'D:\ReijmersLab\TEL\slides'
        
        # add dirs that exist as directories in an animals qupath folder
        self.qupath_subdirs = { 
            'atlas_mask_dir':'atlas_masks',
            'geojson_regions_dir':'qupath_export_geojson',
            'qupath_project_filepath':'project.qpproj',
            'abba_states_dir':'abba_states',
            'quant_dir':'quant',
            'counts_dir':'counts',
        }

        # add dirs that contain 'raw' images
        self.image_dir_names = ['fullsize', 'resized'] 

        # define dict of {file_dir:listOfFiletypes} that will be aligned to fullsize image (first el is key to align to)
        # key must be in qupath_subdirs
        # passing a list of filetypes creates seperate pathslist to align
        self.align_files_dict = {
            'fullsize':['.tif'], 
            'resized':['.png'], 
            'geojson_regions_dir':['.geojson'], 
            'atlas_mask_dir':['atlas_mask.tif', 'leftSide.tif'],
            'quant_dir':['nuclei.tif'],
            'counts_dir':['region_df.csv', 'rpdf.csv'], # !!! replace region_counts_df with region_df
        } 

        # define each cohort by the animal ids it contains, and where images can be found
        self.cohorts = [
            # {
            #     'cohort_name': 'cohort1',
            #     'id_range':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            #     'image_dir':os.path.join(self.fullsize_base_dir, 'processed_czi_images_cohort-1')
            # },
            {
                'cohort_name': 'cohort2',
                'id_range':[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                'image_dir':os.path.join(self.fullsize_base_dir, 'processed_czi_images'),
                'abba_projects_by_animal_dir':r'H:\fullsize\fullsize',
                'file_derivatives_dirname': 'qupath', # filestructure is different, instead of an an_dir, is in an_dir/qupath

            },
            {
                'cohort_name': 'cohort3',
                'id_range':[30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                'image_dir':os.path.join(self.fullsize_base_dir, 'processed_czi_images_cohort-3'),
                'abba_projects_by_animal_dir':r'D:\ReijmersLab\TEL\slides\ABBA_projects\byAnimal',
                'file_derivatives_dirname': None,
            },
            {
                'cohort_name': 'cohort4',
                'id_range':[50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69],
                'image_dir':os.path.join(self.fullsize_base_dir, 'processed_czi_images_cohort-4'),
                'abba_projects_by_animal_dir':r'D:\ReijmersLab\TEL\slides\ABBA_projects\byAnimal',
                'file_derivatives_dirname': None,
            },
        ]
        
    
    def __str__(self):
        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith('__')]
        return '\n'.join([f'{a}: {getattr(self, a)}' for a in attributes])

    def __getitem__(self, key):
        ''' 
        accepts a int which indexes directly into the list of objects 
            or a str with indexes by animal name (can start with TEL or not)
        '''
        if isinstance(key, int):
            return self.animals_list[key]
        elif isinstance(key, str):
            key = key.lower()
            if key.strip().isdigit():
                key = 'tel' + key
            if key.startswith('tel'):
                return(self[self.animal_id_dict[key]])
            else:
                raise ValueError(f'cannot handle str {key} as index')
            


    def set_animals_list(self, animal_objs):
        '''
        animals_list: list of animal objects
        animal_id_dict: dict mapping animal_id to thier index in animals_list
        '''
        setattr(self, 'animals_list', animal_objs)
        setattr(self, 'animal_id_dict', 
                dict(zip(
                    [o.animal_id.lower() for o in animal_objs], [i for i in range(len(self.animals_list))]
                ))
        )

    def get_contents(self, adir, filter_str='', endswith=True):
        """
        Filter a list of strings based on a set of filter strings.

        Args:
            adir (list): A directory to read contents as list of strings to be filtered.
            filter_str (list) or (str): A list of filter strings or single string.
            endswith (bool, optional): Determines whether to match the filter strings only at the end of the input strings.
                If True, the filter strings will only be matched to the end of the input strings using the `endswith` method.
                If False, the filter strings will be matched anywhere within the input strings using the `in` operator.
                Defaults to True.

        Returns:
            list: A new list containing the filtered strings.
        """
        filters = [filter_str] if isinstance(filter_str, str) else filter_str # allow filters to be a single string as well
        match_func = str.endswith if endswith else str.__contains__
        filtered_paths = sorted([os.path.join(adir,s) for s in os.listdir(adir) if any(match_func(s, f) for f in filters)])
        assert len(filtered_paths) != 0, f'no content found for {adir} using filters {filters}\ncurrent content:\n{os.listdir(adir)}'
        return filtered_paths


    def init_animals(self, failOnError=True):
        ''' 
        initialize all animal objects 
        '''
        animals_list = []
        for cohort in self.cohorts:
            animal_dirs = self.get_contents(cohort['abba_projects_by_animal_dir'])
            for animal_dir in animal_dirs:
                file_derivatives_dirname = cohort['file_derivatives_dirname']
                if file_derivatives_dirname is not None:
                    animal_dir = os.path.join(animal_dir, file_derivatives_dirname)

                try:
                    an = AnimalDirectory(animal_dir, file_derivatives_dirname=file_derivatives_dirname)
                    animals_list.append(an)
                except:
                    emessage = f'could not create AnimalDirectory for: {animal_dir}'
                    if failOnError:
                        raise ValueError(emessage)
                    else:
                        print(emessage)
        self.set_animals_list(animals_list)
    
    def animal_id_to_int(self, animal_id):
        assert isinstance(animal_id, str), f'{animal_id}'
        assert animal_id.lower().startswith('tel'), f'{animal_id}'
        assert animal_id[3:].isdigit()
        return int(animal_id[3:])

    def get_animal_cohort(self, animal_id):
        for cohort in self.cohorts:
            cohort_ids = cohort.get('id_range')
            if self.animal_id_to_int(animal_id) in cohort_ids:
                return cohort
        raise ValueError(f'could not infer cohort for animal {animal_id}')
    
    def get_animals(self, filter):
        ''' 
        returns a list of animal objects described by the filter
        params:
            filter: (list, int, str) e.g. [0,1,2...], 4, 'cohort2' or 'TEL25'
        '''
        if isinstance(filter, int):
            return self[filter]
        elif isinstance(filter, list):
            return [self[el] for el in filter]
        elif isinstance(filter, str):
            if filter.lower().startswith('tel'):
                return self[filter]
            elif filter.lower().startswith('cohort'):
                cohort_ids = [chrt.get('id_range') for chrt in self.cohorts if chrt.get('cohort_name')==filter.lower()][0]
                animal_ids_to_get = [f'tel{id_int}' for id_int in cohort_ids]
                return [self[el] for el in animal_ids_to_get]
            else:
                raise ValueError (f'cannot parse str filter {filter}')
        else:
            raise ValueError (f'cannot parse filter {filter} of type {type(filter)}')
    
    def clean_animal_dir(self, animals, dir_name):
        dirs_to_remove = []
        for an in animals:
            dir_to_remove = os.path.join(an.base_dir, dir_name)
            print(dir_to_remove, len(os.listdir(dir_to_remove)))
            dirs_to_remove.append(dir_to_remove)

        while True:
            confirm = input('press \'y\' to confirm deleting previous results, \'n\' to quit: ')
            if confirm == 'y':
                # delete a specific directory for a list of animals, e.g. clean_animal_dir(animals, 'counts')
                for adir in dirs_to_remove:
                    shutil.rmtree(adir)
                print('deleted.')
                break
            elif confirm == 'n':
                print('exiting..')
                break
        return 0
        
    
    
    





class AnimalDirectory(AnimalsContainer):
    def __init__(self, animal_base_dir, file_derivatives_dirname=None):
        '''
            Parameters
                - file_derivatives_dirname (str), was added if qupathfiles were not in animal dir but nested inside it e.g. in cohort 2's 'qupath' folder
        '''
        super().__init__()
        self.parent_attrs = dir(self) # those attributes inherited from parent, setting here b/c to filter them out when printing self
        self.base_dir = Path(animal_base_dir)
        self.animal_id = self.base_dir.stem if file_derivatives_dirname is None else self.base_dir.parent.name
        self.cohort = super().get_animal_cohort(self.animal_id)
        self.cohort_name = self.cohort.get('cohort_name')
        

        try:
            # attach other dirs from its qupath project folder
            if self.qupath_subdirs:
                self.add_subdirs(self.qupath_subdirs, makedirs=False)

            # try to find fullsize and resized image dirs
            self.get_image_dirs()

            # try to get the contents of dirs we want to align to fullsize images
            self.align_get_dir_contents()

            # try to align the image file derivatives defined in self.align_files_dict to the fullsize image
            self.align_image_file_derivatives()

        except ValueError as e:
            print(f"Error occurred during initialization:")
            traceback.print_exc()
            return
    

    def __str__(self):
        """
        Return a string representation of the object, including its attributes and their values, without methods and those in parent class or private.
        """
        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr))
                    and not attr.startswith('__') and attr not in self.parent_attrs]
        return f'{self.animal_id}\n' + '-_'*50 + '\n' + '\n'.join([f"{attr} --> {getattr(self, attr)}" for attr in attributes])
    

    def add_subdirs(self, dirdict, makedirs=False):
        ''' attach subdirs as attributes of animal, if makedirs, if dir doesn't exist create it otherwise it just gets attached'''
        for attrname, path_rel_to_basedir in dirdict.items():
            to_add = os.path.join(self.base_dir, path_rel_to_basedir)
            if not makedirs:
                try:
                    assert os.path.exists(to_add), f'{to_add} does not exist'
                except AssertionError:
                    to_add = None
            else:
                if not os.path.exists(to_add):
                    print(f'creating dir at: {to_add}')
                    os.makedirs(to_add)
            setattr(self, attrname, to_add)
    
    

    def get_image_dirs(self):
        ''' for each subdirectory specified under the base image dir, get an animals images '''
        for subdir_name in self.image_dir_names:
            subdir_path = os.path.join(self.cohort.get('image_dir'), subdir_name, self.animal_id)
            if not os.path.exists(subdir_path): raise ValueError(f'image sub dir does not exist: {subdir_path}')
            self.add_subdirs({f'{subdir_name}':subdir_path}, makedirs=False)
            
    
    def align_get_dir_contents(self):
        ''' for each filetype listed in align_files_dict get the paths from that dir if filetype matches '''
        self.dir_attrs_to_align = []
        self.align_get_dir_contents_errors = []
        for dir_name, filetypes in self.align_files_dict.items():
            
            for filetype in filetypes:
                # set the name of the new attr holding the paths
                attr_name_paths = dir_name if len(filetypes) == 1 else filetype.split('.')[0] 
                attr_name_paths = f'{attr_name_paths}_paths'

                try: # add the paths as attributes
                    paths_list = self.get_contents(getattr(self, dir_name), filter_str=[filetype], endswith=True)    
                    setattr(self, attr_name_paths, paths_list)
                    
                except (TypeError, AssertionError) as e:
                    self.align_get_dir_contents_errors.append(f'could not add {attr_name_paths} for {self.animal_id}')
                    setattr(self, attr_name_paths, None)
                    
                self.dir_attrs_to_align.append(attr_name_paths) # store the new attribute name of the pathslist
                
    
    def align_initial_check(self):
        ''' inspect number of pathlists that were successfully fetched, if only 1 i.e. only fullsize do not run alignment '''
        
        warn_str_base = f'WARN: could not align image file derivatives for {self.animal_id}'
        error_messages = '\n'.join(self.align_get_dir_contents_errors)

        numNone, self.numAlignSuccess = sum(x is None for x in self.dir_attrs_to_align), sum(x is not None for x in self.dir_attrs_to_align)

        try: # exit alignment cases
            if self.numAlignSuccess < 2: # none to align
                raise ValueError(f'{warn_str_base} reason: too few dir_attrs_to_align contents were successfully added:\n{error_messages}')
            elif self.numAlignSuccess == 0: # couldn't fetch anything, Something is wrong
                raise ValueError(f'{warn_str_base} reason: dir_attrs_to_align is empty:\n{error_messages}')
        except (ValueError, AssertionError) as e:
            print(e)
            return False
        
        if numNone > 0: # something wasn't fetched, but is okay to run alignment
            print(f'{warn_str_base} reason: PARTIAL SUCCESS {numNone} dir_attrs_to_align contents were not fetched:\n{error_messages}')
        return True # full or partial success


    def align_image_file_derivatives(self):
        if not self.align_initial_check():
            self.d = None
            return
        # create a dict of attr:value to pass as kwargs to AnimalData to align to fullsize image
        fd_attrs = {fd_attr : getattr(self, fd_attr) for fd_attr in self.dir_attrs_to_align}
        # self.d = AnimalData(self.animal_id, self.fullsize_paths, self.resized_paths, self.geojson_regions_dir_paths)
        self.d = AnimalData(self.animal_id, self.quiet, **fd_attrs)
        # self.imgs = self.d.imgs
    
    
    def get_valid_datums(self, filetypes, warn=True):
        ''' 
            get list of datums that have the filetypes specified
            returns list of datums if each filetype specified is not not none, i.e. filetype was aligned and exists 
            e.g. for image_to_df filetypes would be ['fullsize_paths', 'quant_dir_paths', 'geojson_regions_dir_paths']
        '''
        output = []
        for datum in self.d:
            if any ([(getattr(datum, filetype) is None) for filetype in filetypes]):
                if warn: print(f'skipping datum: {Path(datum.fullsize_paths).stem[:-4]}')
                continue
            else:
                output.append(datum)
        return output
        
        
    

    

class AnimalData:
    ''' 
        class to hold the variety of files generated from the images for a given animal
        purpose is to align them by s_number or dropping them if a file type doesn't exist across all types
        TODO: implement a method to drop certain images we want to exclude
        TODO: make boolean attributes for each aligned type, where it has it or not is easily accessible
    '''
    def __init__(self, animal_id, quiet, **kwargs):
        self.animal_id = animal_id
        self.quiet = quiet
        
        # store attributes to align and set key for alignment
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.fd_attr_names = list(kwargs.keys()) # attrs to align to key (fullsize)
        self.fd_attr_key = self.fd_attr_names.pop(0) # key to align with
        
        self.align()
    
    def __str__(self):
        ''' prints attributes that were aligned and the number of images for each type '''
        aligmnet_counts_str = '\n'.join([f'{a}: {c}' for a,c in self.get_alignment_counts().items()])
        return f'{self.animal_id} has {len(self.imgs)} imgs, file derivatives fully aligned {self.numFullAlignedTypes}/{len(self.fd_attr_names)}:\n{aligmnet_counts_str}'
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i]
    
    def __iter__(self):
        return iter(self.imgs)
    
    def get_datum(self, snum):
        """ func to get a single datum from a provided snumber"""
        snum = str(snum).zfill(3)
        for datum in self.imgs:
            if snum == self.get_s_number(datum.fullsize_paths):
                return datum
    
    def get_paths(self, key):
        return [getattr(d, key) for d in self.imgs]
    
    def get_alignment_counts(self):
        return {attr:sum(x is not None for x in list(attrVal.values())) for attr, attrVal in self.to_align.items()}

    def align(self):

        # align filepaths by s number if contents were grabbed, else attr value is list of none
        self.to_align = {}
        key_value = getattr(self, self.fd_attr_key)
        for fd_attr in self.fd_attr_names: # iterate through all except key
            paths_to_align = getattr(self, fd_attr)
            if paths_to_align is not None:
                aligned_paths = self.align_filepaths_by_s_number(paths_to_align, key_value)
            else:
                aligned_paths = dict(zip(key_value, [None]*len(key_value)))
            self.to_align[fd_attr] = aligned_paths
        self.numFullAlignedTypes = sum(v == len(key_value) for k,v in self.get_alignment_counts().items())
       
        self.imgs = [Datum(kws) for kws in self.merge_alignments()]
        

    
    def merge_alignments(self) -> list:
        ''' join the aligned images by merging the dicts aligned by s number using fullsize image path as the key '''
        merged_image_dicts = []
        for key_attr_value in getattr(self, self.fd_attr_key): # for each fullsize image create a dict of all aligned file derivates
            img_dict = {self.fd_attr_key:key_attr_value}
            for to_align_key, to_align_dict in self.to_align.items():
                img_dict[to_align_key] = to_align_dict[key_attr_value]
            merged_image_dicts.append(img_dict)
        return merged_image_dicts

        
    
    def get_s_number(self, apath) -> str:
        ''' 
        helper function that extracts the s000 number from a file path 
        '''
        pattern = '.*_s(\d\d\d)_*.*'
        p = Path(apath).stem if isinstance(apath, str) else apath.stem
        match = re.match(pattern, p)
        if match:
            out_val = match.groups(0)[0]
            return out_val
        raise ValueError(f'could not find s number for: {p}')
    
    def get_s_numbers(self, paths) -> list:
        def check_for_resized_img_paths(snumbers):
            def check_multiples_of_4(snumbers): # for sized paths where s number is a multiple of 4
                return all(int(s) % 4 == 0 for s in snumbers if s.isdigit())
            if check_multiples_of_4(snumbers):
                return [str(int(s)//4).zfill(3) for s in snumbers]
            return snumbers
        return check_for_resized_img_paths([self.get_s_number(p) for p in paths])
    

    def check_lengths_match(self, list1, list2):
        if len(list1) != len(list2):
            print(f"Warning: for {self.animal_id}: Lengths of list1 and list2 (align to) do not match! (Difference in lengths: {len(list1) - len(list2)})")
        elements_not_in_second = set(list1) - set(list2)
        if elements_not_in_second:
            print(f"{self.animal_id} has Elements in {list1} not in {list2}:")
            print(", ".join(str(elem) for elem in elements_not_in_second))
        elements_not_in_first = set(list2) - set(list1)
        if elements_not_in_first:
            print(f"{self.animal_id} Elements in list2 not in list1:")
            print(", ".join(str(elem) for elem in elements_not_in_first))


    def align_filepaths_by_s_number(self, pathlist, pathlist_align_to) -> dict:
        ''' for path in pathlist make sure it matches to align_to, removes those not present '''
        s_numbers = self.get_s_numbers(pathlist)
        mask_s_numbers = self.get_s_numbers(pathlist_align_to)
        
        if not self.quiet:
            self.check_lengths_match(s_numbers, mask_s_numbers)

        # match the elements in align to to the filepaths in pathlist
        matching_dict = {}
        for i in range(len(pathlist_align_to)):
            matching_dict[pathlist_align_to[i]] = pathlist[s_numbers.index(mask_s_numbers[i])] if mask_s_numbers[i] in s_numbers else None
        return matching_dict


class Datum:
    def __init__(self, kwargs):
        self.data_points = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.data_points.append(k)
    
    def __str__(self):
        return '\n'.join(f'{dp}: {getattr(self, dp)}' for dp in self.data_points)
    
    

def test_basic_functions(ac):
    # test the filtering function
    animals = ac.get_animals('cohort3')
    print([an.animal_id for an in animals])
    # test dirs were attached 
    print([an.fullsize for an in animals])
    print([an.geojson_regions_dir for an in animals])




if __name__ == '__main__':
    # initialization
    ac = AnimalsContainer()
    ac.init_animals()

    animals = ac.get_animals('cohort4')
    # print(animals[0])
    # an = animals[0]

    for an in animals[:]:
        print(an.d)
    
    for d in an.d:
        print(d.fullsize_paths, d.resized_paths)
        assert os.path.exists(d.fullsize_paths)
        assert os.path.exists(d.resized_paths)
    

    



        




