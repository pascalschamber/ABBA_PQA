import os
from pathlib import Path
import re
import traceback
import shutil
import random
import yaml
from copy import deepcopy
from .utils_ImgDB import ImgDB


class AnimalsContainer:
    def __init__(self, config_path='./config/animal_data_config.yml', quiet=True):
        '''
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        DESCRIPTION
            handle various datatypes from different animals and cohorts, align files by s number 
        
        CONFIG
            The following parameters must be set set to define where to look for images and various file types (file derivatives)
            NOTE: all files attached to an animal must include "_sXXX_" somewhere in the filename, where X is a digit
            
            image_data_base_dir --> directory to look for images, must contain dirs for each cohort 
            image_dir_names --> dirs inside each cohort directory that contain different image types, each dir must contain folder for each animal
            animal_id_prefix --> string that prefixes animal id number (e.g. 'animal' in 'animal67')
            qupath_subdirs --> directories containing derivative files, should be stored in each animal's qupath project folder
            align_files_dict --> dictionary mapping qupath subdirs to a list of file suffixes to extract from that folder
                                    if multiple elements in list, creates seperate pathslist property to align
                                    first key is the key all others align to, key must be in qupath_subdirs
            cohorts (list[dict]) --> dictionary for each cohort
                cohort_name (str) --> name assigned
                id_range (list[int]) --> list of animal ids 
                image_dir (str) --> name of dir in image_dir_names containing folders for each animal
                abba_projects_by_animal_dir (str) --> path to directory containing a folder for each animal (where qupath data is)
                file_derivatives_dirname (None, str) --> optional, if qupath project folder (file derivatives) nested in animal's directory
            
        ARGUMENTS
            config_path (str) --> path to config .json file
            quiet (bool) --> suppress printing of warnings for alignment mismatches


        EXAMPLE of file structure
            - 1 cohort, 1 animal, with 2 image data types ('fullsize' and 'resized'), all contained in directory "D:/myProject/images"
            - qupath_project data is in another dir "D:/myProject/qupath_data_byAnimal", and each project contains a folder for 
                'counts' which contains a .csv file for each fullsize image
 
            if path to "animal1"s fullsize images is D:/myProject/images/cohort1_images/fullsize/animal1
            if path to "animal1"s 'counts' is D:/myProject/qupath_data_byAnimal/animal1/counts
            params would be: 
                image_data_base_dir = "D:/myProject/images", image_dir_names = ['fullsize', 'resized'], animal_id_prefix='animal'
                qupath_subdirs = {'counts_dir':'counts'}
                align_files_dict = {'fullsize':['.tif'], 'resized':['.png'], 'counts_dir':['.csv']}
                cohorts = [{'cohort_name': 'cohort1', 'id_range':[1]}, 'image_dir':'cohort1_images', 'abba_projects_by_animal_dir':'D:/myProject/qupath_data_byAnimal', 'file_derivatives_dirname':None]

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        # ARGUMENTS
        ###########
        self.quiet = quiet # suppress printing of warnings
        self.config_path = config_path

        # load and set config args
        #################################
        assert os.path.exists(self.config_path), f'config path does not exist {self.config_path}'
        with open(self.config_path, 'r') as file:
            config_args = yaml.safe_load(file)
            
        # parse AnimalsContainer args
        animal_container_args = deepcopy(config_args.get('AnimalsContainer'))
        assert animal_container_args is not None, 'animal container args not found, ensure config file has key \'AnimalsContainer\'.'
        supported_args = [
            'image_data_base_dir', 'image_dir_names', 'animal_id_prefix', 'qupath_subdirs', 'align_files_dict', 'cohorts'
        ]
        for arg in supported_args:
            if arg not in animal_container_args:
                print(f'WARN: {arg} not found in config file', flush=True)
            else:
                setattr(self, arg, animal_container_args[arg])
        
        # parse ImgDB args
        imgdb_args = deepcopy(config_args.get('ImgDB'))
        assert imgdb_args is not None, 'ImgDB args not found, ensure config file has key \'ImgDB\'.'
        self.ImgDB = ImgDB(**imgdb_args)

        
    def __str__(self):
        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith('__')]
        return '\n'.join([f'{a}: {getattr(self, a)}' for a in attributes])

    def __getitem__(self, key):
        ''' 
        accepts a int which indexes directly into the list of objects 
            or a str with indexes by animal name (can start with animal_id_prefix or not)
        '''
        input_key = key
        if isinstance(key, int):
            return self.animals_list[key]
        elif isinstance(key, str):
            key = key.lower()
            if key in self.cohort_names: # parse list of cohorts
                return self.get_animals(key)
            
            if key.strip().isdigit():
                key = self.animal_id_prefix.lower() + key
            if key.startswith(self.animal_id_prefix.lower()):
                return(self[self.animal_id_dict[key]])
            else:
                raise ValueError(f'cannot handle str {key} as index, input_key:{input_key}')
            
        else:
            raise ValueError(f'cannot handle key {key} as index, input_key:{input_key}')
            


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

    def get_contents(self, adir, filter_str='', startswith=False, endswith=True):
        """
        Filter a list of strings based on a set of filter strings.

        Args:
            adir (list): A directory to read contents as list of strings to be filtered.
            filter_str (list) or (str): A list of filter strings or single string.
            startswith (bool, optional): Determines whether to match the filter strings only at the begining of the input strings.
                If True, the filter strings will only be matched to the end of the input strings using the `endswith` method.
            endswith (bool, optional): Determines whether to match the filter strings only at the end of the input strings.
                If True, the filter strings will only be matched to the end of the input strings using the `endswith` method.
                If both endswith and starts with are True, will default to startswith
                If both endswith and starts with are False, the filter strings will be matched anywhere within the input strings 
                    using the `in` operator.
        Returns:
            list: A new list containing the filtered strings.
        """
        filters = [filter_str] if isinstance(filter_str, str) else filter_str # allow filters to be a single string as well
        # determine match function 
        match_func = str.startswith if startswith else str.endswith if endswith else str.__contains__
        filtered_paths = sorted([os.path.join(adir,s) for s in os.listdir(adir) if any(match_func(s, f) for f in filters)])
        assert len(filtered_paths) != 0, f'no content found for {adir} using filters {filters}\ncurrent content:\n{os.listdir(adir)}'
        return filtered_paths


    def init_animals(self, failOnError=True):
        ''' 
        initialize all animal objects 
        '''
        animals_list = []
        self.cohort_names = []
        for cohort in self.cohorts:
            file_derivative_dir = cohort['abba_projects_by_animal_dir']
            file_derivatives_dirname = cohort['file_derivatives_dirname']

            if not os.path.exists(file_derivative_dir):
                print(f"WARN --> cannot find {cohort['cohort_name']}'s file derivative folder ({file_derivative_dir}), data not added for all animals in this cohort")
                file_derivative_dir = os.path.join(self.image_data_base_dir, cohort.get('image_dir'), self.image_dir_names[0])
                file_derivatives_dirname = None
            self.cohort_names.append(cohort['cohort_name'])

            animal_dirs = self.get_contents(file_derivative_dir, filter_str=self.animal_id_prefix, startswith=True)
            for animal_dir in animal_dirs:
                
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
        assert isinstance(animal_id, str), f'cannot convert {animal_id} to int, not a string'
        assert animal_id.lower().startswith(self.animal_id_prefix.lower()), f'cannot convert {animal_id} to int, animal_id_prefix does not match'
        assert animal_id.lower().replace(self.animal_id_prefix.lower(), '').isdigit(), f'cannot convert {animal_id} to int, does not contain only prefix and digits'
        return int(animal_id.lower().replace(self.animal_id_prefix.lower(), ''))

    def get_animal_cohort(self, animal_id):
        ''' returns cohort dictionary for a given animal id '''
        for cohort in self.cohorts:
            cohort_ids = cohort.get('id_range')
            if self.animal_id_to_int(animal_id) in cohort_ids:
                return cohort
        raise ValueError(f'could not infer cohort for animal {animal_id}')
    
    def maybe_flatten_list(self, lst):
        """
        Flattens a nested list if provided. If the list is not nested, simply returns the list.
        
        Args:
            lst (list): Input list which may or may not be nested.

        Returns:
            list: Flattened list if nested, or the original list if not nested.
        """
        flattened_list = []
        for item in lst:
            if isinstance(item, list):
                flattened_list.extend(self.maybe_flatten_list(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def get_animals(self, filter):
        ''' 
        returns a list of animal objects described by the filter
        params:
            filter: (list, int, str) 
                e.g. [0,1,2...], 4, 'cohort2', 'TEL25', or ['cohort2', 'cohort3', 'cohort4']
        '''
        assert hasattr(self, 'cohort_names'), 'need to initialize AnimalsContainer by calling .init_animals() first'
        if isinstance(filter, int):
            return self[filter]
        elif isinstance(filter, list):
            return self.maybe_flatten_list([self[el] for el in filter])
        elif isinstance(filter, str):
            if filter.strip().isdigit():
                return self[filter]
            elif filter.lower().startswith(self.animal_id_prefix.lower()):
                return self[filter]
            elif filter.lower().startswith('cohort'):
                cohort_ids = [chrt.get('id_range') for chrt in self.cohorts if chrt.get('cohort_name')==filter.lower()][0]
                animal_ids_to_get = [f'{id_int}' for id_int in cohort_ids]
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
            subdir_path = os.path.join(self.image_data_base_dir, self.cohort.get('image_dir'), subdir_name, self.animal_id)
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
    
    def sample_list(self, alist, sample):
        ''' sample (tup[int, int, (int)]) --> only get image paths in this range '''
        assert isinstance(sample, tuple), f'sample must be of type tuple, got type: {type(sample)}'
        sampled_list = []
        for i in list(range(*sample)):
            try: sampled_list.append(alist[i])
            except IndexError: pass
        if len(sampled_list) == 0: 
            print('WARN --> sampled_list is empty, returning empty list.')
        return sampled_list

    def get_valid_datums(self, filetypes, warn=True, SAMPLE=None, SHUFFLE=False):
        ''' 
            get list of datums that have the filetypes specified
            returns list of datums if each filetype specified is not not none, i.e. filetype was aligned and exists 
            e.g. for image_to_df filetypes would be ['fullsize_paths', 'quant_dir_paths', 'geojson_regions_dir_paths']
            ARGS
                - filetypes (list[str]) --> check if these attributes are valid
                - warn (bool) --> whether to print datums that are being skipped
                - SAMPLE (tup[int, int, (int)]) --> only get image paths in this range, tup is passed to range()
                - SHUFFLE (bool) --> whether to shuffle the returned datums
            TODO 
                - implement skip already completed, e.g. provided an attribute, check if datum has that attribute, skip if attribute is not None 
        '''
        assert isinstance(filetypes, list), f'filetypes must be of type list, got type: {type(filetypes)}'
        output = []
        for datum in self.d:
            if any ([(getattr(datum, filetype) is None) for filetype in filetypes]):
                if warn: print(f'skipping datum: {Path(datum.fullsize_paths).stem[:-4]}')
                continue
            else:
                output.append(datum)
        if len(output) == 0: 
            print('WARN --> no valid datums, returning empty list.'); return output
        
        if SAMPLE is not None: output = self.sample_list(output, SAMPLE)
        if SHUFFLE: random.shuffle(output)

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
    

    



        




