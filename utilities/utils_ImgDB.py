from copy import deepcopy

class ImgDB:
    def __init__(self, **kwargs):
        self.image_channels = kwargs.get('image_channels')
        self.colocal_nuclei_info = kwargs.get('colocal_nuclei_info')
        self.normalization_params = kwargs.get('normalization_params')
        self.threshold_params = kwargs.get('threshold_params')
        self.colocalid_ch_map = {}
        self.colocal_ids = {}
        self.colocalizations = [] # list of dicts: coIds=(1,2), coChs=(0,1), assign_colocal_id=3, {3:{'intersecting_label_column':'ch0_intersecting_label', 'intersecting_colocal_id':1}}
        self.ingest_kwargs()
    
    def ingest_kwargs(self):
        # get assigned mapping of image channels to colocalids
        for ch_dict in self.image_channels:
            clc_id = self.check_not_none(ch_dict.get("colocal_id"))
            assert clc_id not in self.colocal_ids, f"colocal_id must be unique, got {clc_id} but have {self.colocal_ids}"
            self.colocalid_ch_map[self.check_not_none(ch_dict.get("ch_idx"))] = clc_id
            self.colocal_ids[clc_id] = {k:v for k,v in ch_dict if k!="colocal_id"}
        
        # parse colocal nuclei info 
        if self.colocal_nuclei_info is not None:
            for clc_dict in self.colocal_nuclei_info:
                coChs = self.check_not_none(clc_dict.get("ch_idx"))
                if len(coChs) != 2:
                    raise ValueError (f"colocalization must be between 2 channels, got {coChs}")
                coIds = [self.get_colocal_id_from_ch_idx(ch) for ch in coChs]
                # ensure assigned_colocal_id is unique
                assign_colocal_id = self.check_not_none(clc_dict.get("colocal_id"))
                if assign_colocal_id in self.colocal_ids:
                    raise ValueError (f"colocal_id must be unique, got {assign_colocal_id} but have {self.colocal_ids}")
                self.colocal_ids[assign_colocal_id] = {k:v for k,v in clc_dict if k!="colocal_id"}
                self.colocalizations.append({'coChs':coChs, 'coIds':coIds, 'assign_colocal_id':assign_colocal_id, 
                                             'intersecting_label_column':f"ch{coChs[0]}_intersecting_label", "intersecting_colocal_id":coIds[0], 
                                             "other_intensity_name":f"{self.colocal_ids[coIds[1]]['name']}_intensity"})
        # sort colocal ids
        self.sort_colocal_ids()
        self.reformat_json_params('normalization_params')
        self.reformat_json_params('threshold_params')
        
    def sort_colocal_ids(self):
        setattr(self, 'colocal_ids', {k: self.colocal_ids[k] for k in sorted(self.colocal_ids)})

    def reformat_json_params(self, param_attr):
        # reformat param dicts so keys are int not strs (.json requires them to be strings)
        old_params, new_params = getattr(self, param_attr), {}
        for cohort, param_dicts in old_params.items():
            new_params[cohort] = {}
            for ch, param_dict in param_dicts.items():
                new_params[cohort][int(ch)] = deepcopy(param_dict)
        delattr(self, param_attr)
        setattr(self, param_attr, new_params)
            
    def get_colocal_id_from_ch_idx(self, ch_idx):
        val = self.colocalid_ch_map.get(ch_idx)
        if val is not None:
            return val
        raise KeyError(f"cannot find {ch_idx} in colocalid_ch_map")
    
    def get_count_channel_name_from_colocal_id():
        pass

    def check_not_none(self, var):
        if var is None:
            raise KeyError(f"{var} is None, ensure it is properly defined in config file.")
        return var

    def check_exists(self, attr):
        if not hasattr(self, attr):
            raise KeyError(f"{attr} not found, ensure it is defined in config file.")
        elif getattr(self, attr) is None:
            raise KeyError(f"{attr} is None, ensure it is defined in config file.")
        else:
            return 0  # all good

    def get_normalization_params(self):
        # for prediction, returns dict mapping cohorts to channel normalization values
        self.check_exists("normalization_params")
        return self.normalization_params

    def get_colocalid_ch_map(self):
        # for img2df, returns dict mapping channels in intensity image to colocal id
        self.check_exists("colocalid_ch_map")
        return self.colocalid_ch_map
    
    def get_count_channel_names(self):
        self.sort_colocal_ids()
        return [f"n{self.colocal_ids[k]['name']}" for k in self.colocal_ids]
    def get_clc_nuc_info(self):
        clc_nuc_info = {}
        for coloc in self.colocalizations:
            clc_nuc_info[coloc['assign_colocal_id']] = {k:v for k,v in coloc.items() if k != 'assign_colocal_id'}
        return clc_nuc_info
        
    def get_threshold_params(self):
        self.check_exists("threshold_params")
        return self.threshold_params
