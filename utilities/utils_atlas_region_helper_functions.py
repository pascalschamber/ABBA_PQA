import json
from collections import deque


'''
############################################################################################################
    DESCRIPTION
    ~~~~~~~~~~~
        helper functions to load an ontology json file, to extract atlas ids and the various properties
    
    NOTES
    ~~~~~
        5/29 - updated load_ontology and extract_ont_label_key to allow parsing of abba v3p1 ontology, 
            organization is different as all attrs we care about except children list are contained 
            in attribute named data instead of just keys in the dict
    
    ONTOLOGY DESCTIPTIONS
    ~~~~~~~~~~~~~~~~~~~~~
    - regions by st_level for Basic cell groups and regions, much higher when including all 4 st_level 1's
        1--> nRegions: 1
        2--> nRegions: 3
        3--> nRegions: 4
        4--> nRegions: 1
        5--> nRegions: 13
        6--> nRegions: 34
        7--> nRegions: 19
        8--> nRegions: 298
        9--> nRegions: 177
        10--> nRegions: 46
        11--> nRegions: 507
############################################################################################################
'''
class Ontology:
    def __init__(self):
        self.ont_ids = load_ontology()
        self.names_dict = dict(zip([d['name'] for d in self.ont_ids.values()], self.ont_ids.keys()))
        
        # st5 color map
        self.parent_level_colormap = {
        'Cortical subplate': '#3288bd', 'Isocortex': '#66c2a5', 'Olfactory areas': '#abdda4', 'Pallidum': '#e6f598', 'Hippocampal formation': '#ffffbf', 'Striatum': '#fee08b', 'Thalamus': '#fdae61', 'Hypothalamus': '#f46d43', 'Midbrain': '#d53e4f',
        'Pons':'#980043', 'Medulla':'#756bb1', 'Cerebellar cortex':'#d9d9d9', 'Cerebellar nuclei':'#969696',
        }

    def map_region_to_parent_st_level(self, region_names):
        output = {} # dict mapping parent at a specific st_level to children region names
        for reg in region_names:
            pass




def print_possible_attributes():
    ''' prints the possible attributes of a atlas region as stored in the ontology '''
    print({"id": 997,
     "atlas_id": -1,
     "ontology_id": 1,
     "acronym": "root",
     "name": "root",
     "color_hex_triplet": "FFFFFF",
     "graph_order": 0,
     "st_level": 0,
     "hemisphere_id": 3,
     "parent_structure_id": None,
     "children": []
    })

def print_regions_by_st_level(ont_ids):
    # can also filter by a specific id level e.g. ont_ids[8]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # example output
    # 1--> nRegions: 1 
    #     [8] 
    #     ['Basic cell groups and regions']
    # 2--> nRegions: 3 
    #     [567, 343, 512] 
    #     ['Cerebrum', 'Brain stem', 'Cerebellum']
    # 3--> nRegions: 4 
    #     [688, 623, 1129, 1065] 
    #     ['Cerebral cortex', 'Cerebral nuclei', 'Interbrain', 'Hindbrain']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    st_order_dict = gather_ids_by_st_level(ont_ids, as_dict=True)
    for k,v in sorted(st_order_dict.items(), key=lambda x: int(x[0])):
        print(f'{k}--> nRegions: {len(v)}', '\n\t', v, '\n\t',  get_attributes_for_list_of_ids(ont_ids, v, 'name'))

def ont2df():
    ont_ids = load_ontology()
    names_dict = dict(zip([d['name'] for d in ont_ids.values()], ont_ids.keys()))
    ont_df = []
    for k,v in names_dict.items():
        ont_df.append({'region_id': v, 'region_name':k, 'acronym':ont_ids[v]['acronym']})
    # ont_df = pd.DataFrame(ont_df)
    # ont_df.to_excel('ontology_dataframe.xlsx')
    return ont_df


def load_lowest_structural_level_only(json_path):
    raw_ontology = load_raw_ontology(json_path)
    result = find_empty_children(raw_ontology)
    return result

def load_raw_ontology(json_path=r"F:\ABBA\abba_atlases\1.json"):
    with open(json_path, 'r') as f:
        img_labels_json = json.load(f)
    return img_labels_json['msg'][0]['children']


def load_ontology(json_path=r"F:\ABBA\abba_atlases\1.json"):
    ''' 
        load a list of dicts where keys are atlas_ids and values are dict of all properties and children
            read the 1.json file in abba_atlases                OR
            read the v3p1-ontology created during abba export 
                (though not all functions may work as org is different, e.g. attrs in data attribute)
    '''
    # read the json file
    with open(json_path, 'r') as f:
        img_labels_json = json.load(f)

    # handle the different ontologies, which have varying organizations of the attributes
    if json_path.endswith('1.json'):
        all_groups = img_labels_json['msg'][0]['children']
    elif json_path.endswith('Adult Mouse Brain - Allen Brain Atlas V3p1-Ontology.json'):
        all_groups = img_labels_json['root']['children']

    # convert nested ontology to dict of id:attributes
    ids = iterate_dicts(all_groups)
    return ids

def iterate_dicts(dicts, result=None, counter=0):
    ''' convert nested ontology to dictionary where keys are atlas ids and values are attributes of that region '''
    if result==None:
        result = {}

    for d in dicts:     
        children = d.get('children')
        # result[d['id']] = d['name']
        result[d['id']] = d

        if children:
            counter+=1
            iterate_dicts(children, result=result, counter=counter)
    return result

def get_attributes_from_ids(ontology_by_ids, list_of_ids, get_keys):
    assert isinstance(get_keys, list)
    return [get_attributes_for_list_of_ids(ontology_by_ids, list_of_ids, get_key, warn=False) for get_key in get_keys]

def get_attributes_for_list_of_ids(ontology_by_ids, list_of_ids, get_key, warn=False):
    ''' though some atlas regions do not appear in ontology they seem to be insignificant '''
    extracted_attributes = []
    atlas_region_lookup_error_ids = []
    for x in list_of_ids:
        try:
            val = extract_ont_label_key(ontology_by_ids, x, get_key)
        except KeyError:
            atlas_region_lookup_error_ids.append(x)
            val = 'notFound'
        extracted_attributes.append(val)
    lookup_error_ids_unique = set(atlas_region_lookup_error_ids)
    if atlas_region_lookup_error_ids and warn:
        print(f'WARN n unique atlas ids not found: {len(lookup_error_ids_unique)} ({lookup_error_ids_unique}) total: ({len(atlas_region_lookup_error_ids)})')
    return extracted_attributes

def extract_ont_label_key(ontology_by_ids, px_label, get_key='acronym'):
    ''' look up atlas id in ontology, returns acronym but can also return any other key such as name'''
    values = ontology_by_ids[px_label]
    if 'data' in values and get_key != 'children': # e.g. for v3p1 ontology
        values = values['data']
    return values[get_key]






def find_empty_children(d):
    empty_children = []
    
    if isinstance(d, dict):
        if "children" in d and not d["children"]:
            empty_children.append(d)
            
        for value in d.values():
            empty_children.extend(find_empty_children(value))
    elif isinstance(d, list):
        for item in d:
            empty_children.extend(find_empty_children(item))
    
    return empty_children




def extract_ids_breadth_first(data):
    ids = []
    queue = deque([data])
    
    while queue:
        current = queue.popleft()
        
        if isinstance(current, dict):
            if "children" in current:
                for child in current["children"]:
                    ids.append(child["id"])
                    queue.append(child)
        elif isinstance(current, list):
            for item in current:
                queue.append(item)
    
    return ids




def get_children(ont_ids, id):
    return [d['id'] for d in ont_ids[id]['children']]


def gather_ids_by_st_level(d, as_dict=False):
    # returns list of tuples (stl, id) or dict where keys are stl
    # contains duplicates !!!
    # need to filter these out I belive, like so:
    # set_st_order = []
    # for el in st_order_ids:
    #     if el not in set_st_order:
    #         set_st_order.append(el)
    
    ids_and_levels = []
    
    if isinstance(d, dict):
        if "id" in d and "st_level" in d:
            ids_and_levels.append((d["st_level"], d["id"]))
            
        for value in d.values():
            ids_and_levels.extend(gather_ids_by_st_level(value))
    elif isinstance(d, list):
        for item in d:
            ids_and_levels.extend(gather_ids_by_st_level(item))
    if as_dict:
        return list_of_tups_to_dict(ids_and_levels)
    else:
        return ids_and_levels

def list_of_tups_to_dict(lot):
    ch_st_levels = {}
    for tup in lot:
        stl, id = tup
        if stl not in ch_st_levels:
            ch_st_levels[stl] = []
        ch_st_levels[stl].append(id)
    return ch_st_levels


def parent_ontology_at_st_level(ont_ids, st_level_parents, start_index=8):
    # slice the ontology so the parent level is defined by a specific st_level
    # st_level_parents should an st_level (as int)
    st_order_dict = gather_ids_by_st_level(ont_ids[start_index], as_dict=True) # restrict to basic regions 

    st_parents = []
    for reg_id in st_order_dict[st_level_parents]:
        st_parents.append(ont_ids[reg_id])
    
    # import json
    # with open('st_8_parent_ont_ids.json', 'w') as f:
    #     json.dump(st_8_parents, f)

    return st_parents

def get_all_parents(ont, reg_id, parent_ids=None, max_region_id=997):
    if parent_ids is None:
        parent_ids = []
    if (reg_id==max_region_id) or (reg_id is None):
        return parent_ids
    parent_id = ont.ont_ids[reg_id].get('parent_structure_id', None)
    parent_ids.append(parent_id)
    return get_all_parents(ont, parent_id, parent_ids=parent_ids)

def get_all_children(nested_list):
    # returns a list of all children ids, where nested list is children key of ont for a specific region_id
    ids = []

    def helper(nested_item):
        if isinstance(nested_item, list):
            for item in nested_item:
                helper(item)
        elif isinstance(nested_item, dict):
            if 'id' in nested_item:
                ids.append(nested_item['id'])
            if 'children' in nested_item:
                helper(nested_item['children'])
    assert isinstance(nested_list, list), 'please pass a list'
    helper(nested_list)
    return ids

def map_children2parent(ont_ids, ont_slice):
    child_to_parent_mapping = {}
    for d in ont_slice:
        child_ids = get_all_children(d['children'])
        child_names = get_attributes_for_list_of_ids(ont_ids, child_ids, 'name')
        for chn in child_names:
            child_to_parent_mapping[chn] = d['name']
    return child_to_parent_mapping



def filter_regions_by_st_level(ont_ids, ont_slice, max_st_lvl=None):
    # get regions at a specific st_level
    max_st_lvl = 99 if max_st_lvl is None else max_st_lvl # optionally cap st_lvl at a certain value
    # st_order_ids = [el[1] for el in gather_ids_by_st_level(ont_slice, as_dict=False)] # replaced so cap could be implemented
    set_st_order = [] # need to convert to a set because of redundancies in ont_ids during iteration (each region appears by itself and under parent)
    for st_lvl, reg_id in gather_ids_by_st_level(ont_slice, as_dict=False):
        if (reg_id not in set_st_order) and (st_lvl <=max_st_lvl):
            set_st_order.append(reg_id)
    st_order_names = get_attributes_for_list_of_ids(ont_ids, set_st_order, 'name')
    return st_order_names

def get_st_parents(ont_ids, child_regions, lvl):
    # get parent regions at a specific st_level
    ont_slice_colors = parent_ontology_at_st_level(ont_ids, lvl)
    st_order_dict_colors = gather_ids_by_st_level(ont_slice_colors, as_dict=True)
    child2parent_mapping_colors = {k:v for k,v in map_children2parent(ont_ids, ont_slice_colors).items() if k in list(child_regions)}
    return child2parent_mapping_colors