#import relevent libraries 
import os
import json
import numpy as np 
from inverseDesign import of_Q
from inverseDesign import placeParams
from inverseDesign import costWrapper
from genConst import L3const
from defineCrystal import L3Crystal

#convert one dictionary
def convert_keys_to_string(original_dict):
    converted_dict = {}
    for key, value in original_dict.items():
        string_key = str(key)
        converted_dict[string_key] = value
    return converted_dict

#converts a list of dictionary keys to strings for saving 
def convert_keys_in_dict_list(dict_list):
    # Initialize a list to hold the converted dictionaries
    converted_list = []

    # Iterate through each dictionary in the list
    for d in dict_list:
        # Convert the keys of each dictionary
        converted_dict = convert_keys_to_string(d)
        converted_list.append(converted_dict)

    return converted_list

#for converting results from scypy optomization to json savable
def convert_optimize_result_to_jsonable(result):
    # Initialize a dictionary to hold the converted results
    jsonable_result = {}

    # Iterate through each attribute in the result object
    for key in result.keys():
        value = result[key]

        # Convert NumPy arrays to lists for JSON serialization
        if isinstance(value, np.ndarray):
            jsonable_result[key] = value.tolist()
        # Ensure other types are JSON serializable
        elif isinstance(value, (int, float, str, bool, type(None))):
            jsonable_result[key] = value
        # Convert other complex types to strings or appropriate formats
        else:
            jsonable_result[key] = str(value)

    return jsonable_result


#create callback class so that callback can have access to addtional key words and 
#itteration number
class callbackFunction:
    def __init__(self):
        self.iteration = 0
        self.objective_function = None
        self.filepath = None
        self.dx = None
        self.dy = None
        self.dr = None
        self.kwargs = None
        self.constraints = None

    def addData(self,filepath,objective_function=None,dx={},dy={},dr={},constraints=None,**kwargs):
        self.objective_function = objective_function
        self.filepath = filepath
        self.dx = dx
        self.dy = dy
        self.dr = dr
        self.kwargs = kwargs
        self.constraints = constraints

    #callback function for saving the data 
    def callback(self,params,*args,**kwargs):
        
        #get the optimal value
        value,freq = costWrapper(params,objective_function=self.objective_function,returnFreq=True,dx=self.dx,dy=self.dy,dr=self.dr, **self.kwargs)

        #output something so we know where we are at
        print(self.iteration,': ',value)

        #put the parameters into the correct place
        dx,dy,dr = placeParams(params,save=True,dx=self.dx,dy=self.dy,dr=self.dr,**self.kwargs)

        #convert dict tupples to strings
        dx,dy,dr = convert_keys_in_dict_list([dx,dy,dr])

        #format the data
        step = {'value':value, 'freq':freq, 'dx':dx, 'dy':dy, 'dr':dr}

        # Read existing data
        with open(self.filepath, 'r') as file:
            data = json.load(file)

        #either add the itteration data or create the key word and add
        data['iterations'][self.iteration] = step

        #write the file new data back to the json
        with open(self.filepath, 'w') as file:
            json.dump(data, file, indent=4)
        
        #add to the itteration count 
        self.iteration += 1



#add the different defaults that will be used for the various logging function
#pick the optoins for the GME computation
options = {'verbose': False}

#define defualts for experiments
defaults = {'dx': {}, 'dy': {}, 'dr': {},'Nx': 16, 'Ny': 10, 'dslab': 0.6, 'n_slab': 12,'ra': 0.29,
            'gmax': 2, 'options': options, 'method': 'l-bfgs-b', 'objective_function': of_Q, 'nk':1,
            'bounds': None, 'constraints': None , 'gradients': 'exact', 'compute_im': False, 'callback': None,
            'constraints':False,'constFunc':None,'minFreq':0,'maxfreq':1000,'minrad':0,'mindist':0,"optMode":0,
            'crystal':L3Crystal,'sidelength':0}

# Function to add default values from the 'defaults' dictionary to 'metadata'
def add_missing_defaults(metadata):
    for default_key, default_value in defaults.items():
        if default_key not in metadata:
            metadata[default_key] = default_value
    return metadata

#add function for running a set of inverse designes for the given parameters and saves the 
#metadata and results to a json file
def experiment(data,process):
    if not isinstance(data, dict) or 'name' not in data:
        raise ValueError("The data must be a dictionary containing the 'name' key.")

    name = data['name']
    directory_path = os.path.join('results', name)

    if not os.path.exists('results'):
        os.makedirs('results')

    os.makedirs(directory_path, exist_ok=True)

    #run all of the trials
    for key, metadata in data.items():
        if key == 'name':
            continue
        #show wich thing we are running 
        print('---------- Running ', key, ' ----------')

        #make the path for the file we are adding data into
        file_path = os.path.join(directory_path, key + '.json')

        #set up the file to store the data 
        # Check if the file exists
        if not os.path.exists(file_path):
            # Initialize with default values if the file doesn't exist
            jsondata = {'metadata': {}, 'results': {},'iterations': {}}
        else:
            # Read the JSON file if it exists
            with open(file_path, 'r') as file:
                jsondata = json.load(file)

            # Reset specified keys to empty dictionaries
            jsondata['metadata'] = {}
            jsondata['results'] = {}
            jsondata['iterations'] = {}

        # Write the changes or new data back to the file
        with open(file_path, 'w') as file:
            json.dump(jsondata, file, indent=4)

        if not isinstance(metadata, dict):
            raise ValueError(f"The value for the key '{key}' must be a dictionary.")

        # Add missing defaults to the metadata before processing
        metadata = add_missing_defaults(metadata)

        #if constraints=True then generate constraints, else make it None
        if metadata['constraints'] == True:
            metadata['constraints'] = metadata['constFunc'](**metadata)
        else:
            metadata['constraints'] = None

        #add to the parameters to the callback function class and then replace callback instance with funciton
        metadata['callback'] = callbackFunction()
        metadata['callback'].addData(file_path,**metadata)
        metadata['callback'] = metadata['callback'].callback

        #change the values of options to allgn with computation
        metadata['options']['numeig'] = metadata['optMode']+1
        metadata['options']['gradients'] = metadata['gradients']
        metadata['options']['compute_im'] = metadata['compute_im']

        #if constraints=True then generate constraints, else make it None
        if metadata['constraints']:
            metadata['constraints'] = metadata['constFunc'](**metadata)
        else:
            metadata['constraints'] = None
        # Process the data and add a 'results' entry to it
        results = process(**metadata)

        #convert dict tupples to strings
        metadata['dx'],metadata['dy'],metadata['dr'] = convert_keys_in_dict_list([metadata['dx'],metadata['dy'],metadata['dr']])

        #convert the funciton parameter to just a name of a function
        metadata['objective_function'] = metadata['objective_function'].__name__
        metadata['callback'] = metadata['callback'].__name__
        metadata['crystal'] = metadata['crystal'].__name__

        #convert constraints to simple true false
        if metadata['constraints'] != None:
            metadata['constraints'] = True
        if metadata['constFunc'] != None:
            metadata['constFunc'] = metadata['constFunc'].__name__
        

        #convert results to something json can save 
        results =convert_optimize_result_to_jsonable(results)

        # Read existing data
        with open(file_path, 'r') as file:
            jsondata = json.load(file)

        #either add the itteration data or create the key word and add
        jsondata['metadata'] = metadata
        jsondata['results'] = results

        #write the file new data back to the json
        with open(file_path, 'w') as file:
            json.dump(jsondata, file, indent=4)

#add function that retreves data from json file 
def read_json_file(file_path):
    """
    Reads a JSON file and returns the data.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    dict: The data from the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        return None