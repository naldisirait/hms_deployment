import numpy as np
import torch
import pandas as pd
import json

def convert_to_standard_date(date_input):
    """
    Convert the input date to the format '%Y-%m-%d %H:%M:%S'.
    Handles numpy.datetime64 and various string formats.
    """
    # Check if input is of type numpy.datetime64
    if isinstance(date_input, np.datetime64):
        # Convert numpy.datetime64 to datetime object
        date_input = date_input.astype('M8[s]').astype(datetime)
        return date_input.strftime('%Y-%m-%d %H:%M:%S')
    
    # List of possible date formats to check for string inputs
    date_formats = [
        '%Y-%m-%d %H:%M:%S',  # Full date and time
        '%Y-%m-%d %H:%M',     # Date and time without seconds
        '%Y-%m-%d',           # Date without time
        '%Y/%m/%d',           # Date with slashes
        '%d-%m-%Y',           # Day-Month-Year format
        '%d/%m/%Y',           # Day/Month/Year with slashes
        '%Y%m%d',             # Compact date format
        '%d-%b-%Y',           # Date with month as abbreviation (e.g., 28-Sep-2024)
        '%B %d, %Y',          # Full month name (e.g., September 28, 2024)
        '%b %d, %Y'           # Abbreviated month name (e.g., Sep 28, 2024)
    ]
    
    # Try to parse the input if it is a string or another non-numpy datetime input
    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(str(date_input), date_format)
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')  # Convert to standard format
        except ValueError:
            continue

    # If none of the formats match, raise an error or handle the invalid input
    raise ValueError(f"Unrecognized date format: {date_input}. Please provide a valid date.")


def convert_df_to_dict_hms(data_path):
    df = pd.read_excel(data_path)
    dates = df['time'].values
    new_date = [convert_to_standard_date(i) for i in dates]
    columns = df.columns
    out = {}
    for col in df.columns:
        if "INDEX" in col:
            out[col] = df[col].values

    df_numeric = df.drop(columns=['time'])
    array_numeric = df_numeric.to_numpy()
    ch_wilayah = np.mean(array_numeric, axis=1)
    return out, ch_wilayah[-144:], new_date

def get_input_hms(ingested_data,ingested_data_name, path_conf_grided_to_df,path_config_stas_to_grid,path_config_grid_to_subdas):
    """
    Function to get the input of hec hms, grided to dataframe
    Args:
        grided_prec(list): a list of grided precipitation each time
        dates (list): dates of data
        path_conf_grided_to_df(str): path to json file of the configuration grid to df
    Returns:
        df: dataframe of precipitation for hecdss
    """
    station_to_grided_config,grided_to_subdas_config = get_transformation_config(path_config_stas_to_grid, path_config_grid_to_subdas)

    all_grided_data_hms, dates, all_time_prec_subdas = process_precip_from_satelit(station_to_grided_config=station_to_grided_config,
                                                                        grided_to_subdas_config=grided_to_subdas_config,
                                                                        ingested_data=ingested_data,
                                                                        ingested_data_name=ingested_data_name)
    conf_grided_to_df = open_json_file(path_conf_grided_to_df)
    indexes = conf_grided_to_df['indexes']
    grided_prec = np.array(all_grided_data_hms)
    t, len_lat, len_lon = grided_prec.shape
    prec = np.reshape(grided_prec, (t,-1))
    dict_prec = {'time' : dates}
    for idx in indexes:
        dict_prec[f'INDEX{idx}'] = prec[:,idx]
    df = pd.DataFrame(dict_prec)
    return all_grided_data_hms,df 

def convert_prec_grided_to_ch_wilayah(prec_grided, idx_chosen):
    if isinstance(prec_grided, list):
        prec_grided = np.array(prec_grided)
    t = len(prec_grided)
    prec_grided = np.reshape(prec_grided, (t,-1))
    prec_grided = prec_grided[:,idx_chosen]
    ch_wilayah = np.mean(prec_grided, axis = 1)
    return ch_wilayah
    
def open_json_file(filepath):
    # Open the json file
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def get_transformation_config(path_config_stas_to_grid, path_config_grid_to_subdas):
    """
    function to get the configuration of percentage distributed station to grid and grid to subdas
    this function is static, since the value is constant
    """
    # Open the stasiun_to_grid.json file and load the data
    with open(path_config_stas_to_grid, 'r') as file:
        pct_stasiun_at_grid = json.load(file)
    
    # Open the stasiun_to_grid.json file and load the data
    with open(path_config_grid_to_subdas, 'r') as file:
        pct_gsmap_at_subbasin = json.load(file)
    
    return pct_stasiun_at_grid, pct_gsmap_at_subbasin

def get_prec_stas_from_data_api(data, stasiun):
    """
    function to retrieve rainfall value from a dataframe
    Args:
        data (dataframe): dataframe one time from big lake
        stasiun (string): stasiun name
    Retruns:
        prec_val(float): precipitation value 
    """
    prec_val = data[data['name']==stasiun]['rainfall'].values
    return prec_val

def transform_station_to_grided(config_pct, data):
    """
    Function to transform station precipitation into grided data
    Args:
        config_pct (json file): file contains percentage of each station contribute to each grid
    Returns:
        data: grided precipitation data
    """
    precip = np.zeros((14,17)).reshape(-1)
    for idx,val in config_pct.items():
        prec_total = 0 
        stasiuns = val['nama_stasiun']
        pcts = val['percentage']
        for stasiun,pct in zip(stasiuns,pcts):
            prec_value = get_prec_stas_from_data_api(data,stasiun)
            prec_value = prec_value * (pct/100)
            prec_total+=prec_value
        precip[int(idx)] = prec_total
    precip = np.reshape(precip, (14,17))
    return precip
    
def transform_grided_to_subdas(pct, prec_grid):
    """
    Function to conver grided precipitation data into precipitation at each subdas 
    
    Args:
        pct(json file): a dictionary contains an information of percentage every grid to each subdas
        prec_grid(np.array): an array of precipitation
    
    Returns:
        prec_subdas(dict): a dictionary {"subdas_name": [prec_value]}        
    """
    prec_grid = prec_grid.reshape(-1)
    prec_subbdas = {}
    for key,val in pct.items():
        idx = val['index']
        pct = np.array(val['percentage'])
        total_prec = prec_grid[idx] * (pct/100)
        prec_subbdas[key] = [np.sum(total_prec)]
    return prec_subbdas
    
def prec_subdas_to_tensor(all_prec_subdas):
    """
    function to convert dictionary subdas data into tensor for the input of ML1
    Args:
        all_prec_subdas(dict): precipitation data at each subdas
    Returns:
        tensor_prec (torch.tensor): flatten precipitation data
    """
    data = []
    for key,val in all_prec_subdas.items():
        data.extend(val)
    data = np.array(data)
    tensor_prec = torch.tensor(data, dtype = torch.float32)
    return tensor_prec

def get_precsubdas_per_time(station_to_grided_config,grided_to_subdas_config, data_input, data_input_name):
    # 1. Check the input data name
    if data_input_name == "Stasiun":
        grided_data = transform_station_to_grided(station_to_grided_config, data_input)
    elif data_input_name == "Satelit":
        grided_data = data_input
    else:
        return None
    #2. Convert prec grided into subDAS
    prec_subdas = transform_grided_to_subdas(grided_to_subdas_config, grided_data)
    
    return grided_data, prec_subdas 

def combine_dict(dict1,dict2):
    """
    function to combine dict, because the data transformation script is made for one time data.
    so we need to combine all time data
    Args:
        dict1(dict) : base dictionary
        dict2(dict): dict of subdas that we want to add to the dict1

    Returns:
        dict1 (dict): combined dictionary

    """
    for key,val in dict2.items():
        dict1[key].extend(val)
    return dict1

def process_precip_from_stasiun(station_to_grided_config,grided_to_subdas_config,ingested_data,ingested_data_name):
    all_time_prec_subdas = {}
    all_grided_data = []
    dates = []
    for n,(key,val) in enumerate(ingested_data.items()):
        dates.append(key)
        grided_data, prec_subdas = get_precsubdas_per_time(station_to_grided_config=station_to_grided_config,
                                                          grided_to_subdas_config=grided_to_subdas_config,
                                                          data_input = val,
                                                         data_input_name = ingested_data_name)
        all_grided_data.append(grided_data)
        if n == 0:
            all_time_prec_subdas = prec_subdas
        else:
            all_time_prec_subdas = combine_dict(all_time_prec_subdas,prec_subdas)

    return all_grided_data, dates, all_time_prec_subdas

def process_precip_from_satelit(station_to_grided_config,grided_to_subdas_config,ingested_data,ingested_data_name):
    all_time_prec_subdas = {}
    all_grided_data = []
    dates = []
    for n,(key,val) in enumerate(ingested_data.items()):
        dates.append(key)
        grided_data, prec_subdas = get_precsubdas_per_time(station_to_grided_config=station_to_grided_config,
                                                          grided_to_subdas_config=grided_to_subdas_config,
                                                          data_input = val,
                                                         data_input_name = ingested_data_name)
        all_grided_data.append(grided_data)
        if n == 0:
            all_time_prec_subdas = prec_subdas
        else:
            all_time_prec_subdas = combine_dict(all_time_prec_subdas,prec_subdas)

    return all_grided_data, dates, all_time_prec_subdas


def get_input_ml1(ingested_data,ingested_data_name,path_config_stas_to_grid,path_config_grid_to_subdas):
    """
    function to get the input of ml1
    Args:
        ingested_data(dict): N hours precipitation data from big lake
        ingested_data_name(str): either Stasiun or Satelit
        path_config_stas_to_grid(dict): configuration of distribution of stasiun to each grid
        path_config_stas_to_grid(dict): configuration of distribution each grid to each subdas
    Returns:
        tensor_input (tensor): flatten all the subdas precipitation data 
    """
    station_to_grided_config,grided_to_subdas_config = get_transformation_config(path_config_stas_to_grid, path_config_grid_to_subdas)
    if ingested_data_name == "Stasiun":
        all_grided_data, dates, all_time_prec_subdas = process_precip_from_stasiun(station_to_grided_config=station_to_grided_config,
                                                                               grided_to_subdas_config=grided_to_subdas_config,
                                                                               ingested_data=ingested_data,
                                                                               ingested_data_name=ingested_data_name)
    elif ingested_data_name == "Satelit":
        all_grided_data, dates, all_time_prec_subdas = process_precip_from_satelit(station_to_grided_config=station_to_grided_config,
                                                                               grided_to_subdas_config=grided_to_subdas_config,
                                                                               ingested_data=ingested_data,
                                                                               ingested_data_name=ingested_data_name)
    else:
        return None
    flatten_tensor_input = prec_subdas_to_tensor(all_time_prec_subdas)
    len_flat = len(flatten_tensor_input)
    flatten_tensor_input = flatten_tensor_input.reshape(1,len_flat)
    return all_grided_data, dates, flatten_tensor_input