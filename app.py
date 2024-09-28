from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import numpy as np
import time
import pickle
import torch
from datetime import datetime

#import modul from this project
from src.data_processing import get_input_ml1, get_input_hms, convert_prec_grided_to_ch_wilayah, open_json_file
from src.data_ingesting import get_prec_from_big_lake
from src.utils import inference_model, to_tensor, get_current_datetime, open_json_file
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable

def get_input_debit_sample(name):
    try:
        with open('Kasus Validasi ML2.pkl', 'rb') as file:
            data = pickle.load(file)
        
        debit = data[name]  # Ensure 'debit' is extracted correctly
        len_flat = len(debit)

        # Make sure 'debit' is a NumPy array
        if not isinstance(debit, np.ndarray):
            debit = np.array(debit)  # Convert to a NumPy array if it's not already
        debit = debit.tolist()
        # Convert to a PyTorch tensor
        debit = torch.tensor(debit, dtype=torch.float32)
        #debit = torch.from_numpy(debit)

        # Reshape to match the required shape
        debit = debit.reshape(1, len_flat)
        debit = debit.numpy()

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the error after logging it
    return debit

def get_non_flood_depth():
    path_config_depth = "./configs/conf of non flood.pkl"
    with open(path_config_depth,"rb") as file:
        loaded_data = pickle.load(file)
    depth = loaded_data['depth']
    return depth

# Define input data model
class InputData(BaseModel):
    inputs: List[List[float]]

#import model ml1 and ml2
from models.discharge.model_ml1 import load_model_ml1
from models.inundation.model_ml2  import load_model_ml2

app = FastAPI()
def do_prediction():
    tstart = time.time()
    start_run_pred = get_current_datetime()

    #1. Define all constants and load models
    hours = 144
    jumlah_subdas = 114
    input_size_ml1 = hours * jumlah_subdas #jumlah jam dikali jumlah subdas
    output_size_ml1 = 168 #jumlah debit yang diestimasi, 24 jam terakhir adalah hasil forecast
    model_ml1 = load_model_ml1(input_size=input_size_ml1, output_size=output_size_ml1)

    input_size_ml2 = 72
    output_size_ml2 = 3078 * 2019 #rows x columns 
    model_ml2 = load_model_ml2(input_size=input_size_ml2, output_size=output_size_ml2)

    #2. Ingest Data input
    path_config_stas_to_grid = "./configs/configuration of stasiun to grid.json"
    path_config_grid_to_subdas = "./configs/configuration of grid to subdas.json"
    path_config_grid_to_df = "./configs/configuration of grided to df.json"

    conf_grid_to_df = open_json_file(path_config_grid_to_df)
    index_grided_chosen = conf_grid_to_df['indexes']

    ingested_data_name, ingested_data, runtime_ingest_data = get_prec_from_big_lake(hours)
    #ingested_data_name_hms, ingested_data_hms = get_prec_from_big_lake(hours_hms)
    print(f"Ingested data runtime {runtime_ingest_data}")
    print(f"ingested_data type {type(ingested_data)}")

    #3.1 Inference ML1
    all_grided_data, dates, input_ml1 =  get_input_ml1(ingested_data,
                                                   ingested_data_name,
                                                   path_config_stas_to_grid,
                                                   path_config_grid_to_subdas)
    
    print(f"Type of all_grided_data {type(all_grided_data)}")
    print(f"Type of dates {type(all_grided_data)}")
    print(f"Type of input_ml1 {type(input_ml1)}, shape input_ml1: {input_ml1.shape}")
    
    output_ml1 = inference_model(model_ml1,input_ml1)
    print(f"Type of output_ml1 {type(output_ml1)}, shape output_ml1: {output_ml1.shape}")

    #4.1 Inference ML2 using output from ML1
    output_ml1 = output_ml1[:,-input_size_ml2:]
    input_ml2 = np.expand_dims(output_ml1, axis=-1)

    print(f"input_ml2 type: {type(input_ml2)}, shape: {input_ml2.shape}")
    input_ml2 = torch.tensor(input_ml2, dtype=torch.float32)
    output_ml2 = inference_model(model_ml2, input_ml2)
    print(f"output_ml2 raw type: {type(output_ml2)}, shape: {output_ml2.shape}")
    output_ml2 = output_ml2[0,:].reshape(3078,2019)
    print(f"output_ml2 after slicing and reshape type: {type(output_ml2)}, shape: {output_ml2.shape}")

    if np.max(output_ml1) < 200:
        output_ml2 = get_non_flood_depth()
    #5. Bundle the Output
    #Convert output ml1 to dict
    ch_wilayah = convert_prec_grided_to_ch_wilayah(prec_grided=all_grided_data, idx_chosen=index_grided_chosen)
    dates, dict_output_ml1 = output_ml1_to_dict(dates=dates, output_ml1=output_ml1[0,:].tolist(), precipitation=ch_wilayah)

    #Convert output ml2 to dict
    dict_output_ml2 = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=output_ml2)
    
    end_run_pred = get_current_datetime()
    tend = time.time()
    prediction_runtime = tend-tstart

    output = {"Prediction Time Start": str(start_run_pred), 
              "Prediction time Finished": str(end_run_pred), 
              "Prediction Output ml1": dict_output_ml1,
              "Prediction Output ml2": dict_output_ml2}
    
    output = ensure_jsonable(output)
    print(f"Prediction time {prediction_runtime}s")

    return output

@app.post("/predict")
async def predict():
    output = do_prediction()
    return output

#Local test
# uvicorn app:app --reload
