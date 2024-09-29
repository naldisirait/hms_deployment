#import modules
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import numpy as np
import time
import pickle
import torch
from datetime import datetime
import pandas as pd

#import modul from this project
from src.data_processing import transform_input_demo_into_ready_to_use, get_input_ml1, get_latest_n_entries, open_json_file, convert_df_to_dict_hms
from src.data_ingesting import get_prec_from_big_lake
from src.utils import inference_model, to_tensor, get_current_datetime, open_json_file
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable
from RunHECHMSPalu import run_hms_palu

#import model ml1 and ml2
from models.discharge.model_ml1 import load_model_ml1
from models.inundation.model_ml2  import load_model_ml2

def data_demo_to_input_hms(filename_demo,path_config_stas_to_grid,path_config_grid_to_subdas, path_conf_grided_to_df):
    ingested_data_name = "Stasiun"
    ingested_data = transform_input_demo_into_ready_to_use(filename_demo)
    all_grided_data, dates, _ =  get_input_ml1(ingested_data,
                                               ingested_data_name,
                                               path_config_stas_to_grid,
                                               path_config_grid_to_subdas)
    
    conf_grided_to_df = open_json_file(path_conf_grided_to_df)
    indexes = conf_grided_to_df['indexes']
    grided_prec = np.array(all_grided_data)
    t, len_lat, len_lon = grided_prec.shape
    prec = np.reshape(grided_prec, (t,-1))
    dict_prec = {'time' : dates}
    for idx in indexes:
        dict_prec[f'INDEX{idx}'] = prec[:,idx]
    df = pd.DataFrame(dict_prec)
    input_hms, ch_wilayah, dates = convert_df_to_dict_hms(df)
    return input_hms, ch_wilayah, dates

def data_demo_to_input_ml1(hours,filename_demo,path_config_stas_to_grid,path_config_grid_to_subdas):
    ingested_data_name = "Stasiun"
    ingested_data = transform_input_demo_into_ready_to_use(filename_demo)
    ingested_data = get_latest_n_entries(ingested_data,hours)
    all_grided_data, dates, input_ml1 =  get_input_ml1(ingested_data,
                                                       ingested_data_name,
                                                       path_config_stas_to_grid,
                                                       path_config_grid_to_subdas)
    return input_ml1

def get_non_flood_depth():
    path_config_depth = "./configs/conf of non flood.pkl"
    with open(path_config_depth,"rb") as file:
        loaded_data = pickle.load(file)
    depth = loaded_data['depth']
    return depth

def get_input_ml2_hms(filename_demo,input_size_ml2, path_config_stas_to_grid, path_config_grid_to_subdas, path_conf_grided_to_df):
    input_hms, ch_wilayah, dates = data_demo_to_input_hms(filename_demo,
                                                          path_config_stas_to_grid,
                                                          path_config_grid_to_subdas, 
                                                          path_conf_grided_to_df)
    debit_3days, all_debit_from_hms = run_hms_palu(input_hms)
    input_ml2_hms = debit_3days.reshape((1,input_size_ml2,1))
    input_ml2_hms = torch.tensor(input_ml2_hms, dtype=torch.float32)
    return input_ml2_hms, debit_3days, all_debit_from_hms, ch_wilayah, dates

# Define input data model
class InputData(BaseModel):
    inputs: List[List[float]]

app = FastAPI()
def do_prediction():
    input_debit = "hms"
    tstart = time.time()
    start_run_pred = get_current_datetime()

    #1. Define all constants and load models
    hours_hms =  720
    hours = 144
    jumlah_subdas = 114
    input_size_ml1 = hours * jumlah_subdas #jumlah jam dikali jumlah subdas
    output_size_ml1 = 168 #jumlah debit yang diestimasi, 24 jam terakhir adalah hasil forecast
    model_ml1 = load_model_ml1(input_size=input_size_ml1, output_size=output_size_ml1)

    input_size_ml2 = 72
    output_size_ml2 = 3078 * 2019 #rows x columns 
    model_ml2 = load_model_ml2(input_size=input_size_ml2, output_size=output_size_ml2)

    #2. Ingest Data input
    filename_demo = "./data/demo/hujan_kasus2.xlsx"
    path_config_stas_to_grid = "./configs/configuration of stasiun to grid.json"
    path_config_grid_to_subdas = "./configs/configuration of grid to subdas.json"
    path_config_grid_to_df = "./configs/configuration of grided to df.json"

    #3.1 Inference ML1
    input_ml1 = data_demo_to_input_ml1(hours=hours,
                                       filename_demo=filename_demo,
                                       path_config_stas_to_grid=path_config_stas_to_grid,
                                       path_config_grid_to_subdas=path_config_grid_to_subdas)
    output_ml1 = inference_model(model_ml1,input_ml1)

    #4.1 Inference ML2 using output from ML1
    output_ml1 = output_ml1[:,-input_size_ml2:]
    input_ml2_from_ml1 = np.expand_dims(output_ml1, axis=-1)

    #print(f"input_ml2 type: {type(input_ml2)}, shape: {input_ml2.shape}")
    input_ml2_from_ml1 = torch.tensor(input_ml2_from_ml1, dtype=torch.float32)
    output_ml2_from_ml1 = inference_model(model_ml2, input_ml2_from_ml1)
    output_ml2_from_ml1 = output_ml2_from_ml1[0,:].reshape(3078,2019)
    
    #print(f"output_ml2 after slicing and reshape type: {type(output_ml2)}, shape: {output_ml2.shape}")
    input_ml2_from_hms, debit_3days, all_debit_from_hms, ch_wilayah, dates = get_input_ml2_hms(filename_demo=filename_demo,
                                                                                               input_size_ml2=input_size_ml2, 
                                                                                               path_config_stas_to_grid=path_config_stas_to_grid,
                                                                                               path_config_grid_to_subdas=path_config_grid_to_subdas,
                                                                                               path_conf_grided_to_df=path_config_grid_to_df)
    output_ml2_from_hms = inference_model(model_ml2, input_ml2_from_hms)
    output_ml2_from_hms = output_ml2_from_hms[0,:].reshape(3078,2019)

    if np.max(debit_3days) < 200:
        output_ml2_from_hms = get_non_flood_depth()

    if np.max(output_ml1)< 200:
        output_ml2_from_ml1 = get_non_flood_depth()

    #5. Bundle the Output
    #Convert output ml1 to dict
    dates, dict_output_ml1 = output_ml1_to_dict(dates=dates, output_ml1=output_ml1.tolist(), precipitation=ch_wilayah)
    dates, dict_output_hms = output_ml1_to_dict(dates=dates, output_ml1=debit_3days.tolist(), precipitation=ch_wilayah)

    #Convert output ml2 to dict
    dict_output_ml2_from_hms = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=output_ml2_from_hms)
    dict_output_ml2_from_ml1 = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=output_ml2_from_ml1)

    end_run_pred = get_current_datetime()
    tend = time.time()
    prediction_runtime = tend-tstart
    if input_debit == "hms":
        dict_out1 = dict_output_hms
        dict_out2 = dict_output_ml2_from_hms
    else:
        dict_out1 = dict_output_ml1
        dict_out2 = dict_output_ml2_from_ml1

    output = {"Prediction Time Start": str(start_run_pred), 
              "Prediction time Finished": str(end_run_pred), 
              "Prediction Output ml1": dict_out1,
              "Prediction Output ml2": dict_out2,
              "test output": "ddungs"}
    
    output = ensure_jsonable(output)
    print(f"Prediction time {prediction_runtime}s")

    return output

@app.post("/predict")
async def predict():
    output = do_prediction()
    return output

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8002)

#Local test
# uvicorn app:app --reload
