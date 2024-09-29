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
from src.data_processing import get_input_ml1, get_input_hms, convert_prec_grided_to_ch_wilayah, open_json_file, convert_df_to_dict_hms
from src.data_ingesting import get_prec_from_big_lake
from src.utils import inference_model, to_tensor, get_current_datetime, open_json_file
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable
from RunHECHMSPalu import run_hms_palu

#import model ml1 and ml2
from models.discharge.model_ml1 import load_model_ml1
from models.inundation.model_ml2  import load_model_ml2
from app import do_prediction

if __name__ == "__main__":
    output = do_prediction()