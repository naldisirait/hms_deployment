from hms.runHECHMS import run_hms_palu
from src.utils import inference_model, to_tensor, get_current_datetime, open_json_file
import numpy as np

path_config_grid_to_df = "./configs/configuration of grided to df.json"

conf_grid_to_df = open_json_file(path_config_grid_to_df)
index_grided_chosen = conf_grid_to_df['indexes']

dumm_prec = {}
for index in index_grided_chosen:
    dumm_prec[f"INDEX{index}"] = np.random.rand(720)

if __name__ == "__main__":
    value = run_hms_palu(dumm_prec)
    print(f"Return debit the last 3 days is {value}")
