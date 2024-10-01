import time
import multiprocessing as mp
print(f"CPUs avaible {mp.cpu_count()}")
tstart = time.time()
from RunHECHMSPalu import run_hms_palu
from src.utils import open_json_file
import numpy as np
path_config_grid_to_df = "./configs/configuration of grided to df.json"
conf_grid_to_df = open_json_file(path_config_grid_to_df)
index_grided_chosen = conf_grid_to_df['indexes']

#create samples data to run
samples_input_hms = []
for i in range(10):
    dumm_prec = {}
    for index in index_grided_chosen:
        dumm_prec[f"INDEX{index}"] = np.random.rand(720) + 2
    samples_input_hms.append(dumm_prec)

if __name__ == "__main__":
    with mp.Pool(processes=5) as pool:
        outputs = pool.map(run_hms_palu, samples_input_hms)
    #value_72, all_value = run_hms_palu(dumm_prec)
tend = time.time()
print(f"Runtiime {tend-tstart}")


import pickle
import time
from RunHECHMSPalu import run_hms_palu
from src.utils import open_json_file
import numpy as np
import pandas as pd
import multiprocessing as mp

# Function to create sliding windows
def create_sliding_windows(df, window_size, step_size,columns_to_select):
    # List of columns you want to select
    # Select only those columns
    dates = df['time'].values
    df_selected = df[columns_to_select]
    windows = []
    start_dates = []
    end_dates = []
    for i in range(0, len(df_selected) - window_size + 1, step_size):
        window = df_selected.iloc[i:i + window_size]
        windows.append(window)

        # Convert numpy.datetime64 to pandas datetime and format
        start_date = pd.to_datetime(dates[i]).strftime('%Y%m%d %H%M')
        end_date = pd.to_datetime(dates[i + window_size - 1]).strftime('%Y%m%d %H%M')
        
        start_dates.append(start_date)
        end_dates.append(end_date)

    return windows, start_dates, end_dates
