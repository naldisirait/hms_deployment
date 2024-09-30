import pickle
import time
from RunHECHMSPalu import run_hms_palu
from src.utils import open_json_file
import numpy as np
import pandas as pd


# Function to create sliding windows
def create_sliding_windows(df, window_size, step_size):
    windows = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        windows.append(window)
    return windows



# filename = r"./data/input_hms/windowing data input hms.pkl"
# with open(filename, 'rb') as file:
#     loaded_data = pickle.load(file)

if __name__ == "__main__":
    df = pd.read_excel("./data/input_hms/CH 2016 -2023.xlsx")
    # List of columns you want to select
    columns_to_select = ['CH BANGGA BAWAH', 'CH TONGOA', 'CH INTAKE LEWARA', 'SAMBO', 'CH TUVA']
    # Select only those columns
    df_selected = df[columns_to_select]

    # Set the window size (720 rows) and step size (1 row or 1 hour)
    window_size = 720
    step_size = 1

    # Create the sliding windows
    windows = create_sliding_windows(df_selected, window_size, step_size)

    windows = windows[0:3000]

    new_data = []
    for i in range(3000):
        asd = windows[i]
        dicts = {}
        for col in columns_to_select:
            dicts[col] = asd[col].values
        new_data.append(dicts)

    output_all_runs = []
    for data in new_data:
        _, all_val = run_hms_palu(data)
        output_all_runs.append(all_val)

    filename = "output 3000 simulasi hms.pkl"
    # Open the file in write-binary mode and dump the object
    with open(filename, 'wb') as file:
        pickle.dump(output_all_runs, file)