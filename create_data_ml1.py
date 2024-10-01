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

def run_hms_with_mp(config_and_data):
    start_date, end_date, data_precip, project_paths = config_and_data
    _, all_val_debit = run_hms_palu(precip_dict=data_precip, path_hecms_model=project_paths)
    output_runs  = {"start date precip": start_date,
                    "end date precip": end_date,
                    "data prcip": data_precip,
                    "data debit": all_val_debit}
    filename = f"data/output_hms/start hujan {start_dates[n]} end hujan {end_dates[n]}.pkl"
    # Open the file in write-binary mode and dump the object
    with open(filename, 'wb') as file:
        pickle.dump(output_runs, file)

if __name__ == "__main__":
    tstart = time.time()
    n_cpus = 6
    project_paths = ["HECHMS_Update1","HECHMS_Update2","HECHMS_Update3","HECHMS_Update4","HECHMS_Update5", "HECHMS_Update6"]
    df = pd.read_excel("./data/input_hms/CH 2016 -2023.xlsx")
    # Set the window size (720 rows) and step size (1 row or 1 hour)
    window_size = 720
    step_size = 1
    columns_to_select = ['CH BANGGA BAWAH', 'CH TONGOA', 'CH INTAKE LEWARA', 'SAMBO', 'CH TUVA']
    # Create the sliding windows
    windows, start_dates, end_dates = create_sliding_windows(df, window_size, step_size,columns_to_select)
    
    total_sim = 3000
    windows = windows[0:total_sim]
    #buat data ke dalam list agar bisa di kirim kedalam fungsi untuk di run paralel
    new_data = []
    new_start_dates = []
    new_end_dates = []
    new_hms_project_paths = []
    n = 0
    for i in range(total_sim):
        asd = windows[i]
        dicts = {}
        for col in columns_to_select:
            dicts[col] = asd[col].values
        new_data.append(dicts)
        new_start_dates.append(start_dates[i])
        new_end_dates.append(end_dates[i])
        if n == 6:
            n = 0
        new_hms_project_paths.append(project_paths[n])
        n+=1
    #bundle into tuples and run hms with multiprocessing
    config_and_data = [(new_start_dates[i], new_end_dates[i], new_data[i], new_hms_project_paths[i]) for i in range(total_sim)]
    with mp.Pool(processes=n_cpus) as pool:
        outputs = pool.map(run_hms_palu,config_and_data)
    tend = time.time()
    print(f"Runtime {tend-tstart}s")

# # filename = r"./data/input_hms/windowing data input hms.pkl"
# # with open(filename, 'rb') as file:
# #     loaded_data = pickle.load(file)

# if __name__ == "__main__":
#     tstart = time.time()
#     df = pd.read_excel("./data/input_hms/CH 2016 -2023.xlsx")
#     # Set the window size (720 rows) and step size (1 row or 1 hour)
#     window_size = 720
#     step_size = 1
#     columns_to_select = ['CH BANGGA BAWAH', 'CH TONGOA', 'CH INTAKE LEWARA', 'SAMBO', 'CH TUVA']
#     # Create the sliding windows
#     windows, start_dates, end_dates = create_sliding_windows(df, window_size, step_size,columns_to_select)

#     windows = windows[0:20]

#     new_data = []
#     for i in range(20):
#         asd = windows[i]
#         dicts = {}
#         for col in columns_to_select:
#             dicts[col] = asd[col].values
#         new_data.append(dicts)

#     output_all_runs = []
#     for n,data in enumerate(new_data):
#         _, all_val = run_hms_palu(data)
#         output_runs  = {"start date precip": start_dates[n],
#                         "end date precip": end_dates[n],
#                         "data prcip": data,
#                         "data debit": all_val}
#         filename = f"data/output_hms/start hujan {start_dates[n]} end hujan {end_dates[n]}.pkl"
#         # Open the file in write-binary mode and dump the object
#         with open(filename, 'wb') as file:
#             pickle.dump(output_runs, file)
#     tend = time.time()
#     print(f"Runtime {tend-tstart}s")