import pickle
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

#import modul from this project
from src.data_processing import transform_input_demo_into_ready_to_use, get_input_ml1, get_latest_n_entries, open_json_file, convert_df_to_dict_hms
from src.data_ingesting import get_prec_from_big_lake
from src.utils import inference_model, to_tensor, get_current_datetime, open_json_file
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable
from RunHECHMSPalu import run_hms_palu

# Function to create sliding windows
def create_sliding_windows_input_hms(df,window_size,step_size,total_sim,columns_to_select,path_config_stas_to_grid,path_config_grid_to_subdas,path_config_grid_to_df):
    """
    Function to create an input data for HMS Simulation
    Args:
        df: dataframe all precipitation
        window_size: total hour of precipitation input
        step_size: step for sliding windows
        columns_to_select: selected columns for station filtering
    Returns:
        windows(list[dict]): every data ready to run hec-hms
        every_all_grided_data(list(list(list))): grided data from every window sliced
        start dates(list(str)): start date every simulation data
        end dates(list(str)): end date every simulation data
    """
    ingested_data_name = "Stasiun"
    windows = [] #dictionary of every windows data
    every_precip_input_for_simulation_hms = []
    df_stasiuns = []
    start_dates = []
    end_dates = []
    n = 0
    for i in range(0, len(df) - window_size + 1, step_size):
        n+=1
        df_window = df.iloc[i:i + window_size]
        df_stasiuns.append(df_window)
        dates_window = df_window['time'].values
        ingested_data = transform_input_demo_into_ready_to_use(df_window)
        all_grided_data, dates, _ =  get_input_ml1(ingested_data=ingested_data,
                                                   ingested_data_name=ingested_data_name,
                                                   path_config_stas_to_grid=path_config_stas_to_grid,
                                                   path_config_grid_to_subdas=path_config_grid_to_subdas)
        conf_grided_to_df = open_json_file(path_config_grid_to_df)
        indexes = conf_grided_to_df['indexes']
        grided_prec = np.array(all_grided_data)
        t, len_lat, len_lon = grided_prec.shape
        prec = np.reshape(grided_prec, (t,-1))
        dict_prec = {'time' : dates}
        for idx in indexes:
            dict_prec[f'INDEX{idx}'] = prec[:,idx]
        df_prec_chosen_index = pd.DataFrame(dict_prec)
        input_hms, ch_wilayah, dates = convert_df_to_dict_hms(df_prec_chosen_index)
        windows.append(input_hms)
        every_precip_input_for_simulation_hms.append(df_prec_chosen_index)

        # Convert numpy.datetime64 to pandas datetime and format 
        start_date = pd.to_datetime(dates_window[0]).strftime('%Y%m%d %H%M')
        end_date = pd.to_datetime(dates_window[-1]).strftime('%Y%m%d %H%M')
        start_dates.append(start_date)
        end_dates.append(end_date)
        if n== total_sim:
            break
    return windows, df_stasiuns, every_precip_input_for_simulation_hms, start_dates, end_dates 

def run_hms_with_mp(config_and_data):
    start_date, end_date, input_hms, precip_index, precip_stasiuns, project_paths = config_and_data
    _, all_val_debit = run_hms_palu(precip_dict=input_hms, path_hecms_model=project_paths)
    output_runs  = {"start date precip": start_date,
                    "end date precip": end_date,
                    "data precip dataframe index": precip_index,
                    "data precip stasiuns":precip_stasiuns,
                    "data debit": all_val_debit}
    filename = f"data/output_hms/start hujan {start_date} end hujan {end_date}.pkl"
    # Open the file in write-binary mode and dump the object
    with open(filename, 'wb') as file:
        pickle.dump(output_runs, file)

if __name__ == "__main__":
    #initial time start
    tstart = time.time()
    #Paths    
    path_config_stas_to_grid = "./configs/configuration of stasiun to grid.json"
    path_config_grid_to_subdas = "./configs/configuration of grid to subdas.json"
    path_config_grid_to_df = "./configs/configuration of grided to df.json"

    #set constants
    total_simulated = 0 #total simulated before
    total_sim = 20 #total simulation to run
    window_size = 720 #total hours of precipitation input
    step_size = 1 #step size for slicing
    n_cpus = 6 #number of cpus used for simulation, you need to match the total project paths HECHMSUpdate1 copied
    project_paths = ["HECHMS_Update1","HECHMS_Update2","HECHMS_Update3","HECHMS_Update4","HECHMS_Update5", "HECHMS_Update6"]
    
    #read data
    df = pd.read_excel("./data/input_hms/CH 2016 -2023.xlsx")
    columns_to_select = ['CH BANGGA BAWAH', 'CH TONGOA', 'CH INTAKE LEWARA', 'SAMBO', 'CH TUVA']
    # Create the sliding windows, windows adalah seluruh data input hujan dalam bentuk dict 
    windows, df_stasiuns, every_precip_input_for_simulation_hms, start_dates, end_dates = create_sliding_windows_input_hms(
        df=df,                                           
        window_size=window_size,
        total_sim=total_sim,
        step_size=step_size,
        columns_to_select=columns_to_select,
        path_config_stas_to_grid=path_config_stas_to_grid,
        path_config_grid_to_subdas=path_config_grid_to_subdas,
        path_config_grid_to_df=path_config_grid_to_df
        )
    
    # chose the data
    windows = windows[total_simulated:total_simulated+total_sim]
    df_stasiuns = df_stasiuns[total_simulated:total_simulated+total_sim]
    every_precip_input_for_simulation_hms = every_precip_input_for_simulation_hms[total_simulated:total_simulated+total_sim]
    
    #buat data ke dalam list agar bisa di kirim kedalam fungsi untuk di run paralel
    new_start_dates = start_dates[total_simulated:total_simulated+total_sim]
    new_end_dates = start_dates[total_simulated:total_simulated+total_sim]
    new_hms_project_paths = []
    n = 0
    for i in range(total_simulated,total_sim+total_simulated):
        if n == 6:
            n = 0
        new_hms_project_paths.append(project_paths[n])
        n+=1

    #bundle into tuples and run hms with multiprocessing
    config_and_data = [(new_start_dates[i], new_end_dates[i], windows[i],every_precip_input_for_simulation_hms[i],df_stasiuns[i],new_hms_project_paths[i]) for i in range(total_sim)]
    with mp.Pool(processes=n_cpus) as pool:
        outputs = pool.map(run_hms_with_mp,config_and_data)
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