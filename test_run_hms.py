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
