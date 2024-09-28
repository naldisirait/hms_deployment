#Convert Indexed Rainfall to HEC-DSS
import pandas as pd
import numpy as np
import openpyxl
import os
import datetime
import pandas as pd
import numpy as np
import openpyxl
from hecdss import HecDss, ArrayContainer
#Run HEC-HMS
import os
import subprocess
from hecdss import *
#from datetime import datetime


# Read the Excel file
# excel_file = './data/demo/DataX3.xlsx'
excel_file = "./data/demo/Contoh data sebelum hecdss.xlsx"
df = pd.read_excel(excel_file)
df['time'] = pd.to_datetime(df['time'])
dateTime = df['time'] 

# Read the Excel file
#excel_file = 'Contoh data sebelum hecdss.xlsx'
#df = pd.read_excel(excel_file)

# Prepare .dss files
dssIn = HecDss("sample7.dss") #Use this file as a reference.
tsc = dssIn.get("//SACRAMENTO/PRECIP-INC//1Day/OBS/")  # read example due to unexpected error when do independent writing
tsc.units = "MM"

# Prepare datetime
start_date = datetime.datetime(2024, 1, 1, 0, 0)
end_date = datetime.datetime(2024, 1, 30, 23, 0)
# Calculate the number of hours between the two dates
num_hours = int((end_date - start_date).total_seconds() // 3600)
# Create a list of hourly datetime objects
dates = [start_date + datetime.timedelta(hours=i) for i in range(num_hours + 1)]

# Loop through each column in index_columns
index_columns = [col for col in df.columns if col.startswith('INDEX')]

dssOutFile = "./HECHMS_Update1/CH/GridGSMAP_corrected.dss"

# Check if the file exists
if os.path.exists(dssOutFile):
    os.remove(dssOutFile)

for column in index_columns:
    dssOut = HecDss(dssOutFile)
    pathString = f"/HUJAN/{column}/PRECIP-INC//1HOUR/SIMULATED/"

    # Extract values as a NumPy array
    values_array = df[column].to_numpy() + 3

    # fill DSS
    tsc.values = values_array
    tsc.times = dates
    tsc.id = pathString
    dssOut.put(tsc)
    dssOut.close()

# Manage dss 
dssIn.close()

def create_control_script(script_path, project_name, project_path, run_name):
    """Create the HEC-HMS control script."""
    with open(script_path, 'w') as script_file:
        script_file.write("from hms.model.JythonHms import *\n")
        script_file.write(f'OpenProject("{project_name}", "{project_path}")\n')
        script_file.write(f'Compute("{run_name}")\n')
        script_file.write("Exit(1)\n")

def run_hec_hms(script_path):
    """Run HEC-HMS using the generated script."""
    #hec_hms_cmd = "hec-hms"
    hec_hms_cmd = "/home/mhews/hms_deployment/HEC-HMS-4.12/hec-hms.sh"
    command = [hec_hms_cmd, "-s", script_path]
    try:
        # Run the HEC-HMS command
        process = subprocess.run(command, check=True, capture_output=False, text=False)
        print("HEC-HMS ran successfully.")
        print("Output:")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running HEC-HMS:")
        print(e.stderr)

if __name__ == "__main__":
    project_name = "HMSPalu"
    project_path = "./HECHMS_Update1/HMSPalu"  # Update to your project path
    run_name = "JAN2024"
    
    # Create the control script
    control_script_path = "./HECHMS_Update1/HMSPalu/compute.script"  # Path to save the control script
    create_control_script(control_script_path, project_name, project_path, run_name)
    
    # Run HEC-HMS with the control script
    run_hec_hms(control_script_path)
	
#import matplotlib.pyplot as plt

# Open a DSS file
file_loc = "./HECHMS_Update1/HMSPalu"
file_name = "JAN2024.dss"

file_path = file_loc + '/' + file_name
print(file_path)
dss = HecDss(file_path)
# Retrieve and print data
data_path = "//Outlet-Banjir/FLOW//1Hour/RUN:JAN2024/"
#t1 = datetime(2024, 1, 1)    
data = dss.get(data_path)
#print(data.values)
Val = data.values
#plt.plot(Val)
print(Val)