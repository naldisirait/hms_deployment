#Convert Indexed Rainfall to HEC-DSS
import pandas as pd
import numpy as np
import openpyxl
import os
import datetime
import openpyxl
from hecdss import HecDss, ArrayContainer
#Run HEC-HMS
import os
import subprocess
from hecdss import *
#from datetime import datetime

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
    hec_hms_cmd = "/home/mhews/old_deployment/hms/HEC-HMS-4.12/hec-hms.sh"
    command = [hec_hms_cmd, "-s", script_path]
    
    try:
        # Run the HEC-HMS command
        process = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, close_fds=True)
        #process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("HEC-HMS ran successfully.")
        print("Output:")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running HEC-HMS:")
        print(e.stderr)

def run_hms_palu(precip_dict):
    """
    function to run HMS
    Args:
        precip_dict(dict): key(str), val(np.array)
    Returns:
        debit(array):
    """
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

    dssOutFile = "./HECHMS_Update1/CH/GridGSMAP_corrected.dss"

    # Check if the file exists
    if os.path.exists(dssOutFile):
        os.remove(dssOutFile)

    for key,val in precip_dict.items():
        dssOut = HecDss(dssOutFile)
        pathString = f"/HUJAN/{key}/PRECIP-INC//1HOUR/SIMULATED/"

        # fill DSS
        tsc.values = val
        tsc.times = dates
        tsc.id = pathString
        dssOut.put(tsc)
        dssOut.close()

    # Manage dss 
    dssIn.close()

    project_name = "HMSPalu"
    project_path = "./HECHMS_Update1/HMSPalu"  # Update to your project path
    run_name = "JAN2024"
    
    # Create the control script
    control_script_path = "./HECHMS_Update1/HMSPalu/compute.script"  # Path to save the control script
    create_control_script(control_script_path, project_name, project_path, run_name)
    
    # Run HEC-HMS with the control script
    run_hec_hms(control_script_path)

    # Open a DSS file
    file_loc = "./HECHMS_Update1/HMSPalu"
    file_name = "JAN2024.dss"

    file_path = file_loc + '/' + file_name
    dss = HecDss(file_path)
    # Retrieve and print data
    data_path = "//Outlet-Banjir/FLOW//1Hour/RUN:JAN2024/"    
    data = dss.get(data_path)
    Val = data.values
    print(Val)
    return Val[672:744], Val
