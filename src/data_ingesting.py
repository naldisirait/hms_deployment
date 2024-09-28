import re
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import pandas as pd
import time

# os.environ['HADOOP_HOME'] = '/etc/hadoop'
# os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'

def get_prec_from_big_lake(hours):
    tstart = time.time()
    ingested_data = get_prec_stasiun_from_big_lake(hours)
    if ingested_data != None:
        ingested_name = "Stasiun"
    else:
        ingested_data = get_prec_gsmap_from_big_lake(hours)
        ingested_name = "Satelit"
    tend = time.time()
    runtime = tend - tstart
    return ingested_name, ingested_data, runtime

def get_hdfs_path(date):
    date_str = date.strftime('%Y%m%d.%H%M')  # Format the date as 'YYYYMMDD.HHMM'
    hdfs_path = f"hdfs://master-01.bnpb.go.id:8020/user/warehouse/JAXA/curah_hujan/{date.strftime('%Y/%m/%d')}/gsmap_now_rain.{date_str}.nc"
    return hdfs_path

def slice_data_to_palu(xr_data):
    #potong data hanya pada bagian DAS Palu saja
    left, right, top, bottom = 119.1499, 120.75, -0.5499, -1.85
    xr_palu = xr_data.sel(Latitude=slice(bottom, top), Longitude=slice(left, right))
    return xr_palu

def get_prec_only_palu(file_path):
    ds = xr.open_dataset(file_path, decode_times=False)
    ds_palu = slice_data_to_palu(ds)
    # Flip the latitude dimension (reverse the order)
    ds_palu = ds_palu.isel(Latitude=slice(None, None, -1))
    prec_values = ds_palu['hourlyPrecipRateGC'][0].values
    if prec_values.shape != (14,17):
        prec_values = np.zeros((14,17))
    return prec_values

def get_grided_prec_palu(hdfs_path):
    print(f"Trying to get the precipitation Palu from GSMAP Jaxa {hdfs_path}")
    from pyspark.sql import SparkSession
    filename = hdfs_path[-31:]
    local_path = f'data/gsmap/{filename}'
    
    spark = SparkSession.builder \
        .appName("master") \
        .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
        .config("spark.hadoop.hadoop.security.authorization", "true") \
        .config("spark.security.credentials.hive.enabled","false") \
        .config("spark.security.credentials.hbase.enabled","false") \
        .enableHiveSupport().getOrCreate()
    
    # Mengakses FileSystem melalui JVM
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

    # Membuat objek Path di HDFS dan lokal
    hdfs_file_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
    local_file_path = spark._jvm.org.apache.hadoop.fs.Path(local_path)

    # Gunakan FileUtil untuk menyalin dari HDFS ke sistem lokal
    spark._jvm.org.apache.hadoop.fs.FileUtil.copy(fs, hdfs_file_path, spark._jvm.org.apache.hadoop.fs.FileSystem.getLocal(hadoop_conf), local_file_path, False, hadoop_conf)
    prec_val_palu = get_prec_only_palu(local_path)
    os.remove(local_path)
    return prec_val_palu
    
def open_pickle_gsmap(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def dump_pickle_gsmap(data,file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data,file)

def generated_half_hourly_dates_backwards(total_generated):
    # Step 1: Get the current date and time
    current_date = datetime.now()
    # Step 2: Adjust the current time to the nearest 00 or 30 backward
    minute = current_date.minute
    if minute >= 30:
        # If the current minute is 30 or above, round down to 30
        adjusted_date = current_date.replace(minute=30, second=0, microsecond=0)
    else:
        # If the current minute is below 30, round down to 00
        adjusted_date = current_date.replace(minute=0, second=0, microsecond=0)
    # Step 3: Generate total_generated half-hourly dates backward, from the adjusted time
    generated_dates = [adjusted_date - timedelta(minutes=30 * i) for i in range(total_generated)]
    # Step 4: Reverse the list to have the latest date at the last index (oldest to newest)
    generated_dates.reverse()
    return generated_dates

def convert_half_hourly_gsmap_prec_to_hourly(month_gsmap_data):
    #Step 1: Create a new dictionary for hourly data
    hourly_data_dict = {}

    #Convert string keys to datetime objects for easier manipulation
    data_dict_datetime_keys = {datetime.strptime(k, '%Y-%m-%d %H:%M:%S'): v for k, v in month_gsmap_data.items()}
    
    #Sort the dictionary by datetime keys (in case it's unordered)
    sorted_dates = sorted(data_dict_datetime_keys.keys())
    
    # Step 3: Iterate through the sorted dates and combine every two consecutive half-hour intervals
    for i in range(0, len(sorted_dates), 2):
        if i + 1 < len(sorted_dates):
            # Get the two consecutive 2D arrays
            date1 = sorted_dates[i]
            date2 = sorted_dates[i + 1]
    
            array1 = data_dict_datetime_keys[date1]
            array2 = data_dict_datetime_keys[date2]
    
            # Step 4: Average the two arrays (element-wise) to create hourly data
            hourly_array = (array1 + array2) / 2
    
            # Step 5: Create a new key representing the hourly timestamp (use the second date of the pair)
            new_key = date2.strftime('%Y-%m-%d %H:%M:%S')
    
            # Step 6: Add the hourly array to the new dictionary
            hourly_data_dict[new_key] = hourly_array
    return hourly_data_dict

def get_list_gsmap_now_from_biglake():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("master") \
        .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
        .config("spark.hadoop.hadoop.security.authorization", "true") \
        .config("spark.security.credentials.hive.enabled","false") \
        .config("spark.security.credentials.hbase.enabled","false") \
        .enableHiveSupport().getOrCreate()
        
    # Define the HDFS path
    hdfs_path = "/user/warehouse/JAXA/curah_hujan"

    # List HDFS files recursively
    hdfs_files = list_hdfs_files_recursive(spark, hdfs_path)
    hdfs_rain = [i for i in hdfs_files if "gsmap_now_rain" in i]
    return hdfs_rain

def get_prec_gsmap_from_big_lake(hours):
    """
    function to get the gsmap precipitation from biglake 
    Args:
        hours(int): number of hours to lookback
    Returns:
        hourly_gsmap_month_data(dict): key(date): grided precip value
    """
    gsmap_pickle_path = "./data/gsmap/gsmap_latest_month.pkl"
    missed_data_path =  "./data/gsmap/missed_data.pkl"
    total_data = hours * 2
    generated_dates = generated_half_hourly_dates_backwards(total_generated=total_data)
    latest_month_gsmap_data = open_pickle_gsmap(file_path=gsmap_pickle_path)
    new_month_gsmap_data = {}

    #get all gsmap now filenames from biglake
    all_gsmap_now_data_in_biglake = set(get_list_gsmap_now_from_biglake())
    #check all the file needed
    gsmap_now_needed = []
    for date in generated_dates:
        date = date - timedelta(hours=8)
        date_str = date.strftime('%Y-%m-%d %H:%M:%S')
        hdfs_path = get_hdfs_path(date=date)
        gsmap_now_needed.append(hdfs_path)

    missed_data = [file for file in gsmap_now_needed if file not in all_gsmap_now_data_in_biglake]
    dict_missed_data_gsmap = {"missed data gsmap": missed_data}
    total_missed = len(missed_data)
    if len(missed_data) > int(total_data/2):
        print(f"There is {len(total_missed)} missing data from jaxa, then return None")
        return None
    
    for date in generated_dates:
        date = date - timedelta(hours=8)
        date_str = date.strftime('%Y-%m-%d %H:%M:%S')
        if date_str in latest_month_gsmap_data:
            new_month_gsmap_data[date_str] = latest_month_gsmap_data[date_str]
        else:
            hdfs_path = get_hdfs_path(date=date)
            if hdfs_path in missed_data:
                print(f"Missing data replaced with zeros, {hdfs_path}")
                new_month_gsmap_data[date_str] = np.zeros((14,17))
            else:
                new_month_gsmap_data[date_str] = get_grided_prec_palu(hdfs_path=hdfs_path)
    hourly_gsmap_month_data = convert_half_hourly_gsmap_prec_to_hourly(new_month_gsmap_data)
    if hours == 720:
        dump_pickle_gsmap(data=new_month_gsmap_data,file_path=gsmap_pickle_path)
        dump_pickle_gsmap(data=dict_missed_data_gsmap, file_path=missed_data_path)
    return hourly_gsmap_month_data

# Function to extract the timestamp using regex
def extract_timestamp(file_path):
    # Regular expression to extract the timestamp from the file path
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})')
    match = timestamp_pattern.search(file_path)
    if match:
        date_str = match.group(1)  # YYYY-MM-DD
        time_str = match.group(2)  # HH-MM-SS
        timestamp_str = f'{date_str} {time_str}'
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H-%M-%S')  # Convert to datetime object
        # Replace minute, second, and microsecond with 0
        new_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        return new_timestamp
    else:
        return None
        
def check_last_hours_data(file_paths, hours):
    """
    Function to check if data is available for the last `N hours`. If all data is available, it returns the list of file paths for 
    those hours. If any data is missing, it reports the expected hours and marks the missing ones as 'miss'.
    
    Args:
    file_paths (list): List of file paths
    hours (int): Integer defining the number of hours to check 
    
    Returns: 
        tuple (str, list): List of file paths and expected hours, marking missing hours as 'miss' if not available.
    """
    # Get current time and the time 'hours' ago
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = current_time - timedelta(hours=hours)

    # Collect all timestamps and map them to their corresponding file paths
    available_files = [(extract_timestamp(path), path) for path in file_paths if extract_timestamp(path) is not None]

    # Prepare result list with expected hours
    result_list = []

    # Check for each expected hour
    for hour in range(hours + 1):  # Include the current hour
        expected_time = start_time + timedelta(hours=hour)
        # Check if this expected time exists in the available files
        file_for_hour = next((path for (timestamp, path) in available_files if timestamp == expected_time), None)

        if file_for_hour:
            result_list.append((expected_time.strftime('%Y-%m-%d %H:%M:%S'), file_for_hour))
        else:
            result_list.append((expected_time.strftime('%Y-%m-%d %H:%M:%S'), "miss"))
    # Return results
    return result_list[-hours:]

def check_availability_stasiun(checked_date):
    half = int(len(checked_date)/2)
    n = 0
    date_miss = []
    for date, info in checked_date:
        if info == "miss":
            n+=1
            date_miss.append(date)
    if n >= half:
        print(f"There are {n} missing data, switched to sateliite")
        output = "Not Available"
    else:
        output = "Available"
    return output
    
def list_hdfs_files_recursive(spark, path):
    hadoop = spark._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration()
    conf.set("fs.defaultFS", "hdfs://master-01.bnpb.go.id:8020")
    files = []
    
    def recursive_list_files(path):
        try:
            for f in fs.get(conf).listStatus(path):
                files.append(str(f.getPath()))
                if fs.get(conf).isDirectory(f.getPath()):
                    recursive_list_files(f.getPath())
        except Exception as e:
            print("Error:", e)
    
    recursive_list_files(hadoop.fs.Path(path))
    
    return files

def dummy_zeros_rainfall_for_miss_date():
    dum = {'name': {0: 'CH TONGOA',
            1: 'SAMBO',
            2: 'CH INTAKE LEWARA',
            3: 'CH BANGGA BAWAH',
            4: 'CH TUVA'},
            'rainfall': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}
    df = pd.DataFrame(dum)
    return df

def get_prec_stasiun_from_big_lake(hours):
    """
    Function to get precipitation data from big lake, it will return the last hours if the data is available, 
    Args:
        hours (int): Number of hours to look back
    Returns:
        output(dict/None): dictionary if the data available. None if does not available

    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("master") \
        .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
        .config("spark.hadoop.hadoop.security.authorization", "true") \
        .config("spark.security.credentials.hive.enabled","false") \
        .config("spark.security.credentials.hbase.enabled","false") \
        .enableHiveSupport().getOrCreate()
    
    # Define the HDFS path
    hdfs_path = "/user/warehouse/SPLP/PUPR/curah_hujan/palu"
    # hdfs_path = "/user/warehouse/SPLP/PUPR"
    # hdfs_path = "/user/warehouse/JAXA/curah_hujan"

    # List HDFS files recursively
    hdfs_files = list_hdfs_files_recursive(spark, hdfs_path)
    json_files = [i for i in hdfs_files if (".json" in i and "curah_hujan" in i) and ("unstructed" not in i)]
   
    # Check if data is available for the last N hours
    checked_date = check_last_hours_data(json_files, hours)
    availability = check_availability_stasiun(checked_date)

    if availability == "Available":
        prec_per_time = {}
        for n,(date,path) in enumerate(checked_date):
            if path == "miss":
                prec_per_time[date] = dummy_zeros_rainfall_for_miss_date()
            else:
                json_data = spark.read.option("multiline","true").json(path)
                df = json_data.toPandas()
                prec_per_time[date] = df
        output = prec_per_time
    else:
        output = None
    return output