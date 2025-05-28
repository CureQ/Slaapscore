'''
Adjustable variables
'''

# Folder with all ecg files - Each folder is of 1 participant
ecg_folders = [
    "C:/Users/esmee/OneDrive/Documenten/Hva jaar 4/Afstudeerstage/data/Esmee/MoveSense_data/MoveSense_participant_36/",
    # "C:/Users/jesse/Documents/HvA/1001 Hete Nachten/MoveSense_data/MoveSense_participant_*/",
    # "C:/Users/jesse/Documents/HvA/1001 Hete Nachten/MoveSense_data/MoveSense_participant_*/",
    
]

# Ensure you are are in the right directionary by writing this code line in the Python environment terminal 
'''
 > cd .\ecg-sleep-staging\your_own_data\primary_model
'''

''' 
EXAMPLE VALUES

# Optional demograhpics 
gender = [
    "Male", 
    "Female", 
    "Male"
]
age = [
    40, 
    20, 
    60
]
'''


###
''' 
Analyse all ECG files 
'''

import pandas as pd
import os
import subprocess
from Prepare_ECG_files_android import get_hdf5_file_android # android achter de functies gezet (ook in de bestanden prepare_ECG_files en ECG_results_android)
from ECG_results_android import get_results_table_android

# Loop through all folders with ecg files
for ecg_folder in ecg_folders:
    # Get all ECG files
    ecg_files = []
    # Generate directory tree of ECG folder
    for root, dirs, files in os.walk(ecg_folder):
        print(files)
        # Loop through all files in ECG folder
        for file in files:
            # Search only for all ecg .CSV files
            if "ecg" in file.lower() and file.endswith(".csv"):
                ecg_files.append(file)

    file_counter = 1
    preprocessed_files, result_files = [], []
    # Preprocess all ECG files
    for ecg_file in ecg_files:
        ecg_file_name = ecg_folder + ecg_file

        # Analyse ecg file name
        preprocessed_file, preprocessed_df = get_hdf5_file_android(ecg_file_name, file_counter)
        preprocessed_files.append(preprocessed_file)
        print(preprocessed_files)


        ''' Run Machine Learning model '''
        # Ensure you are are in the right directionary by writing this code line in the Python environment terminal 
        '''
        > cd .\ecg-sleep-staging\your_own_data\primary_model
        '''

        # Path to train script
        train_script = os.getcwd() + "/train.py"

        # Run model in Python environment terminal with subprocess
        subprocess.run(["python", train_script, preprocessed_file], check=True)


        ''' Compute results '''
        # Get result files
        results_file_name = os.getcwd() + "/results.h5"
        result_files.append(results_file_name)

        # Analyse HDF5 files
        get_results_table_android(ecg_file_name, results_file_name, preprocessed_file, file_counter, preprocessed_df)

        print(result_files)
        file_counter += 1