# Bestandsnaam: analyse_ECG_files_android
# Naam: Esmee Springer
# Voor het laatst bewerkt op: 02-06-2025

'''
Adjustable variables
'''

# Definieer het pad naar een map met MoveSense bestanden
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
# Importeer de benodigde pakkages
import pandas as pd
import os
import subprocess

# Importeer functies voor het voorbereiden en analyseren van ECG-bestanden
from Prepare_ECG_files_android import get_hdf5_file_android 
from ECG_results_android import get_results_table_android

# Loop door alle ECG-mappen
for ecg_folder in ecg_folders:
    # Lijst om alle ECG-bestanden op te slaan die worden gevonden
    ecg_files = []
    # Doorzoek alle mappen en submappen in de huidige ECG-map
    for root, dirs, files in os.walk(ecg_folder):
        print(files)
        # Loop door alle bestanden in deze map
        for file in files:
            # Zoek alleen naar bestanden die "ecg" bevatten en eindigen op .csv
            if "ecg" in file.lower() and file.endswith(".csv"):
                ecg_files.append(file)

    # Tel hoeveel bestanden verwerkt zijn
    file_counter = 1
    # Lijsten om de voorbewerkte en resultaatbestanden op te slaan
    preprocessed_files, result_files = [], []

    # Verwerk elk gevonden ECG-bestand
    for ecg_file in ecg_files:
        # Voeg de map en bestandsnaam samen tot het volledige pad
        ecg_file_name = ecg_folder + ecg_file

        # Verwerk het ECG-bestand (voor Android data) en sla het resultaat op
        preprocessed_file, preprocessed_df = get_hdf5_file_android(ecg_file_name, file_counter)
        preprocessed_files.append(preprocessed_file)
        print(preprocessed_files)


        ''' Run Machine Learning model '''
        # Let op: zorg dat je in de juiste map werkt wanneer je dit uitvoert
        '''
        > cd .\ecg-sleep-staging\your_own_data\primary_model
        '''

        # Pad naar het script dat het model traint
        train_script = os.getcwd() + "/train.py"

        # Run het model via subprocess, met het voorbewerkte bestand als input
        subprocess.run(["python", train_script, preprocessed_file], check=True)


        ''' Compute results '''
        # Stel het pad in naar het gegenereerde resultatenbestand
        results_file_name = os.getcwd() + "/results.h5"
        result_files.append(results_file_name)

        # Analyseer de resultaten en maak de resultatentabel aan
        get_results_table_android(ecg_file_name, results_file_name, preprocessed_file, file_counter, preprocessed_df)

        # Toon de lijst met resultatenbestanden tot nu toe
        print(result_files)
        # Verhoog de teller zodat elk bestand apart wordt verwerkt
        file_counter += 1