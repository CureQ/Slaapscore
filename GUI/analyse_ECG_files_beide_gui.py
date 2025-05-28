# Bestandsnaam: analyse_ECG_files_beide_gui.py
# Geschreven door: Esmee Springer
# Voor het laatst bewerkt op: 22-05-2025

'''
Adjustable variables
'''

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


''' 
Analyse all ECG files 
'''

# Functie om de ECG-analyse uit te voeren op de opgegeven map (root_dir)
def run_ecg_analysis(root_dir):
    
    # Zet de opgegeven map in een lijst
    ecg_folders = [root_dir]

    # Importeer benodigde pakkages
    import pandas as pd
    import os
    import subprocess

    # Importeer functies voor het voorbereiden en analyseren van ECG-bestanden
    from Prepare_ECG_files_android_gui import get_hdf5_file_android 
    from ECG_results_android_gui import get_results_table_android
    from Prepare_ECG_files_gui import get_hdf5_file
    from ECG_results_gui import get_results_table

    # Functie om het preprocessing en machine learning model uit te voeren
    def process_ecg_file(ecg_file_name, file_counter, results_file_name, is_android=False):
        
        # Preprocessing
        if is_android:
            preprocessed_file, preprocessed_df = get_hdf5_file_android(ecg_file_name, file_counter)
        else:
            preprocessed_file = get_hdf5_file(ecg_file_name, file_counter)

        # Start het machine learning model via train.py
        train_script = os.path.join(os.getcwd(), "train.py")
        subprocess.run(["python", train_script, preprocessed_file], check=True)

        # Analyseer het modelresultaat en schrijf de output naar een resultatenbestand
        if is_android:
            get_results_table_android(ecg_file_name, results_file_name, preprocessed_file, file_counter, preprocessed_df)
        else:
            get_results_table(ecg_file_name, results_file_name, preprocessed_file, file_counter)

    # Loop door alle ECG-mappen
    for ecg_folder in ecg_folders:
        # Zoek alle ECG-bestanden in de map en submappen
        ecg_files = []
        for root, dirs, files in os.walk(ecg_folder):
            
            # Print de huidge map die wordt doorzocht
            print(f"Processing folder: {root}")
            
            # Voeg bestanden toe aan de lijst als ze eindigen op '.csv' en 'ecg' in de bestandsnaam bevatten
            for file in files:
                if file.endswith(".csv") and "ecg" in file.lower():
                    ecg_files.append(file)

        # Start het verwerken van de bestanden
        # Start teller om bij te houden hoeveel bestanden zijn verwerkt
        file_counter = 1
        preprocessed_files, result_files = [], []

        # Pad naar results.h5 (zorg ervoor dat dit bestand altijd overschreven wordt)
        results_file_name = os.path.join(os.getcwd(), "results.h5")

        # Verwijder oude resultaten voordat we starten
        if os.path.exists(results_file_name):
            os.remove(results_file_name)
            print(f"Verwijderd oud results.h5 bestand")

        # Verwerk de ECG-bestanden
        for ecg_file in ecg_files:
            ecg_file_name = os.path.join(ecg_folder, ecg_file)

            # Bepaal of het bestand Android of Apple is op basis van de naam
            is_android = "ECG" in ecg_file  # Android heeft "ECG" in de bestandsnaam

            # Verwerk het bestand (Apple of Android)
            process_ecg_file(ecg_file_name, file_counter, results_file_name, is_android)

            # Voeg het bestand toe aan de lijsten van verwerkte bestanden
            file_counter += 1

        print(f"Verwerkte {file_counter - 1} bestanden in map: {ecg_folder}")

# Wanneer dit script direct wordt uitgevoerd, start dan de ECG-analyse
if __name__ == "__main__":
    test_path = "C:/Users/esmee/OneDrive/Documenten/Hva jaar 4/Afstudeerstage/data/Esmee/MoveSense_data/MoveSense_participant_35/"
    run_ecg_analysis(test_path)