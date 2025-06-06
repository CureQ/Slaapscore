# Bestandsnaam: analyse_ECG_files_validatie.py
# Naam: Esmee Springer
# Voor het laatst bewerkt op: 06-06-2025

# Importeren van pakkages
import pandas as pd
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import csv
import json
import re
from io import StringIO
import pyedflib
import glob
from scipy.signal import find_peaks
import random
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

# Importeer functies voor het voorbereiden en analyseren van ECG-bestanden
from Prepare_ECG_files_validatie import get_hdf5_file_validatie
from ECG_results_validatie import get_results_table_validatie
from ECG_results_validatie import get_results

# Functie om een folder volledig te verwijderen
def remove_folder(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)  # Verwijder de gehele map inclusief submappen en bestanden
            print(f"De map {folder_path} is verwijderd.")
        except Exception as e:
            print(f"Fout bij het verwijderen van {folder_path}: {e}")

# Verwijder de Results_validatie map (en alle submappen) als deze bestaat
results_validatie_dir = r'C:/shhs/Results_validatie'
remove_folder(results_validatie_dir)

# Folder met de edf bestanden met ECG data. 
shhs_dir = r'C:/shhs/polysomnography'

# Zoek naar alle EDF-bestanden in de supmappen "shhs1" en "shhs2"
all_shhs_paths = glob.glob(os.path.join(shhs_dir, 'edfs', 'shhs1', '*.edf')) + \
                 glob.glob(os.path.join(shhs_dir, 'edfs', 'shhs2', '*.edf'))

# Sorteer de gevonden paden alfabetisch
all_shhs_paths = sorted(all_shhs_paths)

# Selecteer willekeurig 1000 bestanden (zonder vervanging)
n_files = 1000
selected_paths = random.sample(all_shhs_paths, min(n_files, len(all_shhs_paths)))
print(f"{len(selected_paths)} bestanden geselecteerd voor verwerking")

# Pad naar de map waar je de geselecteerde bestanden wilt opslaan
output_dir = r'C:/shhs/selected_edf_files'
xml_output_dir = r'C:/shhs/selected_xml_files'

# Functie om een map leeg te maken (inhoud verwijderen, map blijft bestaan)
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        files = glob.glob(os.path.join(folder_path, '*')) # Zoek alle bestanden in de map
        for f in files:
            try:
                os.remove(f) # Verwijder elk bestand
            except Exception as e:
                print(f"Kon bestand niet verwijderen: {f}, fout: {e}")
    else:
        os.makedirs(folder_path) # Maak de map aan als deze nog niet bestaat

# Maak beide outputmappen leeg voordat nieuwe bestanden worden weggeschreven
clear_folder(output_dir)
clear_folder(xml_output_dir)

# Zorg ervoor dat de outputmap bestaat (maak aan indien nodig)
os.makedirs(output_dir, exist_ok=True)

# Verwerk elk geselecteerd EDF- bestand
for i, edf_file in enumerate(selected_paths):

    print(f"\n[{i+1}/{len(selected_paths)}] Inladen van: {edf_file}")
    
    try:
        # Open EDF-bestand
        f = pyedflib.EdfReader(edf_file)

        # Basisinformatie ophalen
        n_signals = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sampling_rates = f.getSampleFrequencies()

        print(f"Aantal signalen: {n_signals}")
        print(f"Signaalnamen: {signal_labels}")
        print(f"Samplefrequenties: {sampling_rates}")

        # Zoek naar ECG signaal op naam of index
        try:
            ecg_index = signal_labels.index('ECG')  
        except ValueError:
            print("ECG-signaal niet gevonden in dit bestand. Sla over.")
            f.close()
            continue

       # Controleer de samplefrequentie van het ECG-signaal hoog genoeg is
        ecg_fs = sampling_rates[ecg_index]
        min_required_fs = 101  # Minimale vereiste frequentie in Hz
        if ecg_fs < min_required_fs:
            print(f"Samplefrequentie van ECG is te laag ({ecg_fs} Hz), bestand wordt overgeslagen.")
            f.close()
            continue 

        # Lees ECG-data in
        ecg_data = f.readSignal(ecg_index)

        # Debug: Toon eerste 10 waarden van de ECG-data
        print("Eerste 10 ECG-datapunten:", ecg_data[:10])

        # Sluit het EDF-bestand
        f.close()

        # Kopieer het EDF-bestand naar de outputmap
        shutil.copy(edf_file, os.path.join(output_dir, os.path.basename(edf_file)))
        print(f"Bestand gekopieerd naar: {output_dir}/{os.path.basename(edf_file)}")

        # Zoek de bijbehorende XML-bestandsnaam op basis van bestandsnaam
        edf_filename = os.path.basename(edf_file)
        base_filename = os.path.splitext(edf_filename)[0]  # zonder .edf
        xml_filename = base_filename + '-profusion.xml'

        # Bepaal uit welke submap (shhs1 of shhs2) het bestand komt
        if 'shhs1' in edf_file:
            xml_full_path = os.path.join(shhs_dir, 'annotations-events-profusion', 'shhs1', xml_filename)
        elif 'shhs2' in edf_file:
            xml_full_path = os.path.join(shhs_dir, 'annotations-events-profusion', 'shhs2', xml_filename)
        else:
            print("Onbekende submap voor:", edf_file)
            continue

        # Kopieer het XML-bestand naar de XML-outputmap
        if os.path.exists(xml_full_path):
            shutil.copy(xml_full_path, os.path.join(xml_output_dir, xml_filename))
            print(f"Bijbehorend XML-bestand gekopieerd naar: {xml_output_dir}/{xml_filename}")
        else:
            print(f"Waarschuwing: XML-bestand niet gevonden voor {edf_filename}")

    except Exception as e:
        print(f"Fout bij verwerken van bestand {edf_file}: {e}")
        continue

# Folder met de EDF-bestanden
ecg_folders = [
    "C:/shhs/selected_edf_files/",
    # "C:/shhs/selected_edf_files/",
    # "C:/shhs/selected_edf_files/",
    
]

# Loop door alle folders met edf-bestanden en verzamel de paden naar de EDF-bestanden
for ecg_folder in ecg_folders:
    # Verkrijg alle EDF-bestanden
    edf_files = []
    for root, dirs, files in os.walk(ecg_folder):
        print(f"Bestanden in {root}: {files}")
        # Loop door alle bestanden in de map
        for file in files:
            # Zoek alleen naar .edf bestanden 
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))

xml_folders = [
    "C:/shhs/selected_xml_files/",
    # Je kunt hier eventueel andere mappen toevoegen waar XML-bestanden zich bevinden
]

# Haal alle XML-bestanden op en koppel deze aan de juiste ECG-bestanden
xml_files = {}  # Een dictionary om XML-bestanden te koppelen aan hun base-bestandsnaam
for xml_folder in xml_folders:
    for root, dirs, files in os.walk(xml_folder):
        print(f"Bestanden in {root}: {files}")
        for file in files:
            if file.endswith(".xml"):
                base_filename = file.replace("-profusion.xml", "")
                xml_files[base_filename] = os.path.join(root, file)
    # Debug: Toon gevonden XML-bestanden
    print(f"Gevonden XML-bestanden: {xml_files}")

# Folder met alle ECG-bestanden
ecg_folders = [
    "C:/shhs/selected_edf_files/",
    # Je kunt hier eventueel andere mappen toevoegen waar ECG-bestanden zich bevinden
]
# Loop door alle folders met edf-bestanden
for ecg_folder in ecg_folders:
    # Haal alle EDF-bestanden op
    edf_files = []
    for root, dirs, files in os.walk(ecg_folder):
        print(f"Bestanden in {root}: {files}")
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))

        
    # Preprocess/verwerk alle ECG bestanden
    preprocessed_files, result_files = [], []
    for ecg_file in edf_files:
        ecg_file_name = os.path.join(ecg_folder, ecg_file)

        # Analyseer EDF-bestanden
        preprocessed_file, preprocessed_df = get_hdf5_file_validatie(ecg_file_name)
        print("ecg_file_name:", ecg_file_name)
        preprocessed_files.append(preprocessed_file)
        print(preprocessed_files)

        ''' Run Machine Learning model '''
        # Ensure you are in the right directory by writing this code line in the Python environment terminal 
        '''
        > cd .\ecg-sleep-staging\your_own_data\primary_model
        '''
        
        # Pad naar het trainingsscript
        train_script = os.getcwd() + "/train.py"
        
        # Start het model via subprocess en geef het voorbewerkte bestand mee als argument
        subprocess.run(["python", train_script, preprocessed_file], check=True)

        ''' Compute results '''
        # Verkrijg het bestand met resultaten als h5 bestand
        results_file_name = os.getcwd() + "/results.h5"
        result_files.append(results_file_name)

        # Verkrijg de resultaten van de HDF5-bestanden, inclusief voorspellingen
        predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions = get_results(results_file_name)

        # Haal bestandsnaam en base naam op (zonder extensie) uit het ECG-bestand
        ecg_filename = os.path.basename(ecg_file)
        base_filename = os.path.splitext(ecg_filename)[0]  # bijvoorbeeld 'shhs1-200318'

        # Zoek naar een XML-bestand met exact dezelfde base besatndsnaam
        xml_full_path = xml_files.get(base_filename)

        if xml_full_path:
            print(f"XML-bestand gevonden voor {ecg_filename}: {xml_full_path}")
        else:
            print(f"Geen XML-bestand gevonden voor {ecg_filename}.")
            continue # Sal over als geen XML-bestand is gevonden

        # Roep de functie get_results_table_validatie aan 
        get_results_table_validatie(ecg_file_name, results_file_name, xml_full_path, predictions, f)

        # Print de lijst met verwerkte resultaatbestanden
        print(result_files)
    

# Pad naar de map waarin de resultaten worden opgeslagen
results_validatie_dir = r'C:/shhs/Results_validatie'

# Initialiseert een lege matrix (5x5) voor het optellen van de confusionmatrices
combined_matrix = pd.DataFrame(0, index=range(2, 7), columns=range(2, 7))
row_labels = []  # Lijst om de rijlabels (kolom A) van de eerste sleep_results.xlsx op te slaan
col_labels = []  # Lijst om de kolomlabels (rij 1) van de eerste sleep_results.xlsx op te slaan

# Loop door alle mappen in Results_validatie
for root, dirs, files in os.walk(results_validatie_dir):
    # Zoek naar sleep_results.xlsx bestanden
    if 'sleep_results.xlsx' in files:
        # Pad naar het sleep_results.xlsx bestand
        file_path = os.path.join(root, 'sleep_results.xlsx')

        # Lees de derde sheet genaamd 'Confusiematrix (abs)'
        try:
            df = pd.read_excel(file_path, sheet_name='Confusiematrix (abs)', header=None)

            # Haal de data uit de cellen B2 tot F6 (indexen 1 tot 5 en kolommen 1 tot 5)
            cm_data = df.iloc[1:6, 1:6].values

            # Tel deze matrix op bij de gecombineerde matrix
            combined_matrix += cm_data

            # Haal de rij- en kolomlabels uit de eerste sheet van het eerste bestand en sla deze eenmalig op
            if not row_labels and not col_labels:
                row_labels = df.iloc[1:6, 0].values.tolist()  # Kolom A (rijlabels)
                col_labels = df.iloc[0, 1:6].values.tolist()  # Rij 1 (kolomlabels)

        except Exception as e:
            print(f"Fout bij het lezen van {file_path}: {e}")

# Bereken de percentages per rij
percentage_matrix = combined_matrix.div(combined_matrix.sum(axis=1), axis=0) * 100

# Statistieken berekenen: Cohen's kappa, F1-score, Accuracy
y_true = []
y_pred = []

# Zet de matix om in lijsten van 'echte' en 'voorspelde' labels
for i in range(combined_matrix.shape[0]):
    for j in range(combined_matrix.shape[1]):
        count = int(combined_matrix.iloc[i, j])
        y_true.extend([i] * count)
        y_pred.extend([j] * count)

# Bereken prestatiematen
kappa = cohen_kappa_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

# Maak een dataframe voor de prestatiematen
statistieken_df = pd.DataFrame({
    'Metric': ["Cohen's kappa", 'F1-score', 'Accuracy'],
    'Waarde': [kappa, f1, accuracy]
})

# Pad voor het gecombineerde Excel-resultaatbestand
output_file = os.path.join(results_validatie_dir, 'combined_confusion_matrix.xlsx')

# Schrijf alle resultaten naar een Excel-bestand met 3 sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Voeg labels toe aan rijen en kolommen
    combined_matrix.index = row_labels
    combined_matrix.columns = col_labels

    percentage_matrix.index = row_labels
    percentage_matrix.columns = col_labels

    # Schrijf absolute matrix naar eerste sheet
    combined_matrix.to_excel(writer, sheet_name='Confusiematrix (abs)', startrow=1, startcol=1, index=True, header=True)

    # Schrijf procentuele matrix naar tweede sheet
    percentage_matrix.to_excel(writer, sheet_name='Confusiematrix (%)', startrow=1, startcol=1, index=True, header=True)

    # Schrijf statistieken naar derde sheet
    statistieken_df.to_excel(writer, sheet_name='Statistieken', index=False, startrow=0, startcol=0)

print(f"Gecombineerde matrix en percentages opgeslagen in: {output_file}")
