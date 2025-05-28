import pandas as pd
import os
import subprocess
from Prepare_ECG_files_validatie import get_hdf5_file_validatie
from ECG_results_validatie import get_results_table_validatie
from ECG_results_validatie import get_results
# from ECG_results_validatie import get_results_table_validatie_multiple_files
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
# from ECG_results_validatie import get_labels_for_confusion
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

# TIJDELIJK UITGEZET HIERONDER--------------------------------------------------------
# # Functie om een folder volledig te verwijderen
# def remove_folder(folder_path):
#     if os.path.exists(folder_path):
#         try:
#             shutil.rmtree(folder_path)  # Verwijder de gehele map
#             print(f"De map {folder_path} is verwijderd.")
#         except Exception as e:
#             print(f"Fout bij het verwijderen van {folder_path}: {e}")

# # Verwijder de Results_validatie map (en alle submappen) als deze bestaat
# results_validatie_dir = r'C:/shhs/Results_validatie'
# remove_folder(results_validatie_dir)

# # folder met de edf bestanden met ECG data. 
shhs_dir = r'C:/shhs/polysomnography'


# all_shhs_paths = glob.glob(os.path.join(shhs_dir, 'edfs', 'shhs1', '*.edf')) + \
#                  glob.glob(os.path.join(shhs_dir, 'edfs', 'shhs2', '*.edf'))

# all_shhs_paths = sorted(all_shhs_paths)

# # Selecteer willekeurig 1000 bestanden (zonder vervanging)
# n_files = 1000
# selected_paths = random.sample(all_shhs_paths, min(n_files, len(all_shhs_paths)))
# print(f"{len(selected_paths)} bestanden geselecteerd voor verwerking")
# TIJDELIJK UITGEZET HIERBOVEN----------------------------------------------------------------

# # Pad naar de map waar je de geselecteerde bestanden wilt opslaan
output_dir = r'C:/shhs/selected_edf_files'
xml_output_dir = r'C:/shhs/selected_xml_files'

#TIJDELIJK UITGEZET-HIERONDER------------------------------------------------------
# # Functie om een map leeg te maken (maar niet verwijderen)
# def clear_folder(folder_path):
#     if os.path.exists(folder_path):
#         files = glob.glob(os.path.join(folder_path, '*'))
#         for f in files:
#             try:
#                 os.remove(f)
#             except Exception as e:
#                 print(f"Kon bestand niet verwijderen: {f}, fout: {e}")
#     else:
#         os.makedirs(folder_path)

# # Leegmaken van beide outputmappen
# clear_folder(output_dir)
# clear_folder(xml_output_dir)
# os.makedirs(output_dir, exist_ok=True)
# TIJDELIJK UITGEZETHIERBOVEN-------------------------------------------------------------------------

#TIJDELIJKE VOOR HERVATTING ----------------------------------------
selected_paths = glob.glob(os.path.join(output_dir, '*.edf'))
selected_paths = sorted(selected_paths)
# -------------------------------------------------------------=----

# Verwerk elk geselecteerd bestand
# TIJDELIJK UITGEZET HIERONDER-------------------
# for i, edf_file in enumerate(selected_paths):
# TIJDE;LIJK UITGEZET HIERBOVEN--------------------
# TIJDELIJK VOOR HERVATTING
for i, edf_file in enumerate(selected_paths[777:], start=777):
#--------------------------------------------------------------------
    print(f"\n[{i+1}/{len(selected_paths)}] Inladen van: {edf_file}")
    
    try:
        # Open EDF-bestand
        f = pyedflib.EdfReader(edf_file)

        # Basisinfo
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

       # Controleer de samplefrequentie (DIT TOEGEVOEGD)
        ecg_fs = sampling_rates[ecg_index]
        min_required_fs = 101  # <-- Hier kun je je eigen grenswaarde zetten
        if ecg_fs < min_required_fs:
            print(f"Samplefrequentie van ECG is te laag ({ecg_fs} Hz), bestand wordt overgeslagen.")
            f.close()
            continue 

        # Lees ECG-data in
        ecg_data = f.readSignal(ecg_index)

        # (Hier kun je je eigen preprocessing pipeline aanroepen, bijv:)
        # ecg_processed = jouw_preprocess_functie(ecg_data)

        # Debug: Toon eerste 10 waarden
        print("Eerste 10 ECG-datapunten:", ecg_data[:10])

        # Sluit bestand
        f.close()

        # Kopieer het EDF-bestand naar de outputmap
        shutil.copy(edf_file, os.path.join(output_dir, os.path.basename(edf_file)))
        print(f"Bestand gekopieerd naar: {output_dir}/{os.path.basename(edf_file)}")

        # XML-bestanden opslaan in een aparte map
        # xml_output_dir = r'C:/shhs/selected_xml_files'
        # os.makedirs(xml_output_dir, exist_ok=True)

        # Zoek de bijbehorende XML-bestandsnaam
        edf_filename = os.path.basename(edf_file)
        base_filename = os.path.splitext(edf_filename)[0]  # zonder .edf
        xml_filename = base_filename + '-profusion.xml'

        # Bepaal submap (shhs1 of shhs2)
        if 'shhs1' in edf_file:
            xml_full_path = os.path.join(shhs_dir, 'annotations-events-profusion', 'shhs1', xml_filename)
        elif 'shhs2' in edf_file:
            xml_full_path = os.path.join(shhs_dir, 'annotations-events-profusion', 'shhs2', xml_filename)
        else:
            print("Onbekende submap voor:", edf_file)
            continue


        # Kopieer het XML-bestand
        if os.path.exists(xml_full_path):
            shutil.copy(xml_full_path, os.path.join(xml_output_dir, xml_filename))
            print(f"Bijbehorend XML-bestand gekopieerd naar: {xml_output_dir}/{xml_filename}")
        else:
            print(f"Waarschuwing: XML-bestand niet gevonden voor {edf_filename}")

    except Exception as e:
        print(f"Fout bij verwerken van bestand {edf_file}: {e}")
        continue

    
# Folder with all ecg files - 
ecg_folders = [
    "C:/shhs/selected_edf_files/",
    # "C:/shhs/selected_edf_files/",
    # "C:/shhs/selected_edf_files/",
    
]

# Loop door alle folders met edf-bestanden
for ecg_folder in ecg_folders:
    # Get all ECG files
    edf_files = []
    # Haal alle edf-bestanden op
    for root, dirs, files in os.walk(ecg_folder):
        print(f"Bestanden in {root}: {files}")
        # Loop door alle bestanden in de map
        for file in files:
            # Zoek alleen naar .edf bestanden (zonder verdere beperking zoals 'ecg' in naam)
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))

# xml_files = []
# # Folder with all xml files
# xml_folders = [
#     "C:/shhs/selected_xml_files/",
#     # Je kunt hier eventueel andere mappen toevoegen waar XML-bestanden zich bevinden
# ]

# # Loop door alle folders met xml-bestanden
# for xml_folder in xml_folders:
#     # Lijst voor XML-bestanden
    
#     # Haal alle XML-bestanden op
#     for root, dirs, files in os.walk(xml_folder):
#         print(f"Bestanden in {root}: {files}")
        
#         # Loop door alle bestanden in de map
#         for file in files:
#             # Zoek alleen naar .xml bestanden
#             if file.endswith(".xml"):
#                 xml_files.append(os.path.join(root, file))

#     # Als je de xml_files lijst wilt gebruiken, bijvoorbeeld voor verdere verwerking:
#     print(f"Gevonden XML-bestanden: {xml_files}")
# Folder with all xml files
xml_folders = [
    "C:/shhs/selected_xml_files/",
    # Je kunt hier eventueel andere mappen toevoegen waar XML-bestanden zich bevinden
]

# We halen eerst alle XML-bestanden op, zodat we deze kunnen koppelen aan de juiste ECG-bestanden
xml_files = {}  # Een dictionary om XML-bestanden te koppelen aan hun base-bestandsnaam
for xml_folder in xml_folders:
    for root, dirs, files in os.walk(xml_folder):
        print(f"Bestanden in {root}: {files}")
        for file in files:
            if file.endswith(".xml"):
                # base_filename = os.path.splitext(file)[0]  # zonder .xml
                # xml_files[base_filename] = os.path.join(root, file)
                base_filename = file.replace("-profusion.xml", "")
                xml_files[base_filename] = os.path.join(root, file)

    print(f"Gevonden XML-bestanden: {xml_files}")

# Folder with all ECG files
ecg_folders = [
    "C:/shhs/selected_edf_files/",
    # Je kunt hier eventueel andere mappen toevoegen waar ECG-bestanden zich bevinden
]
#TIJDELIJK UITGEZET HIERONDER -----------------------------
# # Loop door alle folders met edf-bestanden
# for ecg_folder in ecg_folders:
#     # Haal alle EDF-bestanden op
#     edf_files = []
#     for root, dirs, files in os.walk(ecg_folder):
#         print(f"Bestanden in {root}: {files}")
#         for file in files:
#             if file.endswith(".edf"):
#                 edf_files.append(os.path.join(root, file))

#         #GOEDE CODE
#     # Preprocess all ECG files
#     preprocessed_files, result_files = [], []
#     for ecg_file in edf_files:
#         ecg_file_name = os.path.join(ecg_folder, ecg_file)
# TIJDELIJK UITGEZET HIERBOVEN----------------------------------------
# TIJDELIJK VOOR HERVATTEN-----------------------------------------
# Verzamel en sorteer alle .edf bestanden
edf_files = []
for root, dirs, files in os.walk(ecg_folder):
    for file in files:
        if file.endswith(".edf"):
            edf_files.append(os.path.join(root, file))

edf_files = sorted(edf_files)  # Belangrijk voor consistente volgorde!

# Begin pas vanaf dataset 777
edf_files = edf_files[777:]

print(f"Totaal aantal te verwerken bestanden vanaf index 777: {len(edf_files)}")

preprocessed_files = []
result_files = []

# Loop door deze geslice-de lijst
for ecg_file in edf_files:
    ecg_file_name = ecg_file
    print(f"Verwerken: {ecg_file_name}")
    # --------------------------------------------------------------------
#HIERONDER INSPRINGING TERUGGEZET, DUS STRAKS WEER TAB----------------------
        # Analyse ecg file name
    preprocessed_file, preprocessed_df = get_hdf5_file_validatie(ecg_file_name)
    print("ecg_file_name:", ecg_file_name)
    preprocessed_files.append(preprocessed_file)
    print(preprocessed_files)

    ''' Run Machine Learning model '''
    # Ensure you are in the right directory by writing this code line in the Python environment terminal 
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

    # Verkrijg de resultaten van de HDF5-bestanden, inclusief predictions
    predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions = get_results(results_file_name)
#INSPRINGING BEEINDIGD--------------------------------------

        # # Verzamel alle y_true en y_pred over alle bestanden
        # all_y_true = []
        # all_y_pred = []

        # for ecg_file, result_file in zip(edf_files, result_files):
        #     # Verkrijg de base naam van het ECG-bestand
        #     base_filename = os.path.splitext(os.path.basename(ecg_file))[0]

        #     # Koppel het juiste XML-bestand
        #     xml_path = xml_files.get(base_filename)
        #     if not xml_path:
        #         print(f"Geen XML-bestand gevonden voor {base_filename}, sla over.")
        #         continue

        #     # Verkrijg predictions (uit de al bestaande lijst 'result_files')
        #     predictions = get_results(result_file)[-1]  # Alleen predictions nodig

        #     # Haal de echte en voorspelde labels op
        #     y_true, y_pred = get_labels_for_confusion(xml_path, predictions)

        #     all_y_true.extend(y_true)
        #     all_y_pred.extend(y_pred)

        # # Plot confusion matrix
        # if all_y_true and all_y_pred:
        #     cm = confusion_matrix(all_y_true, all_y_pred)
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #     disp.plot(cmap=plt.cm.Blues)
        #     plt.title("Confusion Matrix over alle geselecteerde bestanden")
        #     plt.show()
        # else:
        #     print("Niet genoeg gegevens om een confusion matrix te genereren.")

#HIERONDER INSPRINGING TERUGGEZET, DUS STRAKS WEER TAB----------------------
    ecg_filename = os.path.basename(ecg_file)
    base_filename = os.path.splitext(ecg_filename)[0]  # bijvoorbeeld 'shhs1-200318'

    # Zoek naar een XML-bestand met exact dezelfde base filename
    xml_full_path = xml_files.get(base_filename)

    if xml_full_path:
        print(f"XML-bestand gevonden voor {ecg_filename}: {xml_full_path}")
    else:
        print(f"Geen XML-bestand gevonden voor {ecg_filename}.")
        continue

    # Roep de functie get_results_table_validatie aan en geef het specifieke XML-bestand door
    get_results_table_validatie(ecg_file_name, results_file_name, xml_full_path, predictions, f)

    print(result_files)
    #INSPRINGING BEEINDIGD---------------------

# # Pad naar de map waarin de resultaten worden opgeslagen
# results_validatie_dir = r'C:/shhs/Results_validatie'

# # Matrix die we willen optellen
# combined_matrix = pd.DataFrame(0, index=range(2, 7), columns=range(2, 7))

# # Loop door alle mappen in Results_validatie
# for root, dirs, files in os.walk(results_validatie_dir):
#     # Zoek naar sleep_results.xlsx bestanden
#     if 'sleep_results.xlsx' in files:
#         # Pad naar het sleep_results.xlsx bestand
#         file_path = os.path.join(root, 'sleep_results.xlsx')

#         # Lees de derde sheet genaamd 'Confusiematrix (abs)'
#         try:
#             df = pd.read_excel(file_path, sheet_name='Confusiematrix (abs)', header=None)
#             # Haal de data uit de cellen B2 tot F6 (indexen 1 tot 5 en kolommen 1 tot 5)
#             cm_data = df.iloc[1:6, 1:6].values

#             # Voeg de matrix toe aan de gecombineerde matrix
#             combined_matrix += cm_data
#         except Exception as e:
#             print(f"Fout bij het lezen van {file_path}: {e}")

# # Schrijf de gecombineerde matrix naar een nieuwe Excel-bestand
# output_file = os.path.join(results_validatie_dir, 'combined_confusion_matrix.xlsx')

# with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
#     combined_matrix.to_excel(writer, sheet_name='Confusiematrix (abs)', startrow=1, startcol=1, header=False, index=False)

# print(f"Gecombineerde matrix opgeslagen in: {output_file}")
    

# Pad naar de map waarin de resultaten worden opgeslagen
results_validatie_dir = r'C:/shhs/Results_validatie'

# Matrix die we willen optellen
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

            # Voeg de matrix toe aan de gecombineerde matrix
            combined_matrix += cm_data

            # Haal de rij- en kolomlabels uit de eerste sheet van het eerste bestand
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

for i in range(combined_matrix.shape[0]):
    for j in range(combined_matrix.shape[1]):
        count = int(combined_matrix.iloc[i, j])
        y_true.extend([i] * count)
        y_pred.extend([j] * count)

kappa = cohen_kappa_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

# Maak dataframe voor output
statistieken_df = pd.DataFrame({
    'Metric': ["Cohen's kappa", 'F1-score', 'Accuracy'],
    'Waarde': [kappa, f1, accuracy]
})

# Maak een nieuwe Excel-bestand met twee sheets
output_file = os.path.join(results_validatie_dir, 'combined_confusion_matrix.xlsx')

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Voeg labels toe aan rijen en kolommen
    combined_matrix.index = row_labels
    combined_matrix.columns = col_labels

    percentage_matrix.index = row_labels
    percentage_matrix.columns = col_labels

    # Schrijf naar eerste sheet
    combined_matrix.to_excel(writer, sheet_name='Confusiematrix (abs)', startrow=1, startcol=1, index=True, header=True)

    # Schrijf naar tweede sheet
    percentage_matrix.to_excel(writer, sheet_name='Confusiematrix (%)', startrow=1, startcol=1, index=True, header=True)

    # Schrijf naar derde sheet
    statistieken_df.to_excel(writer, sheet_name='Statistieken', index=False, startrow=0, startcol=0)

print(f"Gecombineerde matrix en percentages opgeslagen in: {output_file}")













    # file_counter = 1
    # preprocessed_files, result_files = [], []
    # # Preprocess all ECG files
    # for ecg_file in edf_files:
    #     ecg_file_name = os.path.join(ecg_folder, ecg_file)

    #     # Analyse ecg file name
    #     preprocessed_file, preprocessed_df = get_hdf5_file_validatie(ecg_file_name)
    #     print("ecg_file_name:", ecg_file_name)
    #     preprocessed_files.append(preprocessed_file)
    #     print(preprocessed_files)


    #     ''' Run Machine Learning model '''
    #     # Ensure you are are in the right directionary by writing this code line in the Python environment terminal 
    #     '''
    #     > cd .\ecg-sleep-staging\your_own_data\primary_model
    #     '''

    #     # Path to train script
    #     train_script = os.getcwd() + "/train.py"

    #     # Run model in Python environment terminal with subprocess
    #     subprocess.run(["python", train_script, preprocessed_file], check=True)


    #     ''' Compute results '''
    #     # Get result files
    #     results_file_name = os.getcwd() + "/results.h5"
    #     result_files.append(results_file_name)

    #     # Analyse HDF5 files
    #     # get_results_table_validatie(ecg_file_name, results_file_name, xml_files)
    #     # Verkrijg de resultaten van de HDF5-bestanden, inclusief predictions
    #     predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions = get_results(results_file_name)

    #     # Roep de functie get_results_table_validatie aan en geef predictions door
    #     get_results_table_validatie(ecg_file_name, results_file_name, xml_files, predictions)

    #     print(result_files)
    #     file_counter += 1