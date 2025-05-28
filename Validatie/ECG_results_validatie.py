import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta, time
from scipy.interpolate import interp1d
import h5py
from tabulate import tabulate
import pyedflib
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shutil
import glob
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score

# Get all information necessary for results table
def get_participant_info(ecg_file_name, f):
    try:
        # Haal de startdatum en starttijd op
        # Open het EDF-bestand
        # Open het bestand
        # f = pyedflib.EdfReader(ecg_file_name)

        # Haal signaalinformatie op
        n_signals = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sampling_rates = f.getSampleFrequencies()

        # Maak een lege lijst om de data in te verzamelen
        data = []

        # Loop door elk signaal en sla de eerste 10 waarden op
        for i in range(n_signals):
            signal_data = f.readSignal(i)  # Lees signaal
            data.append([signal_labels[i], sampling_rates[i]] + list(signal_data[:10]))

        # Sluit het EDF-bestand
        f.close()

        # Zet de data in een DataFrame
        columns = ["Signaalnaam", "Samplefrequentie"] + [f"Punt {i+1}" for i in range(10)]
        df = pd.DataFrame(data, columns=columns)

        # Print het DataFrame
        # print(df)
        # f = pyedflib.EdfReader(ecg_file_name)
        # Open het EDF-bestand
        # f = pyedflib.EdfReader(ecg_file_name)
        # Haal alle signaallabels en samplefrequenties op
        signal_labels = f.getSignalLabels()
        sampling_rates = f.getSampleFrequencies()

        # Zoek de index van het ECG-signaal
        ecg_index = signal_labels.index("ECG")  # Gebruik de exacte naam

        # Lees het ECG-signaal en de samplefrequentie
        ecg_signal = f.readSignal(ecg_index)
        fs = sampling_rates[ecg_index]  # Samplefrequentie

        # Bereken timestamps (in seconden)
        timestamps = np.arange(0, len(ecg_signal)) / fs

        # Zet alles in een DataFrame
        df_ecg = pd.DataFrame({"timestamp": timestamps, "sample": ecg_signal})

        # Print de eerste paar regels
        # print(df_ecg.head())

        # Sluit het bestand
        f.close()

        filename = os.path.basename(ecg_file_name)  # 'shhs1-200001.edf'

        # Verwijder extensie en splits op '-'
        name_without_ext = os.path.splitext(filename)[0]  # 'shhs1-200001'
        dataset, participant_id = name_without_ext.split('-')

        print("Dataset:", dataset)           # 'shhs1'
        print("Participant ID:", participant_id)  # '200001'

        # Haal de ECG-waarden uit je dataframe en zet ze om naar een numpy array
        raw_ecg_samples = df_ecg["sample"].to_numpy()

        start_timestamp = f.getStartdatetime()  # Dit geeft een datetime object
        # Verkrijg de startdatum en starttijd apart
        start_date = start_timestamp.date()  # Haalt alleen de datum op
        start_time = start_timestamp.time()  # Haalt alleen de tijd op

        # Print de startdatum en starttijd apart
        print(f"Measurement start date: {start_date}")
        print(f"Measurement start time: {start_time}")
        print(f"Start date and start time: {start_timestamp}")

        # Get timestamps
        first_timestamp = df_ecg["timestamp"].iloc[0]
        last_timestamp = df_ecg["timestamp"].iloc[-1]
        # Get duration in milliseconds
        duration_milliseconds = (last_timestamp - first_timestamp)*1000 # tijd staat in seconde, om naar milliseconde te gaan *1000
        print("Measurement took {0} milliseconds.".format(duration_milliseconds))
        # # Get the measurement duration in seconds
        # duration_seconds = num_samples / sample_frequency
        # # Zet het aantal seconden om naar uren, minuten en seconden
        # minutes, seconds = divmod(int(duration_seconds), 60)  # Eerst naar minuten en seconden
        # hours, minutes = divmod(minutes, 60)  # Vervolgens naar uren en minuten

        # Get the measurement duration in seconds
        duration_seconds = int(duration_milliseconds / 1000)
        # Make a single division to produce both the quotient (minutes) and the remainder (seconds)
        minutes, seconds = divmod(duration_seconds, 60)
        hours, minutes = divmod(minutes, 60)

        print("The measurement duration took {0} seconds.".format(duration_seconds))
        print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

        # Add measurement duration to start timestamp
        # Bereken de eindtijd door de meetduur toe te voegen aan de starttijd
        end_timestamp = start_timestamp + timedelta(seconds=duration_seconds)
        print(f"Starttijd: {start_timestamp}")
        print(f"Eindtijd: {end_timestamp}")

        # Return all information for results table
        return start_timestamp, end_timestamp, df_ecg

    except:
        print("Start timestamp could not be extracted.\n")


    # Get df
    #df = pd.read_csv(ecg_file)
    # Get timestamps
    # Haal de samplefrequentie en het aantal samples van het ECG-signaal
    # ecg_index = f.getSignalLabels().index('ECG')  # Zorg ervoor dat ECG aanwezig is
    # sample_frequency = f.getSampleFrequencies()[ecg_index]
    # num_samples = f.getNSamples()[ecg_index]

    # Bereken de meetduur in seconden
    # duration_milliseconds = (num_samples / sample_frequency)*1000 #Keer 1000 zodat het in milliseconde stat
    # print(f"Measurement duration: {duration_milliseconds}")
    

# def get_participant_info(df_ecg):
#     try:
#         # Verkrijg de eerste en laatste timestamp uit het DataFrame
#         first_timestamp = df_ecg["timestamp"].iloc[0]
#         last_timestamp = df_ecg["timestamp"].iloc[-1]
        
#         # Bereken de meetduur in milliseconden
#         duration_milliseconds = (last_timestamp - first_timestamp) * 1000  # tijd in milliseconden
#         print(f"De meetduur is {duration_milliseconds} milliseconden.")
        
#         # Verkrijg de starttijd en eindtijd van de meting
#         start_timestamp = df_ecg["timestamp"].iloc[0]  # Deze zal vaak gelijk zijn aan 'first_timestamp'
#         end_timestamp = df_ecg["timestamp"].iloc[-1]  # Dit zal meestal gelijk zijn aan 'last_timestamp'
        
#         print(f"Starttijd van de meting: {start_timestamp}")
#         print(f"Eindtijd van de meting: {end_timestamp}")

#         # Bereken de tijd in uren, minuten en seconden
#         duration_seconds = int(duration_milliseconds / 1000)  # in seconden
#         minutes, seconds = divmod(duration_seconds, 60)
#         hours, minutes = divmod(minutes, 60)
        
#         print(f"De meetduur is {hours} uren, {minutes} minuten, en {seconds} seconden.")
        
#         # Return alle informatie voor de tabel
#         return start_timestamp, end_timestamp
#     except Exception as e:
#         print(f"Fout: {e}")
#         return None



# Get all results necessary for results table
def get_results(results_file, show_plots=False):
    # Open HDF5 file
    with h5py.File(results_file, 'r') as results:
        # Get confusion matrix
        confusions = results["confusions"][()]
        print(confusions)
    # Get the number of every predicted label
    predicted_label_amount = confusions[0][0]
    predicted_label_amount
    print("0=Wake:\t   ", int(predicted_label_amount[0]))
    print("1=N1/S1:   ", int(predicted_label_amount[1]))
    print("2=N2/S2:   ", int(predicted_label_amount[2]))
    print("3=N3/S3/S4:", int(predicted_label_amount[3]))
    print("4=REM:\t   ", int(predicted_label_amount[4]))
    # Open HDF5 file
    with h5py.File(results_file, 'r') as results:
        predictions = results["predictions"][()][0]
    if show_plots:
        plt.close()
        # Visialize sleep stages over time
        plt.plot(predictions)
        plt.title("Sleep stages per 30-second epochs")
        plt.ylabel("Sleep stages")
        plt.xlabel("30-second epochs")
        plt.yticks(np.unique(predictions))
        plt.show()
    # How many minutes are nessecary to see if someone is really awake
    min_minutes_awake = 5

    # Calculates how many epochs define if someone is awake (*2, because 30-second epochs)
    min_epochs_awake = min_minutes_awake * 2
    wake_state = 0
    wake_up_amount, wake_state_period = 0, 0
    # Loop through all predictions
    for idx, prediction in enumerate(predictions):
        # Check if person is in wake state
        if prediction == wake_state:
            wake_state_period += 1
        else:
            wake_state_period = 0

        # Check if person is in wake state for exactly 5 minutes
        if wake_state_period == min_epochs_awake:
            wake_up_amount += 1
    # Check if wake_state is in begin of sleep
    if np.mean(predictions[:min_epochs_awake]) == wake_state:
        wake_up_amount -= 1
    # Check if wake_state is in end of sleep
    if np.mean(predictions[-min_epochs_awake:]) == wake_state:
        wake_up_amount -= 1

    print(f"Person woke up {wake_up_amount} times at the middle of the night.")

    # Calculate epochs till person is asleep
    awake_epochs, wake_state_period = 0, 0
    # Loop through all predictions
    for idx, prediction in enumerate(predictions):
        # Check if person is in wake state
        if prediction == wake_state:
            awake_epochs +=1
        else:
            break

    # Calculate sleep-on-set latency
    sleep_on_set_latency = int(awake_epochs / 2)
    print(f"Person was awake for {sleep_on_set_latency} minutes before falling asleep.")

    # Calculate epochs from the moment person is awake
    awake_epochs = 0
    # Loop through all predictions backwards
    for idx, prediction in enumerate(np.flip(predictions)):
        # Check if person is in wake state
        if prediction == wake_state:
            awake_epochs +=1
        else:
            break

    # Calculate wake-up-set latency
    wake_up_set_latency = int(awake_epochs / 2)
    print(f"Person was awake for {wake_up_set_latency} minutes before ending the measurement.")

    # Table with some results
    table = [[sleep_on_set_latency, wake_up_set_latency, wake_up_amount]]
    headers = ["Minutes awake before sleep", "Minutes awake before measurement end", "Times woken up"]
    print(tabulate(table, headers, tablefmt="pretty"))

    # Return all calculations for visualisation in table
    return predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions



def calculate_confusion_matrix(xml_file, predictions):
    # Laad XML-bestand
    tree = ET.parse(xml_file)
    root = tree.getroot()

    sleep_stages = []

    # Doorloop alle elementen en zoek de slaapstadia
    for elem in root.iter("SleepStage"):
        sleep_stages.append(elem.text)

    df_annotation = pd.DataFrame(sleep_stages, columns=["SleepStage"])

    # Voeg een epoch kolom toe
    df_annotation["epoch"] = range(1, len(df_annotation) + 1)

    # Zorg ervoor dat de kolom 'predictions' als int64 is
    df_annotation["predictions"] = predictions.astype("int64")

    # Zorg ervoor dat de kolom 'SleepStage' als int64 is (indien niet al het geval is)
    df_annotation["SleepStage"] = df_annotation["SleepStage"].astype("int64")

    # Repareer de slaapstadia
    df_annotation["SleepStage"] = df_annotation["SleepStage"].replace(4, 3)
    df_annotation["SleepStage"] = df_annotation["SleepStage"].replace(5, 4)

    # Bereken de correctheid van de voorspellingen
    df_annotation["correct"] = (df_annotation["SleepStage"] == df_annotation["predictions"])

    # Confusiematrix berekenen
    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(df_annotation["SleepStage"], df_annotation["predictions"], labels=labels)

    # Zet de confusion matrix om naar percentages
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Zet de confusiematrix om naar een DataFrame voor Excel
    cm_df = pd.DataFrame(cm_percentage, columns=["W", "N1", "N2", "N3", "R"], index=["W", "N1", "N2", "N3", "R"])

    return cm, cm_percentage, cm_df, df_annotation["SleepStage"], df_annotation["predictions"]

# def get_labels_for_confusion(xml_file, predictions):
#     # Laad XML-bestand
#     tree = ET.parse(xml_file)
#     root = tree.getroot()

#     sleep_stages = []

#     for elem in root.iter("SleepStage"):
#         sleep_stages.append(int(elem.text))

#     # Maak dataframe en pas slaapstadia aan (zoals eerder)
#     df = pd.DataFrame(sleep_stages, columns=["SleepStage"])
#     df["SleepStage"] = df["SleepStage"].replace({4: 3, 5: 4})

#     # Zorg dat predictions ook juiste lengte en formaat hebben
#     y_true = df["SleepStage"].values[:len(predictions)]
#     y_pred = predictions[:len(y_true)]

#     return y_true, y_pred

def get_results_table_validatie(edf_file, results_file_name, xml_file, predictions, f):
    if not isinstance(edf_file, str) or not isinstance(results_file_name, str):
        print("Fout: edf_file en results_file_name moeten strings zijn.")
        return

    # Debug: Print de invoerparameters
    print(f"Invoerparameters voor get_results_table_validatie:")
    print(f"EDF-bestand: {edf_file}")
    print(f"Results-bestand: {results_file_name}")
    print(f"XML-bestand: {xml_file}")
    print(f"Voorspellingen: {predictions}")

    # Check of het XML-bestand bestaat
    if not os.path.exists(xml_file):
        print(f"Fout: XML-bestand niet gevonden op het pad: {xml_file}")
        return

    # Lees ECG-bestand
    try:
        f = pyedflib.EdfReader(edf_file)
    except Exception as e:
        print(f"Fout bij het openen van het EDF-bestand {edf_file}: {e}")
        return

    # Verkrijg signaallabels en samplefrequenties
    try:
        signal_labels = f.getSignalLabels()
        sampling_rates = f.getSampleFrequencies()
        ecg_index = signal_labels.index("ECG")
        ecg_signal = f.readSignal(ecg_index)
        fs = sampling_rates[ecg_index]
        timestamps = np.arange(0, len(ecg_signal)) / fs
        df_ecg = pd.DataFrame({"timestamp": timestamps, "sample": ecg_signal})
    except Exception as e:
        print(f"Fout bij het verwerken van het ECG-signaal in {edf_file}: {e}")
        f.close()
        return

    # Verkrijg de begintijd en eindtijd van de meting
    try:
        start_timestamp, end_timestamp, df_ecg = get_participant_info(edf_file, f) #df_ecg stond eerst in de haakjes
        # start_timestamp = datetime.datetime.fromtimestamp(start_timestamp)
        # end_timestamp = datetime.datetime.fromtimestamp(end_timestamp)
    except Exception as e:
        print(f"Fout bij het verkrijgen van de begintijd en eindtijd in {edf_file}: {e}")
        f.close()
        return

    # Verkrijg de voorspellingen
    try:
        predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions = get_results(results_file_name)
    except Exception as e:
        print(f"Fout bij het verkrijgen van de resultaten in {results_file_name}: {e}")
        f.close()
        return

    # Bereken slaap- en waakmomenten
    asleep_timestamp = start_timestamp + timedelta(minutes=sleep_on_set_latency)
    awake_timestamp = end_timestamp - timedelta(minutes=wake_up_set_latency)

    # Vul de tabel met resultaten
    headers = ["Datum van meting", start_timestamp.strftime("%d-%m-%Y")]
    table = [
        ["Tijd meting begonnen", start_timestamp.strftime("%H:%M:%S")],
        ["Tijd in slaap gevallen", asleep_timestamp.strftime("%H:%M:%S")],
        ["Tijd wakker geworden", awake_timestamp.strftime("%H:%M:%S")],
        ["Tijd meting beëindigt", end_timestamp.strftime("%H:%M:%S")],
        [],  # lege regel voor overzicht
        ["Minuten wakker voordat deelnemer in slaap viel", sleep_on_set_latency],
        ["Minuten wakker voordat deelnemer meting beëindigde", wake_up_set_latency],
        [],  # lege regel voor overzicht
        ["Aantal keer wakker geworden per nacht", wake_up_amount],
        [],  # lege regel voor overzicht
        ["Totaal aantal minuten gemeten", int(sum(predicted_label_amount))/2],
        ["Totaal aantal minuten in wakker fase", int(predicted_label_amount[0])/2],
        ["Totaal aantal minuten in N1 fase", int(predicted_label_amount[1])/2],
        ["Totaal aantal minuten in N2 fase", int(predicted_label_amount[2])/2],
        ["Totaal aantal minuten in N3 fase", int(predicted_label_amount[3])/2],
        ["Totaal aantal minuten in REM fase", int(predicted_label_amount[4])/2],
        [],  # lege regel voor overzicht
        ["Totaal aantal uren gemeten", int((sum(predicted_label_amount))/2)/60],
        ["Totaal aantal uren geslapen", (
            (int(predicted_label_amount[1])/2) +
            (int(predicted_label_amount[2])/2) +
            (int(predicted_label_amount[3])/2) +
            (int(predicted_label_amount[4])/2)
        ) / 60]
    ]

    # Voeg de datum header toe als eerste rij in de tabel
    table.insert(0, headers)

    # Roep de confusiematrix functie aan
    try:
        cm, cm_percentage, cm_df, true_labels, predicted_labels = calculate_confusion_matrix(xml_file, predictions)

        # Bereken extra prestatiematen
        from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

        kappa = cohen_kappa_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Voeg deze toe aan de tabel onderaan
        table.extend([
            [],  # lege regel voor scheiding
            ["Cohen's Kappa", round(kappa, 3)],
            ["F1-score (weighted)", round(f1, 3)],
            ["Accuracy", round(accuracy, 3)]
        ])

    except Exception as e:
        print(f"Fout bij het berekenen van de confusiematrix: {e}")
        f.close()
        return

    # Controleer of cm_percentage het juiste formaat heeft (2D-matrix)
    print(f"cm_percentage vorm: {cm_percentage.shape}")
    print(f"cm_percentage: {cm_percentage}")

    # Controleer of de confusiematrix correct is
    cm_headers = ["W", "N1", "N2", "N3", "R"]

    # Zet de confusiematrix om in een DataFrame voor het Excel-bestand
    cm_df = pd.DataFrame(cm_percentage, columns=cm_headers, index=cm_headers)
    cm_absolute_df = pd.DataFrame(cm, columns=cm_headers, index=cm_headers)

    # Maak de DataFrame voor de resultaten (met de juiste structuur)
    df_results = pd.DataFrame(table, columns=["Kenmerk", "Waarde"])

    # Extract basename (bijv: 'shhs1-204005.edf' -> 'shhs1_204005')
    edf_base = os.path.basename(edf_file).replace("-", "_").replace(".edf", "")
    result_dir_name = f"Results_{edf_base}"

    # Maak het pad naar de result folder
    base_result_path = r"C:/shhs/Results_validatie"
    full_result_path = os.path.join(base_result_path, result_dir_name)

    # Zorg dat de map bestaat
    os.makedirs(full_result_path, exist_ok=True)

    # Bepaal het Excel-bestandspad
    excel_filename = os.path.join(full_result_path, "sleep_results.xlsx")

    # Debug print waar het bestand opgeslagen wordt
    print(f"Excel bestand wordt opgeslagen in: {excel_filename}")

    # Schrijf naar Excel
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Eerste sheet: Slaapresultaten
            df_results.to_excel(excel_writer=writer, index=False, header=True, sheet_name="Slaapresultaten")
            
            # Tweede sheet: Confusiematrix in percentages
            cm_df.to_excel(excel_writer=writer, index=True, header=True, sheet_name="Confusiematrix (%)")
            
            # Derde sheet: Confusiematrix in absolute aantallen
            cm_absolute_df.to_excel(excel_writer=writer, index=True, header=True, sheet_name="Confusiematrix (abs)")
            
    except Exception as e:
        print(f"Fout bij het opslaan van het Excel-bestand: {e}")
        return

    print("Tabel succesvol opgeslagen in Excel:")
    print(df_results)
    #----------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
# def calculate_overall_performance(xml_files, predictions_list):
#     # Laten we aannemen dat xml_files en predictions_list lijsten zijn van bijbehorende ware labels en voorspellingen
#     true_labels = []
#     predicted_labels = []

#     # Voeg alle labels toe aan de lijsten
#     for xml_file, predictions in zip(xml_files, predictions_list):
#         # Hier voeg je de logica toe om de ware labels uit het XML-bestand te halen.
#         # Voor dit voorbeeld gaan we ervan uit dat xml_file de ware labels bevat en predictions de voorspellingen.
#         # True labels zijn afhankelijk van hoe je ze uit de XML haalt.
#         # Je zou een functie moeten hebben die de ware labels van een XML-bestand haalt, bijvoorbeeld:
#         # true_labels.extend(get_true_labels_from_xml(xml_file))
#         # predicted_labels.extend(predictions)
#         pass

#     # Bereken de Confusiematrix
#     cm = confusion_matrix(true_labels, predicted_labels)
#     cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

#     # Bereken Cohen's Kappa
#     kappa = cohen_kappa_score(true_labels, predicted_labels)

#     # Bereken F1-score (gewogen)
#     f1 = f1_score(true_labels, predicted_labels, average='weighted')

#     # Bereken Accuracy
#     accuracy = accuracy_score(true_labels, predicted_labels)

#     return cm, cm_percentage, pd.DataFrame(cm), kappa, f1, accuracy


# def get_results_table_validatie_multiple_files(xml_files, predictions_list, output_dir):
#     # Verkrijg de overall prestaties over alle bestanden
#     overall_cm, overall_cm_percentage, overall_cm_df, kappa, f1, accuracy = calculate_overall_performance(xml_files, predictions_list)

#     # Maak een algemene tabel van de prestatiematen (Cohen's Kappa, F1-score en Accuracy)
#     overall_performance_table = [
#         ["Cohen's Kappa", round(kappa, 3)],
#         ["F1-score (weighted)", round(f1, 3)],
#         ["Accuracy", round(accuracy, 3)],
#     ]

#     # Zet de confusiematrix om in een DataFrame voor Excel (in percentages)
#     cm_headers = ["W", "N1", "N2", "N3", "R"]
#     overall_cm_df = pd.DataFrame(overall_cm_percentage, columns=cm_headers, index=cm_headers)

#     # Maak een DataFrame voor de prestatiematen (Cohen's Kappa, F1, Accuracy)
#     df_overall_performance = pd.DataFrame(overall_performance_table, columns=["Kenmerk", "Waarde"])

#     # Zorg ervoor dat de map voor resultaten bestaat
#     os.makedirs(output_dir, exist_ok=True)

#     # Stel het pad in voor het Excel-bestand
#     excel_filename = os.path.join(output_dir, "overall_performance_results.xlsx")

#     try:
#         # Schrijf naar Excel
#         with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#             # Eerste sheet: Confusiematrix in percentages
#             overall_cm_df.to_excel(writer, index=True, header=True, sheet_name="Confusiematrix (%)")
            
#             # Tweede sheet: Prestatiematen (Cohen's Kappa, F1-score, Accuracy)
#             df_overall_performance.to_excel(writer, index=False, header=True, sheet_name="Prestatiematen")

#         print(f"Excel bestand wordt opgeslagen in: {excel_filename}")
    
#     except Exception as e:
#         print(f"Fout bij het opslaan van het Excel-bestand: {e}")




#----------------------------------------------------------
# def get_results_table_validatie(edf_file, results_file_name, xml_file, predictions):
#     if not isinstance(edf_file, str) or not isinstance(results_file_name, str):
#         print("Fout: edf_file en results_file_name moeten strings zijn.")
#         return

#     # Lees ECG-bestand
#     f = pyedflib.EdfReader(edf_file)
#     signal_labels = f.getSignalLabels()
#     sampling_rates = f.getSampleFrequencies()
#     ecg_index = signal_labels.index("ECG")
#     ecg_signal = f.readSignal(ecg_index)
#     fs = sampling_rates[ecg_index]
#     timestamps = np.arange(0, len(ecg_signal)) / fs
#     df_ecg = pd.DataFrame({"timestamp": timestamps, "sample": ecg_signal})
#     f.close()

#     # Verkrijg de begintijd en eindtijd van de meting
#     start_timestamp, end_timestamp = get_participant_info(df_ecg)
#     start_timestamp = datetime.fromtimestamp(start_timestamp)
#     end_timestamp = datetime.fromtimestamp(end_timestamp)

#     # Verkrijg de voorspellingen
#     predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions = get_results(results_file_name)

#     asleep_timestamp = start_timestamp + timedelta(minutes=sleep_on_set_latency)
#     awake_timestamp = end_timestamp - timedelta(minutes=wake_up_set_latency)

#     headers = ["Datum van meting", start_timestamp.strftime("%d-%m-%Y")]
#     table = [
#         ["Tijd meting begonnen", start_timestamp.strftime("%H:%M:%S")],
#         ["Tijd in slaap gevallen", asleep_timestamp.strftime("%H:%M:%S")],
#         ["Tijd wakker geworden", awake_timestamp.strftime("%H:%M:%S")],
#         ["Tijd meting beëindigt", end_timestamp.strftime("%H:%M:%S")],
#         [],  # lege regel voor overzicht
#         ["Minuten wakker voordat deelnemer in slaap viel", sleep_on_set_latency],
#         ["Minuten wakker voordat deelnemer meting beëindigde", wake_up_set_latency],
#         [],  # lege regel voor overzicht
#         ["Aantal keer wakker geworden per nacht", wake_up_amount],
#         [],  # lege regel voor overzicht
#         ["Totaal aantal minuten gemeten", int(sum(predicted_label_amount))/2],
#         ["Totaal aantal minuten in wakker fase", int(predicted_label_amount[0])/2],
#         ["Totaal aantal minuten in N1 fase", int(predicted_label_amount[1])/2],
#         ["Totaal aantal minuten in N2 fase", int(predicted_label_amount[2])/2],
#         ["Totaal aantal minuten in N3 fase", int(predicted_label_amount[3])/2],
#         ["Totaal aantal minuten in REM fase", int(predicted_label_amount[4])/2],
#         [],  # lege regel voor overzicht
#         ["Totaal aantal uren gemeten", int((sum(predicted_label_amount))/2)/60],
#         ["Totaal aantal uren geslapen", ((int(predicted_label_amount[1])/2)+(int(predicted_label_amount[2])/2)+(int(predicted_label_amount[3])/2)+(int(predicted_label_amount[4])/2))/60]
#     ]

#     # Roep de confusiematrix functie aan
#     cm, cm_percentage, cm_df = calculate_confusion_matrix(xml_file, predictions)

#     # Voeg de confusiematrix data toe aan de tabel
#     confusion_matrix_data = cm_percentage.flatten()
#     cm_headers = ["W", "N1", "N2", "N3", "R"]
#     table.append(["Confusiematrix (Wakker, N1, N2, N3, R)"] + confusion_matrix_data.tolist())

#     # Debug: Toon de confusiematrix
#     print("Confusiematrix (percentages):\n", cm_percentage)

#     excel_results = pd.DataFrame(table)

#     # ===== AANPASSING HIER =====
#     # Extract basename (bijv: 'shhs1-204005.edf' -> 'shhs1_204005')
#     edf_base = os.path.basename(edf_file).replace("-", "_").replace(".edf", "")
#     result_dir_name = f"Results_{edf_base}"

#     # Maak het pad naar de result folder
#     base_result_path = r"C:/shhs/Results_validatie"
#     full_result_path = os.path.join(base_result_path, result_dir_name)

#     # Zorg dat de map bestaat
#     os.makedirs(full_result_path, exist_ok=True)

#     # Bepaal het Excel-bestandspad
#     excel_filename = os.path.join(full_result_path, "sleep_results.xlsx")

#     # Debug print waar het bestand opgeslagen wordt
#     print(f"Excel bestand wordt opgeslagen in: {excel_filename}")

#     # Schrijf naar Excel
#     with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#         excel_results.to_excel(excel_writer=writer, index=False, header=headers, sheet_name="Slaapresultaten")

#     print(excel_results)
# DE GOEDE_-------------------------------------------------------------------------------------------------------------------------------------------------

# def get_results_table_validatie(edf_file, results_file_name):
#     if not isinstance(edf_file, str) or not isinstance(results_file_name, str):
#         print("Fout: edf_file en results_file_name moeten strings zijn.")
#         return

#     f = pyedflib.EdfReader(edf_file)
#     signal_labels = f.getSignalLabels()
#     sampling_rates = f.getSampleFrequencies()
#     ecg_index = signal_labels.index("ECG")
#     ecg_signal = f.readSignal(ecg_index)
#     fs = sampling_rates[ecg_index]
#     timestamps = np.arange(0, len(ecg_signal)) / fs
#     df_ecg = pd.DataFrame({"timestamp": timestamps, "sample": ecg_signal})
#     f.close()

#     start_timestamp, end_timestamp = get_participant_info(df_ecg)
#     start_timestamp = datetime.fromtimestamp(start_timestamp)
#     end_timestamp = datetime.fromtimestamp(end_timestamp)

#     predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount  = get_results(results_file_name)

#     asleep_timestamp = start_timestamp + timedelta(minutes=sleep_on_set_latency)
#     awake_timestamp = end_timestamp - timedelta(minutes=wake_up_set_latency)

#     headers = ["Datum van meting", start_timestamp.strftime("%d-%m-%Y")]
#     table = [
#         ["Tijd meting begonnen", start_timestamp.strftime("%H:%M:%S")],
#         ["Tijd in slaap gevallen", asleep_timestamp.strftime("%H:%M:%S")],
#         ["Tijd wakker geworden", awake_timestamp.strftime("%H:%M:%S")],
#         ["Tijd meting beëindigt", end_timestamp.strftime("%H:%M:%S")],
#         [],
#         ["Minuten wakker voordat deelnemer in slaap viel", sleep_on_set_latency],
#         ["Minuten wakker voordat deelnemer meting beëindigde", wake_up_set_latency],
#         [],
#         ["Aantal keer wakker geworden per nacht", wake_up_amount],
#         [],
#         ["Totaal aantal minuten gemeten", int(sum(predicted_label_amount))/2],
#         ["Totaal aantal minuten in wakker fase", int(predicted_label_amount[0])/2],
#         ["Totaal aantal minuten in N1 fase", int(predicted_label_amount[1])/2],
#         ["Totaal aantal minuten in N2 fase", int(predicted_label_amount[2])/2],
#         ["Totaal aantal minuten in N3 fase", int(predicted_label_amount[3])/2],
#         ["Totaal aantal minuten in REM fase", int(predicted_label_amount[4])/2],
#         [],
#         ["Totaal aantal uren gemeten", int((sum(predicted_label_amount))/2)/60],
#         ["Totaal aantal uren geslapen", ((int(predicted_label_amount[1])/2)+(int(predicted_label_amount[2])/2)+(int(predicted_label_amount[3])/2)+(int(predicted_label_amount[4])/2))/60]
#     ]

#     print(tabulate(table, headers, tablefmt="grid"))

#     excel_results = pd.DataFrame(table)

#     # ===== AANPASSING HIER =====
#     # Extract basename (bijv: 'shhs1-204005.edf' -> 'shhs1_204005')
#     edf_base = os.path.basename(edf_file).replace("-", "_").replace(".edf", "")
#     result_dir_name = f"Results_{edf_base}"

#     # Maak het pad naar de result folder
#     base_result_path = r"C:/shhs/Results_validatie"
#     full_result_path = os.path.join(base_result_path, result_dir_name)

#     # Zorg dat de map bestaat
#     os.makedirs(full_result_path, exist_ok=True)

#     # Bepaal het Excel-bestandspad
#     excel_filename = os.path.join(full_result_path, "sleep_results.xlsx")

#     # Debug print waar het bestand opgeslagen wordt
#     print(f"Excel bestand wordt opgeslagen in: {excel_filename}")

#     # Schrijf naar Excel
#     with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#         excel_results.to_excel(excel_writer=writer, index=False, header=headers, sheet_name="Slaapresultaten")

#     print(excel_results)


#----------------------------------------------------------------------------------------------------------------------


# def get_results_table_validatie(edf_file, results_file_name, selected_xml_folder):
#     if not isinstance(edf_file, str) or not isinstance(results_file_name, str):
#         print("Fout: edf_file en results_file_name moeten strings zijn.")
#         return

#     # Laad de EDF-bestanden en verkregen gegevens (zoals eerder in je code)
#     f = pyedflib.EdfReader(edf_file)
#     signal_labels = f.getSignalLabels()
#     sampling_rates = f.getSampleFrequencies()
#     ecg_index = signal_labels.index("ECG")
#     ecg_signal = f.readSignal(ecg_index)
#     fs = sampling_rates[ecg_index]
#     timestamps = np.arange(0, len(ecg_signal)) / fs
#     df_ecg = pd.DataFrame({"timestamp": timestamps, "sample": ecg_signal})
#     f.close()

#     start_timestamp, end_timestamp = get_participant_info(df_ecg)
#     start_timestamp = datetime.fromtimestamp(start_timestamp)
#     end_timestamp = datetime.fromtimestamp(end_timestamp)

#     predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions = get_results(results_file_name)

#     asleep_timestamp = start_timestamp + timedelta(minutes=sleep_on_set_latency)
#     awake_timestamp = end_timestamp - timedelta(minutes=wake_up_set_latency)

#     headers = ["Datum van meting", start_timestamp.strftime("%d-%m-%Y")]
#     table = [
#         ["Tijd meting begonnen", start_timestamp.strftime("%H:%M:%S")],
#         ["Tijd in slaap gevallen", asleep_timestamp.strftime("%H:%M:%S")],
#         ["Tijd wakker geworden", awake_timestamp.strftime("%H:%M:%S")],
#         ["Tijd meting beëindigt", end_timestamp.strftime("%H:%M:%S")],
#         [],
#         ["Minuten wakker voordat deelnemer in slaap viel", sleep_on_set_latency],
#         ["Minuten wakker voordat deelnemer meting beëindigde", wake_up_set_latency],
#         [],
#         ["Aantal keer wakker geworden per nacht", wake_up_amount],
#         [],
#         ["Totaal aantal minuten gemeten", int(sum(predicted_label_amount))/2],
#         ["Totaal aantal minuten in wakker fase", int(predicted_label_amount[0])/2],
#         ["Totaal aantal minuten in N1 fase", int(predicted_label_amount[1])/2],
#         ["Totaal aantal minuten in N2 fase", int(predicted_label_amount[2])/2],
#         ["Totaal aantal minuten in N3 fase", int(predicted_label_amount[3])/2],
#         ["Totaal aantal minuten in REM fase", int(predicted_label_amount[4])/2],
#         [],
#         ["Totaal aantal uren gemeten", int((sum(predicted_label_amount))/2)/60],
#         ["Totaal aantal uren geslapen", ((int(predicted_label_amount[1])/2)+(int(predicted_label_amount[2])/2)+(int(predicted_label_amount[3])/2)+(int(predicted_label_amount[4])/2))/60]
#     ]

#     print(tabulate(table, headers, tablefmt="grid"))

#     excel_results = pd.DataFrame(table)

#     # ===== AANPASSING HIER =====
#     # Extract basename (bijv: 'shhs1-204005.edf' -> 'shhs1_204005')
#     edf_base = os.path.basename(edf_file).replace("-", "_").replace(".edf", "")
#     result_dir_name = f"Results_{edf_base}"

#     # Maak het pad naar de result folder
#     base_result_path = r"C:/shhs/Results_validatie"
#     full_result_path = os.path.join(base_result_path, result_dir_name)

#     # Zorg dat de map bestaat
#     os.makedirs(full_result_path, exist_ok=True)

#     # Bepaal het Excel-bestandspad
#     excel_filename = os.path.join(full_result_path, "sleep_results.xlsx")

#     # Debug print waar het bestand opgeslagen wordt
#     print(f"Excel bestand wordt opgeslagen in: {excel_filename}")

#     # Schrijf naar Excel
#     with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
#         excel_results.to_excel(excel_writer=writer, index=False, header=headers, sheet_name="Slaapresultaten")

#         # Loop door alle geselecteerde XML-bestanden in de folder
#         for xml_file in os.listdir(selected_xml_folder):
#             # Controleer of het bestand een XML-bestand is
#             if xml_file.endswith(".xml"):
#                 xml_file_path = os.path.join(selected_xml_folder, xml_file)

#                 # Laad het XML-bestand om de werkelijke slaapstadia (geannoteerd) en voorspellingen te verkrijgen
#                 tree = ET.parse(xml_file_path)
#                 root = tree.getroot()

#                 sleep_stages = []
#                 for elem in root.iter("SleepStage"):
#                     sleep_stages.append(elem.text)

#                 # Maak DataFrame voor annotaties
#                 df_annotation = pd.DataFrame(sleep_stages, columns=["SleepStage"])

#                 # Voeg de voorspellingen toe (in dit geval heb je die al in 'predictions')
#                 df_annotation["epoch"] = range(1, len(df_annotation) + 1)
#                 df_annotation["predictions"] = predictions.astype("int64")

#                 # Zorg ervoor dat de 'SleepStage' kolom als int64 is
#                 df_annotation["SleepStage"] = df_annotation["SleepStage"].astype("int64")

#                 # Repareer de slaapstadia
#                 df_annotation["SleepStage"] = df_annotation["SleepStage"].replace(4, 3)
#                 df_annotation["SleepStage"] = df_annotation["SleepStage"].replace(5, 4)

#                 # Bereken de correctheid van de voorspellingen
#                 df_annotation["correct"] = (df_annotation["SleepStage"] == df_annotation["predictions"])

#                 # Confusiematrix berekenen
#                 labels = [0, 1, 2, 3, 4]
#                 cm = confusion_matrix(df_annotation["SleepStage"], df_annotation["predictions"], labels=labels)

#                 # Zet de confusion matrix om naar percentages
#                 cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

#                 # Zet de confusiematrix om naar een DataFrame voor Excel
#                 cm_df = pd.DataFrame(cm_percentage, columns=["W", "N1", "N2", "N3", "R"], index=["W", "N1", "N2", "N3", "R"])

#                 # Schrijf de confusiematrix naar een nieuw sheet in hetzelfde Excel-bestand
#                 cm_df.to_excel(excel_writer=writer, index=True, sheet_name=f"Confusiematrix_{xml_file[:-4]}")

#     print("Confusiematrixen zijn toegevoegd aan het Excel-bestand.")






