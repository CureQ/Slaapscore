# Bestandsnaam: ECG_results_validatie.py
# Naam: Esmee Springer
# Voor het laastst bewerkt op: 03-06-2025

# Importeren van benodigde pakkages
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

# Functie om alle benodigde informatie op te halen voor de resultatentabel
def get_participant_info(ecg_file_name, f):
    try:
        # Haal het aantal signalen, de signaallabels en de samplefrequentie op
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

        # Maak een DataFrame met de signaalnamen, frequenties en eerste 10 punten
        columns = ["Signaalnaam", "Samplefrequentie"] + [f"Punt {i+1}" for i in range(10)]
        df = pd.DataFrame(data, columns=columns)

        # Haal nogmaals alle signaallabels en samplefrequenties op
        signal_labels = f.getSignalLabels()
        sampling_rates = f.getSampleFrequencies()

        # Zoek de index van het ECG-signaal
        ecg_index = signal_labels.index("ECG")  # Gebruik de exacte naam

        # Lees het ECG-signaal en bijbehorende samplefrequentie
        ecg_signal = f.readSignal(ecg_index)
        fs = sampling_rates[ecg_index]  # Samplefrequentie

        # Bereken timestamps (in seconden)
        timestamps = np.arange(0, len(ecg_signal)) / fs

        # Zet het ECG signaal en de timestamps in een DataFrame
        df_ecg = pd.DataFrame({"timestamp": timestamps, "sample": ecg_signal})

        # Sluit het bestand
        f.close()

        # Haal de bestandsnaam op
        filename = os.path.basename(ecg_file_name)  # 'shhs1-200001.edf'

        # Verwijder extensie en splits op '-' om dataset en participant ID te verkrijgen
        name_without_ext = os.path.splitext(filename)[0]  # 'shhs1-200001'
        dataset, participant_id = name_without_ext.split('-')

        # Print de datasetnaam en participant ID
        print("Dataset:", dataset)           # 'shhs1'
        print("Participant ID:", participant_id)  # '200001'

        # Haal de ECG-waarden uit je dataframe en zet ze om naar een numpy array
        raw_ecg_samples = df_ecg["sample"].to_numpy()

        # Haal de startdatum en starttijd op als datetime-object
        start_timestamp = f.getStartdatetime()  

        # Verkrijg de startdatum en starttijd apart
        start_date = start_timestamp.date()  # Haalt alleen de datum op
        start_time = start_timestamp.time()  # Haalt alleen de tijd op

        # Print de startdatum en starttijd 
        print(f"Measurement start date: {start_date}")
        print(f"Measurement start time: {start_time}")
        print(f"Start date and start time: {start_timestamp}")

        # Haal eerste en laatste timestamp op
        first_timestamp = df_ecg["timestamp"].iloc[0]
        last_timestamp = df_ecg["timestamp"].iloc[-1]

        # Bereken de duur in milliseconden (tijd staat in seconden, dus vermenigvuldig met 1000)
        duration_milliseconds = (last_timestamp - first_timestamp)*1000 
        print("Measurement took {0} milliseconds.".format(duration_milliseconds))

        # Zet milliseconden om naar seconden
        duration_seconds = int(duration_milliseconds / 1000)

        # Bereken uren, minuten en seconden uit de totale seconden
        minutes, seconds = divmod(duration_seconds, 60)
        hours, minutes = divmod(minutes, 60)

        # Print de duur in seconden en als tijdseenheden
        print("The measurement duration took {0} seconds.".format(duration_seconds))
        print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

        # Bereken de eindtijd door de meetduur op te tellen bij de starttijd
        end_timestamp = start_timestamp + timedelta(seconds=duration_seconds)
        print(f"Starttijd: {start_timestamp}")
        print(f"Eindtijd: {end_timestamp}")

        # Geef alle informatie terug die nodig is voor de resultatentabel
        return start_timestamp, end_timestamp, df_ecg

    except:
        # Geef een foutmelding wanneer de starttijd niet kan worden opgehaald
        print("Start timestamp could not be extracted.\n")


# Functie om alle resultaten op te halen die nodig zijn voor de resultatentabel 
def get_results(results_file, show_plots=False):
    # Open HDF5 file met de resultaten
    with h5py.File(results_file, 'r') as results:
        # Haal de confusion matrix op
        confusions = results["confusions"][()]
        print(confusions)

    # Haal het aantal voorspellingen per slaapstadium op
    predicted_label_amount = confusions[0][0]
    predicted_label_amount
    print("0=Wake:\t   ", int(predicted_label_amount[0]))
    print("1=N1/S1:   ", int(predicted_label_amount[1]))
    print("2=N2/S2:   ", int(predicted_label_amount[2]))
    print("3=N3/S3/S4:", int(predicted_label_amount[3]))
    print("4=REM:\t   ", int(predicted_label_amount[4]))

    # Open het HDF5 bestand opnieuw om voorspellingen op te halen
    with h5py.File(results_file, 'r') as results:
        predictions = results["predictions"][()][0]

    # Als show_plots=True, maak dan een grafiek van de slaapstadia in de tijd
    if show_plots:
        plt.close()
        # Visualiseer de slaapstadia over de tijd
        plt.plot(predictions)
        plt.title("Sleep stages per 30-second epochs")
        plt.ylabel("Sleep stages")
        plt.xlabel("30-second epochs")
        plt.yticks(np.unique(predictions))
        plt.show()

    # Definieer hoeveel minuten nodig zijn om wakker zijn te detecteren
    min_minutes_awake = 5

    # Bereken hoeveel epochs (30 sec) overeenkomen met deze minuten
    min_epochs_awake = min_minutes_awake * 2

    # Variabele voor de wake state (wakker)
    wake_state = 0

    # Teller voor aantal keren wakker geworden 's nachts
    wake_up_amount, wake_state_period = 0, 0

    # Loop door alle voorspellingen heen
    for idx, prediction in enumerate(predictions):
        # Controleer of de persoon wakker is
        if prediction == wake_state:
            wake_state_period += 1
        else:
            wake_state_period = 0

        # Check if person is in wake state for exactly 5 minutes
        if wake_state_period == min_epochs_awake:
            wake_up_amount += 1

    # Controleer of de persoon aan het begin van de meting al wakker was (corrigeer de teller)
    if np.mean(predictions[:min_epochs_awake]) == wake_state:
        wake_up_amount -= 1
    # Controleer of de persoon aan het einde van de meting wakker was (corrigeer de teller)
    if np.mean(predictions[-min_epochs_awake:]) == wake_state:
        wake_up_amount -= 1

    print(f"Person woke up {wake_up_amount} times at the middle of the night.")

    # Bereken hoe lang het duurde voordat de persoon in slaap viel
    awake_epochs, wake_state_period = 0, 0
    # Loop door alle voorspellingen
    for idx, prediction in enumerate(predictions):
        # Controleer of de persoon wakker is
        if prediction == wake_state:
            awake_epochs +=1
        else:
            break

    # Bereken de slaaplatentie (=hoelang het duurt voordat de persoon in slaap valt)
    sleep_on_set_latency = int(awake_epochs / 2)
    print(f"Person was awake for {sleep_on_set_latency} minutes before falling asleep.")

    # Bereken het aantal epochs dat de persoon wakker was aan het einde van de meting
    awake_epochs = 0
    # Loop door alle voorspellingen voor wakker zijn aan het einde van de nacht, maar dan backwards
    for idx, prediction in enumerate(np.flip(predictions)):
        # Controleer of de persoon wakker is
        if prediction == wake_state:
            awake_epochs +=1
        else:
            break

    # Bereken de wakker-latentie, hoelang de persoon wakker was voordat de meting stopte
    wake_up_set_latency = int(awake_epochs / 2)
    print(f"Person was awake for {wake_up_set_latency} minutes before ending the measurement.")

    # Maak een tabel met de belangrijkste resultaten
    table = [[sleep_on_set_latency, wake_up_set_latency, wake_up_amount]]
    headers = ["Minutes awake before sleep", "Minutes awake before measurement end", "Times woken up"]
    print(tabulate(table, headers, tablefmt="pretty"))

    # Geef alle berekende waarden terug voor verdere verwerking of visualisatie
    return predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount, predictions


# Functie om een confusion matrix te berekenen op basis van de XML-annotaties en voorspellingen
def calculate_confusion_matrix(xml_file, predictions):
    # Laad XML-bestand met annotaties
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Maak een lijst om slaapstadia in op te slaan
    sleep_stages = []

    # Doorloop alle "Sleepstage"- elementen en voeg deze toe aan de lijst
    for elem in root.iter("SleepStage"):
        sleep_stages.append(elem.text)

    # Zet de annotaties om naar een DataFrame
    df_annotation = pd.DataFrame(sleep_stages, columns=["SleepStage"])

    # Voeg een kolom toe met epoch-nummers (beginnend vanaf 1)
    df_annotation["epoch"] = range(1, len(df_annotation) + 1)

    # Zorg ervoor dat de kolom 'predictions' int64 is
    df_annotation["predictions"] = predictions.astype("int64")

    # Zorg ervoor dat de kolom 'SleepStage' als int64 is (Voor vergelijking)
    df_annotation["SleepStage"] = df_annotation["SleepStage"].astype("int64")

    # Corrigeer de slaapstadia
    # 4 (N4) wordt samengevoegd met 3 (N3)
    # 5 (REM) wordt herschaald naar 4 (correcte index voor REM)
    df_annotation["SleepStage"] = df_annotation["SleepStage"].replace(4, 3)
    df_annotation["SleepStage"] = df_annotation["SleepStage"].replace(5, 4)

    # Bereken de correctheid van de voorspellingen
    df_annotation["correct"] = (df_annotation["SleepStage"] == df_annotation["predictions"])

    # Confusiematrix berekenen op basis van geannoteerde en voorspelde slaapstadia
    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(df_annotation["SleepStage"], df_annotation["predictions"], labels=labels)

    # Zet de confusion matrix om naar percentages
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Zet de confusiematrix om naar een DataFrame voor Excel
    cm_df = pd.DataFrame(cm_percentage, columns=["W", "N1", "N2", "N3", "R"], index=["W", "N1", "N2", "N3", "R"])

    # Geef ruwe matrix, procentuele matrix, als DataFrame en als originele waarden terug.
    return cm, cm_percentage, cm_df, df_annotation["SleepStage"], df_annotation["predictions"]


def get_results_table_validatie(edf_file, results_file_name, xml_file, predictions, f):
    # Controleer of de bestanden van het type string zijn
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

    # Probeer het ECG-bestand te openen
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

    # Verkrijg de starttijd en eindtijd van de meting
    try:
        start_timestamp, end_timestamp, df_ecg = get_participant_info(edf_file, f) 
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

    # Maak een overzichtstabel met resultaten
    headers = ["Datum van meting", start_timestamp.strftime("%d-%m-%Y")]
    table = [
        ["Tijd meting begonnen", start_timestamp.strftime("%H:%M:%S")],
        ["Tijd in slaap gevallen", asleep_timestamp.strftime("%H:%M:%S")],
        ["Tijd wakker geworden", awake_timestamp.strftime("%H:%M:%S")],
        ["Tijd meting beëindigt", end_timestamp.strftime("%H:%M:%S")],
        [],  
        ["Minuten wakker voordat deelnemer in slaap viel", sleep_on_set_latency],
        ["Minuten wakker voordat deelnemer meting beëindigde", wake_up_set_latency],
        [],  
        ["Aantal keer wakker geworden per nacht", wake_up_amount],
        [],  
        ["Totaal aantal minuten gemeten", int(sum(predicted_label_amount))/2],
        ["Totaal aantal minuten in wakker fase", int(predicted_label_amount[0])/2],
        ["Totaal aantal minuten in N1 fase", int(predicted_label_amount[1])/2],
        ["Totaal aantal minuten in N2 fase", int(predicted_label_amount[2])/2],
        ["Totaal aantal minuten in N3 fase", int(predicted_label_amount[3])/2],
        ["Totaal aantal minuten in REM fase", int(predicted_label_amount[4])/2],
        [],  
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

        # Bereken de prestatiematen Cohen's Kappa, F1-score en Accuracy
        from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

        kappa = cohen_kappa_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Voeg prestatiematen toe aan de tabel 
        table.extend([
            [],  
            ["Cohen's Kappa", round(kappa, 3)],
            ["F1-score (weighted)", round(f1, 3)],
            ["Accuracy", round(accuracy, 3)]
        ])

    except Exception as e:
        print(f"Fout bij het berekenen van de confusiematrix: {e}")
        f.close()
        return

    # Toon vorm en inhoud van de confusion matrix in percentages
    print(f"cm_percentage vorm: {cm_percentage.shape}")
    print(f"cm_percentage: {cm_percentage}")

    # Definieer headers voor confusionmatrix
    cm_headers = ["W", "N1", "N2", "N3", "R"]

    # Zet de confusiematrix om in een DataFrame voor het Excel-bestand
    cm_df = pd.DataFrame(cm_percentage, columns=cm_headers, index=cm_headers)
    cm_absolute_df = pd.DataFrame(cm, columns=cm_headers, index=cm_headers)

    # Maak het DataFrame voor de resultaten 
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

    # Schrijf gegevens naar Excel-bestand met meerdere tabbladen
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

    # Print bevestiging met resultaatoverzicht
    print("Tabel succesvol opgeslagen in Excel:")
    print(df_results)
    #----------------------------------------------------------------------------------------------------
