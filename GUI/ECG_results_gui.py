# Bestandsnaam: ECG_results_gui.py
# Geschreven door: Esmee Springer
# Voor het laatst bewerkt op: 23-05-2025

# Importeren van benodigde pakkages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import h5py
from tabulate import tabulate
import shutil
import re
import os

def extract_start_timestamp(ecg_folder):
    # Doorloop alle bestanden in de opgegeven map
    for file in os.listdir(ecg_folder):
        # Controleer of de bestandsnaam overeenkomt met het patroon van het ECG-bestand
        match = re.match(r"(\d{8}T\d{6}Z)_\w+_ecg_stream\.csv", file)
        if match:
            # Haal de datum en de tijd uit de bestandsnaam
            date_time_str = match.group(1)

            # Haal de datum en de tijd op uit de bestandsnaam door te splitsen op T
            date_str, time_str = date_time_str.split("T")

            # Verwijder de Z door deze te vervangen met ""
            time_str = time_str.replace("Z", "")
            try:
                # Zet de gecombineerde string om naar een datetime-object
                start_timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
                return start_timestamp
            except ValueError:
                # Bij foutmelding, ga door naar het volgende bestand
                pass
    # Geef een foutmelding wanneer er geen geldig bestand gevonden is
    raise ValueError("Start timestamp kon niet worden geëxtraheerd uit bestandsnamen in deze map.")


def get_participant_info(ecg_file_name):
    try:
        # Zoek naar het participantnummer in de bestandsnaam
        match = re.search(r'participant[_\-]?(\d+)', ecg_file_name, re.IGNORECASE)
        if match:
            # Zet het gevonden nummer om naar een geheel getal
            participant_number = int(match.group(1))
            print("Measurement of participant {0}".format(participant_number))
        else:
            # Geen deelnemer nummer gevonden
            participant_number = None
            print("Participant number could not be extracted.")
    except Exception as e:
        # Geef een foutmelding als er iets fout gaat bij het zoeken of omzetten
        print("Error extracting participant number:", e)
        participant_number = None

    # Haal de bestandsnaam uit het volledige pad
    file_name = ecg_file_name.split("/")[-1] if "/" in ecg_file_name else ecg_file_name.split("\\")[-1]

    try:
        # Bepaald de map waarin het bestand staat
        folder_path = os.path.dirname(ecg_file_name)

        # Probeer de starttijd van de meting op te halen uit de bestandsnaam
        start_timestamp = extract_start_timestamp(folder_path)

        # Print de starttijd
        print("Start timestamp:", start_timestamp)
    except Exception as e:
        # Geef foutmelding als de starttijd niet bepaald kan worden
        print("Start timestamp could not be extracted:", e)
        start_timestamp = None

    # Lees het ECG-bestand in als DataFrame
    df = pd.read_csv(ecg_file_name)

    # Haal de eerste timestamp op
    first_timestamp = df["timestamp"].iloc[0]

    # Haal de laatste timestamp op
    last_timestamp = df["timestamp"].iloc[-1]

    # Bereken hoelang de meting duurde in milliseconden
    duration_milliseconds = last_timestamp - first_timestamp

    # Zet de tijdsduur om naar seconden en vervolgens naar uren/minuten/seconden
    duration_seconds = int(duration_milliseconds / 1000)
    minutes, seconds = divmod(duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Print de tijd in uren, minuten en seconden
    print("Measurement took {0} milliseconds.".format(duration_milliseconds))
    print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

    # Bereken de duur van de meting als een timedelta object
    duration_timestamp = timedelta(seconds=duration_seconds)

    # Bepaal de eindtijd van de meting door de duur op te tellen bij de starttijd
    end_timestamp = start_timestamp + duration_timestamp if start_timestamp else None
    if start_timestamp and end_timestamp:
        print("Measurement started on {0}".format(start_timestamp))
        print("Measurement  ended  on {0}".format(end_timestamp))

    # Geef het participantnummer, de starttijd en de eindtijd terug
    return participant_number, start_timestamp, end_timestamp


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
    # Print de slaapstadia voorspellingen
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

        # Als persoon precies 5 minuten wakker is geweest, tel het als wakker worden
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
    return predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount


# Haal alle resultaten op en zet ze in een overzichtelijke tabel
def get_results_table(ecg_file_name, results_file_name, preprocessed_file, file_counter):
    # Verkrijg deelnemerinformatie zoals nummer, starttijd en eindtijd
    participant_number, start_timestamp, end_timestamp = get_participant_info(ecg_file_name)    

    # Haal de berekende resultaten op uit het resultatenbestand
    predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount = get_results(results_file_name)

    print("Deelnemer nummer:", participant_number)

    # Bepaal tijdstip van inslapen en wakker worden
    asleep_timestamp = start_timestamp + timedelta(minutes=sleep_on_set_latency)
    awake_timestamp = end_timestamp - timedelta(minutes=wake_up_set_latency)

    # Maak een overzichtstabel met meetgegevens
    headers = ["Datum van meting", str(start_timestamp.date())]
    table = [["Tijd meting begonnen", start_timestamp.time()],
             ["Tijd in slaap gevallen", asleep_timestamp.time()],
             ["Tijd wakker geworden", awake_timestamp.time()],
             ["Tijd meting beëindigt", end_timestamp.time()], 
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
             ["Totaal aantal uren geslapen", ((int(predicted_label_amount[1])/2)+(int(predicted_label_amount[2])/2)+(int(predicted_label_amount[3])/2)+(int(predicted_label_amount[4])/2))/60],
            ]
    # toon overzichtstabel in de console
    print(tabulate(table, headers, tablefmt="grid"))

    # Pad waar de ECG-data staat
    results_ecg_data_base_folder = os.path.dirname(ecg_file_name)
    
    # Bepaal de bovenliggende map van de deelnemer
    parent_folder = os.path.dirname(results_ecg_data_base_folder)

    # Maak een algemene map aan voor alle resultaten
    all_results_folder = os.path.join(parent_folder, "MoveSense_data_resultaten")
    if not os.path.exists(all_results_folder):
        os.makedirs(all_results_folder)
    
    # Maak een map specifiek voor deze deelnemer
    results_ecg_data_folder = "results_ecg_data_participant_{0}".format(participant_number)
    results_ecg_data_folder = "{0}/{1}".format(results_ecg_data_base_folder, results_ecg_data_folder)

    # Maak map aan als deze nog niet bestaat
    if not os.path.exists(results_ecg_data_folder):
        os.makedirs(results_ecg_data_folder)

    # Zet de tabel om in een DataFrame
    excel_results = pd.DataFrame(table)
    excel_filename = "Results_participant_{0}.xlsx".format(participant_number)

    # Create unique and easy to understand filename for ECG data
    results_ecg_data_file_name = "{0}/{1}".format(results_ecg_data_folder, excel_filename)

    # Bepaal de bestandsnaam voor het Excelbestand
    print(results_ecg_data_file_name)

    # Als het Excelbestand al bestaat, voeg dan een nieuw tabblad toe (Wanneer een tabblad dezelfde naam heeft, wordt deze overschreven)
    if os.path.exists(results_ecg_data_file_name):
        writer = pd.ExcelWriter(results_ecg_data_file_name, engine='openpyxl', mode="a", if_sheet_exists="replace") 
    else:
        writer = pd.ExcelWriter(results_ecg_data_file_name, engine='openpyxl') 

    # Geef het werkblad een naam gebasseerd op gevolgde nummer van het bestand
    sheet_name = "Results_day_{0}".format(file_counter)
    print(sheet_name)

    # Schrijf de resultaten naar het Excelbestand
    excel_results.to_excel(excel_writer=writer, index=False, header=headers, sheet_name=sheet_name) # Add excel sheet
    writer.close()
    
    # Kopieer het resultaat ook naar de centrale resultatenmap
    central_results_path = os.path.join(all_results_folder, excel_filename)
    shutil.copy(results_ecg_data_file_name, central_results_path)
    print(f"Resultaat gekopieerd naar centrale map: {central_results_path}")

    # Print het volledige DataFrame 
    print(excel_results)