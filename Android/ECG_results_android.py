# Bestandsnaam: ECG_results_android.py
# Naam : Esmee Springer
# Voor het laatst bewerkt op: 03-06-2025

# Importeren van benodigde pakkages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import h5py
from tabulate import tabulate
import shutil
import os

# Functie om alle benodigde informatie voor de resultatentabel te verkrijgen
def get_participant_info(ecg_file_name, preprocessed_df):
    try:
        # Haal het deelnemernummer uit het pad van het ECG-bestand
        participant_number = ecg_file_name.split("/")[-2]
        participant_number = int(participant_number.split("_")[-1])
        print("Measurement of participant {0}".format(participant_number))
    except:
        print("Participant number could not be extracted.")  

    # Haal de bestandsnaam uit het volledige pad
    file_name = ecg_file_name.split("/")[-1]

    try:
    # Haal de startdatum uit de bestandsnaam
        # Splits de naam op 'T' om de datum en tijd te scheiden
        start_date_time = file_name.split("T")
        # Het deel voor de 'T' bevat de datum; haal jaar, maand en dag eruit
        start_date = start_date_time[0].split("-")[-3:]  
        start_date = "-".join(start_date)  # Zet de datum terug in het juiste formaat YYYY-MM-DD

        print("Measurement start date:", start_date)

        # Haal de starttijd uit de bestandsnaam
        start_time = start_date_time[1]
        start_time = start_time.split(".")[0]
        # Vervang underscores door dubbele punten om de tijd correct te tonen
        start_time = start_time.replace("_", ":")
        print("Measurement start time:", start_time)  

        # Maak een datetime object van de startdatum en -tijd
        start_timestamp = datetime.strptime("{0} {1}".format(start_date, start_time), "%Y-%m-%d %H:%M:%S")
        print(start_timestamp)
    except:
        print("Start timestamp could not be extracted.\n")
        print("Are you sure you have the correct file name?")
        print("Expected file name should end like this: xxxxxxxxTxxxxxxZ_xxxxxxxxxxxx_ecg_stream.csv")
        print("Your file name looks like this:", ecg_file_name)


    # Gebruik de al ingelzen dataframe
    df = preprocessed_df
    print(df["timestamp"].iloc[0])

    # Haal het eerste en laatste timestamp uit het DataFrame
    first_timestamp = df["timestamp"].iloc[0]
    last_timestamp = df["timestamp"].iloc[-1]

    # Bereken de duur van de meting in milliseconden
    duration_milliseconds = (last_timestamp - first_timestamp) * 1000
    print("Measurement took {0} milliseconds.".format(duration_milliseconds))

    # Bereken de duur van de meting in seconden
    duration_seconds = int(duration_milliseconds / 1000)
    # Verdeel de duur in minuten en seconden
    minutes, seconds = divmod(duration_seconds, 60)
    # Verdeel de duur in minuten en seconden
    hours, minutes = divmod(minutes, 60)

    print("The measurement duration took {0} seconds.".format(duration_seconds))
    print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

    # Voeg de duur toe aan het starttijdstip om het eindtijdstip te bepalen
    duration_timestamp = timedelta(seconds=duration_seconds)
    end_timestamp = start_timestamp + duration_timestamp
    print("Measurement started on {0}".format(start_timestamp))
    print("Measurement  ended  on {0}".format(end_timestamp))

    # Geef alle verzamelde informatie terug voor de resultatentabel
    return participant_number, start_timestamp, end_timestamp


# Functie om alle resultaten op te halen die nodig zijn voor de resultatentabel
def get_results(results_file, show_plots=False):
    # Open het HDF5-bestand waarin de resultaten zijn opgeslagen
    with h5py.File(results_file, 'r') as results:
        # Verkrijg de confusion matrix
        confusions = results["confusions"][()]
        print(confusions)

    # Haal het aantal voorspellingen per slaapfase op
    predicted_label_amount = confusions[0][0]
    predicted_label_amount
    print("0=Wake:\t   ", int(predicted_label_amount[0]))
    print("1=N1/S1:   ", int(predicted_label_amount[1]))
    print("2=N2/S2:   ", int(predicted_label_amount[2]))
    print("3=N3/S3/S4:", int(predicted_label_amount[3]))
    print("4=REM:\t   ", int(predicted_label_amount[4]))

    # Open opnieuw het bestand om de voorspellingen (slaapstadia per tijdseenheid) op te halen
    with h5py.File(results_file, 'r') as results:
        predictions = results["predictions"][()][0]

    # Als show_plots=True, toon dan een grafiek van de slaapstadia over tijd
    if show_plots:
        plt.close()
        # Maak een lijnplot van de slaapstadia per 30 seconden
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

    # Loop door alle voorspellingen om wakker worden te detecteren
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

    # Bereken het aantal epochs dat de persoon wakker was aan het begin, totdat het slapen begint
    awake_epochs, wake_state_period = 0, 0
    # Loop door alle voorspellingen om wakker zijn voor het inslapen te meten
    for idx, prediction in enumerate(predictions):
        # Check of de persoon wakker is
        if prediction == wake_state:
            awake_epochs +=1
        else:
            break

    # Bereken de slaaplatentie in minuten (tijd wakker voor het inslapen)
    sleep_on_set_latency = int(awake_epochs / 2)
    print(f"Person was awake for {sleep_on_set_latency} minutes before falling asleep.")

    # Bereken het aantal epochs dat de persoon wakker was aan het einde van de meting
    awake_epochs = 0
    # Loop door alle voorspellingen voor wakker zijn aan het einde van de nacht, maar dan backwards
    for idx, prediction in enumerate(np.flip(predictions)):
        if prediction == wake_state:
            awake_epochs +=1
        else:
            break

    # Bereken de wakker-latentie aan het einde van de meting in minuten
    wake_up_set_latency = int(awake_epochs / 2)
    print(f"Person was awake for {wake_up_set_latency} minutes before ending the measurement.")

    # Maak een tabel met de belangrijkste resultaten
    table = [[sleep_on_set_latency, wake_up_set_latency, wake_up_amount]]
    headers = ["Minutes awake before sleep", "Minutes awake before measurement end", "Times woken up"]
    print(tabulate(table, headers, tablefmt="pretty"))

    # Geef alle berekende waarden terug voor verdere verwerking of visualisatie
    return predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount


# Haal alle resultaten op en zet ze in een overzichtelijke tabel
def get_results_table_android(ecg_file_name, results_file_name, preprocessed_file, file_counter, preprocessed_df):
    # Verkrijg deelnemerinformatie zoals nummer, starttijd en eindtijd
    participant_number, start_timestamp, end_timestamp = get_participant_info(ecg_file_name, preprocessed_df)    

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
             ["Totaal aantal uren geslapen", ((int(predicted_label_amount[1])/2)+(int(predicted_label_amount[2])/2)+(int(predicted_label_amount[3])/2)+(int(predicted_label_amount[4])/2))/60]
            ]
    # Toon overzichtstabel in de console
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

    # Bepaal de bestandsnaam voor het Excelbestand
    excel_filename = "Results_participant_{0}.xlsx".format(participant_number)

    # Bepaal het volledige pad naar het Excelbestand binnen de deelnemersmap
    results_ecg_data_file_name = "{0}/{1}".format(results_ecg_data_folder, excel_filename)
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