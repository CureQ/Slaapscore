import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import h5py
from tabulate import tabulate
import shutil

# Get all information necessary for results table
def get_participant_info(ecg_file_name):
    try:
        # Get the participant number
        participant_number = ecg_file_name.split("/")[-2]
        participant_number = int(participant_number.split("_")[-1])
        print("Measurement of participant {0}".format(participant_number))
    except:
        print("Participant number could not be extracted.")  

    # Get the full file name
    file_name = ecg_file_name.split("/")[-1]

    try:
        # Get the start date out of the file name
        start_date_time = file_name.split("_")[0]
        start_date_time = start_date_time.split("T")
        start_date = start_date_time[0]

        # Get the start time out of the file name
        start_time = start_date_time[1]
        start_time = start_time.split("Z")[0]

        # Get the start date and time in a nice format
        start_timestamp = datetime.strptime("{0} {1}".format(start_date, start_time), "%Y%m%d %H%M%S")
    except:
        print("Start timestamp could not be extracted.\n")
        print("Are you sure you have the correct file name?")
        print("Expected file name should end like this: xxxxxxxxTxxxxxxZ_xxxxxxxxxxxx_ecg_stream.csv")
        print("Your file name looks like this:", ecg_file_name)

    # Get df
    df = pd.read_csv(ecg_file_name)
    # Get timestamps
    first_timestamp = df["timestamp"].iloc[0]
    last_timestamp = df["timestamp"].iloc[-1]
    # Get duration in milliseconds
    duration_milliseconds = last_timestamp - first_timestamp
    print("Measurement took {0} milliseconds.".format(duration_milliseconds))

    # Get the measurement duration in seconds
    duration_seconds = int(duration_milliseconds / 1000)
    # Make a single division to produce both the quotient (minutes) and the remainder (seconds)
    minutes, seconds = divmod(duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print("The measurement duration took {0} seconds.".format(duration_seconds))
    print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

    # Add measurement duration to start timestamp
    duration_timestamp = timedelta(seconds=duration_seconds)
    end_timestamp = start_timestamp + duration_timestamp
    print("Measurement started on {0}".format(start_timestamp))
    print("Measurement  ended  on {0}".format(end_timestamp))

    # Return all information for results table
    return participant_number, start_timestamp, end_timestamp


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
    return predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount


# Get all results in a table
def get_results_table(ecg_file_name, results_file_name, preprocessed_file, file_counter):
    # Get participant info
    participant_number, start_timestamp, end_timestamp = get_participant_info(ecg_file_name)    

    # Get all calculations
    predicted_label_amount, sleep_on_set_latency, wake_up_set_latency, wake_up_amount = get_results(results_file_name)

    print("Deelnemer nummer:", participant_number)

    # Time person fell asleep and woke up
    asleep_timestamp = start_timestamp + timedelta(minutes=sleep_on_set_latency)
    awake_timestamp = end_timestamp - timedelta(minutes=wake_up_set_latency)

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
    print(tabulate(table, headers, tablefmt="grid"))

    import os
    # Folder for ECG data
    results_ecg_data_base_folder = os.path.dirname(ecg_file_name)
    #toegevoegd----------------------------------------
    # Bepaal de bovenliggende map van de huidige participant-map (bijv. 'MoveSense_data_participant_12')
    parent_folder = os.path.dirname(results_ecg_data_base_folder)

    # Maak een algemene map aan voor alle resultaten
    all_results_folder = os.path.join(parent_folder, "MoveSense_data_resultaten")
    if not os.path.exists(all_results_folder):
        os.makedirs(all_results_folder)
    # tot hier toegevoegd zojuist---------------------------------

    results_ecg_data_folder = "results_ecg_data_participant_{0}".format(participant_number)
    results_ecg_data_folder = "{0}/{1}".format(results_ecg_data_base_folder, results_ecg_data_folder)

    # Create folder for results data if not exists
    if not os.path.exists(results_ecg_data_folder):
        os.makedirs(results_ecg_data_folder)

    # Write all data to Excel sheet
    excel_results = pd.DataFrame(table)
    excel_filename = "Results_participant_{0}.xlsx".format(participant_number)

    # Create unique and easy to understand filename for ECG data
    results_ecg_data_file_name = "{0}/{1}".format(results_ecg_data_folder, excel_filename)
    print(results_ecg_data_file_name)

    # Check if excel file exists
    if os.path.exists(results_ecg_data_file_name):
        writer = pd.ExcelWriter(results_ecg_data_file_name, engine='openpyxl', mode="a", if_sheet_exists="replace") # Write an excel data sheet without loosing original data
    else:
        writer = pd.ExcelWriter(results_ecg_data_file_name, engine='openpyxl') # Create excel file with the first data sheet

    # Create sheet name with the day
    sheet_name = "Results_day_{0}".format(file_counter)
    print(sheet_name)

    excel_results.to_excel(excel_writer=writer, index=False, header=headers, sheet_name=sheet_name) # Add excel sheet
    writer.close()
    #toegevoegd------------------------------------------------
    # Bestemmingspad in de centrale resultatenmap
    central_results_path = os.path.join(all_results_folder, excel_filename)

    # Kopieer het bestand
    shutil.copy(results_ecg_data_file_name, central_results_path)
    print(f"Resultaat gekopieerd naar centrale map: {central_results_path}")
    # tot hier toegevoegd---------------------------------------------
    print(excel_results)