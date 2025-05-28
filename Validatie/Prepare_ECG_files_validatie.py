import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime, timedelta, time
import datetime
from datetime import timedelta, time
from scipy.interpolate import interp1d
import csv
import json
import re
from io import StringIO
import pyedflib
import glob
import os
from scipy.signal import find_peaks
import shutil
import h5py

        
def get_hdf5_file_validatie(edf_file, show_plots=False):
    # # Cardiosomnography data preprocessing
    # 
    # The ECG data needs to be preprocessed, before the ECG data can be given as input for the neural network.
    # 
    # ### Read the raw MoveSense ECG data

    # Open het bestand
    f = pyedflib.EdfReader(edf_file)

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
    print(df)

    # Open het EDF-bestand
    f = pyedflib.EdfReader(edf_file)

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
    print(df_ecg.head())

    # Sluit het bestand
    f.close()


    # ### Visualize ECG values of raw data

    if show_plots:
        plt.close()
        # Plot the ECG values of a whole night
        plt.plot(df_ecg["timestamp"], df_ecg["sample"])
        plt.title("Raw data ECG measurement")
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("ECG values")
        plt.show()

    # <br><br>
    # 
    # ---
    # 
    # ### First, get all provided information about the measurement
    # Haal bestandsnaam eruit

    filename = os.path.basename(edf_file)  # 'shhs1-200001.edf'

    # Verwijder extensie en splits op '-'
    name_without_ext = os.path.splitext(filename)[0]  # 'shhs1-200001'
    dataset, participant_id = name_without_ext.split('-')

    print("Dataset:", dataset)           # 'shhs1'
    print("Participant ID:", participant_id)  # '200001'

    # Haal de ECG-waarden uit je dataframe en zet ze om naar een numpy array
    raw_ecg_samples = df_ecg["sample"].to_numpy()  

    # Haal de startdatum en starttijd op
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

    # Get the measurement duration in seconds
    duration_seconds = int(duration_milliseconds / 1000)
    # Make a single division to produce both the quotient (minutes) and the remainder (seconds)
    minutes, seconds = divmod(duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    print("The measurement duration took {0} seconds.".format(duration_seconds))
    print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

    # Bereken de eindtijd door de meetduur toe te voegen aan de starttijd
    end_timestamp = start_timestamp + timedelta(seconds=duration_seconds)
    print(f"Starttijd: {start_timestamp}")
    print(f"Eindtijd: {end_timestamp}")

    # Bereken de gemiddelde tijd tussen metingen
    measurement_interval = duration_seconds / len(df_ecg)

    # Bereken de samplefrequentie (Hertz)
    hertz = round(1 / measurement_interval)
    # Print de resultaten
    print("Gemiddeld tijdsinterval tussen elke meting is: ")
    print(" - {0:.3f} seconden.".format(measurement_interval))
    print(" - {0:.3f} milliseconden.".format(measurement_interval * 1000))
    print("\nSamplefrequentie: {0} Hertz.".format(hertz))

# Add extra column with timestamps
    def add_timestamps(df_ecg, start_timestamp, end_timestamp):
        # Start timestamp and end timestamp
        print("Starting timestamp: ", start_timestamp)
        print("Ending timestamp:", end_timestamp, "\n\n")
        
        # Calculate the total amount of minutes between start- and end time
        total_minutes = (end_timestamp - start_timestamp).total_seconds() / 60
        # Calculate the total amount of measurement per minute
        measurements_per_minute = len(df_ecg) / total_minutes
        
        # Create a list of timestamps
        df_ecg["Timestamp"] = [start_timestamp + timedelta(minutes=i/measurements_per_minute) for i in range(len(df_ecg))]
        return df_ecg
    
    # Add timestamps for each measurement
    df_ecg = add_timestamps(df_ecg, start_timestamp, end_timestamp)
    df_ecg

    # Visualize ecg data at any time
    def plot_data(df_ecg, ecg_samples=np.array([]), title="Raw data ECG measurement", y_range=[0,0], x_range=[0,0]):
        plt.close()
        # Plot the ECG values of a whole night
        if ecg_samples.any():
            ecg_samples = pd.Series(ecg_samples)
            plt.plot(df_ecg["Timestamp"], ecg_samples)
        else:
            #print(df["Timestamp"])
            # print(df.columns)
            # print(df)
            plt.plot(df_ecg["Timestamp"], df_ecg["sample"])

        # Plot y range
        if y_range != [0,0]:
            plt.ylim(y_range[0], y_range[1])
        # Plot x range
        if x_range != [0,0]:
            plt.xlim(x_range[0], x_range[1])


        plt.title(title)
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    if show_plots:
        plot_data(df_ecg)

        # <br><br>
        # 
        # ---
        # 
        # ## Filter noise
        # 
        # ### High pass filter
        # 
        # Filter High-pass at 0.5 Hertz to remove baseline wander.

    # Signal processing functions for filtering the data
    from scipy.signal import butter, filtfilt, iirnotch, resample

    # High-pass filter to remove baseline wander
    # High-pass filter to remove baseline wander
    def highpass_filter(data, cutoff=0.5, fs=hertz, order=4): # cutoff stond op 0,5)
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    # Apply highpass filter to raw ecg samples
    highpass_filtered_ecg = highpass_filter(raw_ecg_samples)
    highpass_filtered_ecg

    # ### Remove line noise
    # 
    # Line noise (50/60 Hertz) and any other constant-frequency noise should be removed with notch filters.

    # Notch filter to remove power line noise (e.g., 50/60 Hz)
    def notch_filter(data, freq=50, fs=hertz, quality_factor=30):
        nyquist = 0.5 * fs
        freq = freq / nyquist
        b, a = iirnotch(freq, quality_factor)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    # Apply notch filter for both 50 Hz and (if needed) 60 Hz
    filtered_ecg = notch_filter(highpass_filtered_ecg, freq=50, fs=hertz)  # Apply 50 Hz notch filter
    filtered_ecg

    filtered_ecg = notch_filter(filtered_ecg, freq=60, fs=hertz)  # Apply 60 Hz notch filter (if applicable)
    filtered_ecg

    df_ecg["Timestamp"][0]

    x_range = [df_ecg["Timestamp"][998000], df_ecg["Timestamp"][1000000]]
    y_range = [-1, 1]
    if show_plots:
        plot_data(df_ecg, y_range=y_range, x_range=x_range)

    title = "Filtered ECG data on noise"
    if show_plots:
        plot_data(df_ecg, filtered_ecg, title)

    # <br><br>
    # 
    # ---
    # 
    # ## Sample data at 256 Hertz
    # 
    # Sample the data from the originally measured 125 Hertz to the new 256 Hertz.
    # 
    # This resampling takes place, because the neural network is trained on ECG datasets of 256 Hertz.
    # 
    # #### Sample ECG data

    new_hertz = 256 # New amount of Hertz
    # Resample the ECG data from 191 Hz to 256 Hz
    resampled_ecg = resample(filtered_ecg, int(len(filtered_ecg) * (new_hertz / hertz)))
    resampled_ecg

    #resample time axis
    original_timestamps = df_ecg["timestamp"].to_numpy() # Original timestamps numpy array
    original_timestamps

    # Lineaire interpolation for timestamps
    time_original = np.arange(len(original_timestamps)) / hertz  # Original timesteps in seconds
    time_resampled = np.linspace(0, time_original[-1], len(resampled_ecg))  # New timesteps in seconds

    print("Original timesteps: ", time_original[:5], "...", time_original[-5:])
    print("Resampled timesteps:", time_resampled[:5], "...", time_resampled[-5:])

    # Interpolate original timestamps to new time-axis
    resampled_timestamps = np.interp(time_resampled, time_original, original_timestamps)

    print("Original timestamps: ", original_timestamps[:5], "...", original_timestamps[-5:])
    print("Resampled timestamps:", resampled_timestamps[:5], "...", resampled_timestamps[-5:])

    resampled_df = pd.DataFrame({
        'timestamp': resampled_timestamps,
        'sample': resampled_ecg
    })

    # Add timestamps for each measurement
    resampled_df = add_timestamps(resampled_df, start_timestamp, end_timestamp)
    resampled_df

    title = f"Resampled ECG data from {hertz} Hz to {new_hertz} Hz"
    if show_plots:
        plot_data(resampled_df, title=title)

    # <br><br>
    # 
    # ---
    # 
    # ## Scale data - 1 / 2
    # 
    # ### Subtract median data
    # 
    # The median of all data should be subtracted. The median of all ECG data should be equal to 0.

    # Calculate the median of the resampled ECG data
    median_ecg = np.median(resampled_ecg)
    median_ecg

    # Subtract the median to center the data around 0
    centered_ecg = resampled_ecg - median_ecg

    np.abs(np.median(centered_ecg))

    resampled_df["sample"] = centered_ecg
    resampled_df

    title = "Median of ECG data is now equal to 0"
    if show_plots:
        plot_data(resampled_df, centered_ecg, title, y_range=y_range, x_range=x_range)

    # ---
    # 
    # ## Spikes & HeartRate
    # 
    # ### Calculate threshold of heartbeat spikes

    spike_threshold = np.std(resampled_df["sample"])
    spike_threshold

    if show_plots:
        plt.close()
        # Plot ECG values with threshold line
        plt.plot(df_ecg["Timestamp"][998000:1000000], df_ecg["sample"][998000:1000000])
        plt.ylim([-1, 1])
        # Plot threshold line
        plt.axhline(spike_threshold, color="r")
        plt.title("Spike threshold = {:.2f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    # Gebruik find_peaks om pieken te detecteren
    peaks, properties = find_peaks(df_ecg["sample"], height=spike_threshold, distance=fs*0.6)

    if show_plots:
        # Plot ECG met gedetecteerde pieken
        plt.figure(figsize=(10, 5))
        plt.plot(df_ecg["Timestamp"], df_ecg["sample"], label="ECG Signaal")
        plt.ylim([-1, 1])
        plt.axhline(spike_threshold, color="r", linestyle="--", label="Threshold")

        # Plot pieken
        plt.scatter(df_ecg["Timestamp"].iloc[peaks], df_ecg["sample"].iloc[peaks], color='red', marker='o', label="Gedetecteerde pieken")

        # Labels en titel
        plt.title("Hartslag pieken detecteren met find_peaks")
        plt.xlabel("Tijd (s)")
        plt.ylabel("ECG Amplitude")
        plt.xticks(rotation=30)
        plt.legend()
        plt.show()

    # Toon aantal pieken
    print(f"Aantal gedetecteerde pieken: {len(peaks)}")

    # Haal de tijdstempels van de gedetecteerde pieken
    peak_timestamps = df_ecg["Timestamp"].iloc[peaks]

    # Bereken de tijdsverschillen tussen de pieken (RR-intervals) in seconden
    time_differences = peak_timestamps.diff().dt.total_seconds()

    # Verwijder de eerste NaN waarde (er is geen verschil voor de eerste piek)
    #time_differences = time_differences[1:]

    # Bekijk de eerste paar tijdsverschillen
    print(time_differences.head(20))

    # Bereken de hartslag per piek (BPM)
    bpm_values = 60 / time_differences

    # Toon de BPM-waarden voor de eerste paar pieken
    print(bpm_values.head(20))

    # Maak een nieuw DataFrame voor de gedetecteerde pieken
    spike_value_df = df_ecg.iloc[peaks]  # Gebruik de indices van de pieken om de relevante rijen te selecteren

    # Voeg de tijdsverschillen toe aan dit nieuwe DataFrame (al eerder berekend)
    spike_value_df["time_differences"] = time_differences.values
    # Bereken de hartslag voor elke piek (in bpm)
    spike_value_df["heartrate"] = 60 / spike_value_df["time_differences"]

    # Bekijk het nieuwe DataFrame
    print(spike_value_df.head(20))

    if show_plots:
        # Plot van de hartslag over tijd
        plt.figure(figsize=(10, 5))
        plt.plot(spike_value_df["Timestamp"], spike_value_df["heartrate"])
        plt.title("Hartslag (BPM) over tijd")
        plt.xlabel("Tijd")
        plt.ylabel("Hartslag (BPM)")
        plt.xticks(rotation=30)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ### Plot spikes
    # inzoomen voor duidelijke weergave van ECG-signaal
    resampled_x_range = [0, 2048000]
    spikes_x_range = [0, 6864]

    if show_plots:
        plt.close()
        # Plot ECG values with threshold line
        plt.plot(resampled_df["Timestamp"][resampled_x_range[0]:resampled_x_range[1]], resampled_df["sample"][resampled_x_range[0]:resampled_x_range[1]])
        #plt.plot(df["Timestamp"][998000:1000000], df["sample"][998000:1000000])
        plt.ylim([-2000, 2000])
        # Plot threshold line
        plt.axhline(spike_threshold, color="r")
        # Plot all spikes
        plt.scatter(spike_value_df["Timestamp"][spikes_x_range[0]:spikes_x_range[1]], spike_value_df["sample"][spikes_x_range[0]:spikes_x_range[1]], c="r", s=10)
        # Customize lay-out
        plt.title("Spike threshold = {:.2f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

        plt.close()
        # Plot ECG values with threshold line
        plt.plot(resampled_df["Timestamp"], resampled_df["sample"])
        # Plot threshold line
        # plt.axhline(spike_threshold, color="r")
        # Plot all spikes
        plt.scatter(spike_value_df["Timestamp"], spike_value_df["sample"], c="r", s=10)
        # Customize lay-out
        # plt.title("ECG measurement \nfrom {0} \n  till {1}\nSpike threshold = {2:.0f}".format(start_timestamp, end_timestamp, spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("ECG values")
        plt.show()

        # ### Calculate InterBeat Interval

    # # Calculate time differences between all spikes (again, but only between real spikes now)
    # spike_value_df["time_differences"] = spike_value_df['Timestamp'].diff().dt.total_seconds() * 1000  # Difference in milliseconds
    # spike_value_df["time_differences"][:10]

    # # Fill first value of time difference with 1000 instead of NaN, because there is no previous value (This detects the first spike)
    # first_index = spike_value_df.index[0]
    # spike_value_df.loc[first_index, "time_differences"] = 1000
    # spike_value_df[:5]

    # # ### Calculate heart rate

    # spike_value_df["heartrate"] = 60 / spike_value_df["time_differences"] * 1000
    # spike_value_df

    if show_plots:
        plt.close()
        # Plot the ECG values of a whole night
        plt.plot(spike_value_df["Timestamp"], spike_value_df["heartrate"])
        plt.title("Calculated time from R - R from ECG data")
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("Heartrate R - R time difference")
        plt.show()

    # ---
    # 
    # ## Scale data - 2 / 2
    # 
    # ### Scale ECG to range [-0.5, 0.5]
    # 
    # First measure the minimum and maximum values of every heartbeat. (So not of all ECG data, but just of the heartbeats).<br>
    # The data should be scaled, such that the 90th percentile (or greater) of the minimum and maximum heartbeat values lies within the range [-0.5, 0.5].
    # 
    # Movement artifacts and other noise may exceed the amplitude of most heartbeats. <br>
    # Noisy data values may lie within the range of [-1.0, -0.5] and [0.5, 1.0]. 

    # Only get all heartbeat values (spikes)
    spike_values = spike_value_df["sample"]
    spike_values

    # Calculate the 90th percentile of min/max heartbeat values
    min_value = np.percentile(spike_values, 10)
    max_value = np.percentile(spike_values, 90)
    print("The 90th percentile of the minimum heartbeat values:", min_value)
    print("The 90th percentile of the maximum heartbeat values:", max_value)

    # Scale factor based on the biggest, most absolut value
    scale_factor = 0.5 / max(abs(min_value), abs(max_value))
    print("Scale factor:", scale_factor)

    # Scale using the 90th percentile of min/max heartbeat values, without an adjustment
    scaled_ecg = centered_ecg * scale_factor
    resampled_df["sample"] = scaled_ecg
    resampled_df

    print("Is the median still equal to 0?")
    print("Median:", np.abs(np.median(scaled_ecg)))

    title = "Scaled ECG data to scale [-0.5, 0.5]"
    if show_plots:
        plot_data(resampled_df, scaled_ecg, title, x_range=x_range, y_range=[-1, 1])

    if show_plots:
        plot_data(resampled_df, scaled_ecg, title)

    # ### Clamp outliers
    # 
    # All noisy datapoints and even the "tall" heartbeats should lie between the range of [-1.0, 1.0].<br>
    # Outliers should all be clamped to [-1.0, 1.0].

    # Clamp values to [-1.0, 1.0] to handle noise and tall heartbeats
    clamped_ecg = np.clip(scaled_ecg, -1.0, 1.0)
    clamped_ecg

    print("Is the median still equal to 0?")
    print("Median:", np.abs(np.median(clamped_ecg)))

    title = "Clamped outliers between [-1.0, 1.0]"
    if show_plots:
        plot_data(resampled_df, clamped_ecg, title, x_range=x_range, y_range=[-1, 1])

    if show_plots:
        plot_data(resampled_df, clamped_ecg, title)

    # <br><br>
    # 
    # ---
    # 
    # ## Reshape into 30-second epochs
    # 
    # In the last preprocessing step, the ECG values should be divided into 30-second epochs. <br>
    # These 30-second epochs should be added to a new dataset as individual rows. <br>
    # The shape of the dataset will be [epoch_count * 7680]. The 7680 columns originate from 30 seconds * 256 Hertz. 

    # Calculate the new 2D array shape
    epoch_length = 30 * new_hertz # 30 seconds * 256 Hertz = 7680 datapoints per epoch
    epoch_count = len(clamped_ecg) // epoch_length
    print("Dataset shape: ({}, {})".format(epoch_count, epoch_length))

    # Trim ECG data down to the next nearest 30 second epoch length
    trimmed_ecg = clamped_ecg[:epoch_count * epoch_length] # ECG length is now a multiple of 30 seconds
    print("Amount of measurements taken into account:", len(trimmed_ecg))

    # Reshape the ECG data into 2D array
    ecgs = clamped_ecg[:epoch_count * epoch_length].reshape((epoch_count, epoch_length))
    pd.DataFrame(ecgs)

    # <br><br>
    # 
    # ---
    # 
    # ## Convert array to HDF5 file
    # 
    # Convert the 2D Numpy array with all the preprocessed ECG values of a single night of sleep to a HDF5 file format, which is required by the feed-forward neural network to predict a sleep score.
    # 
    # Structure of the HDF5 dataset file:
    # - `ecgs`:
    #   - 2D array of floats (size: epoch_count x 7680)
    #   - Where 7680 = 30 x 256Hz.
    #   - Network was trained on raw ECG data that had been filtered and scaled appropriately. See **Data preprocessing** above.
    # - `demographics`:
    #   - 2D array of floats (size: 2 x 1):
    #     - 1st: sex (0=female, 1=male)
    #     - 2nd: age (age_in_years/100)
    # - `midnight_offset`:
    #   - A float that represents the clock time offset to the nearest midnight of when the recording began:
    #     - 0 = midnight
    #     - -1 = 24hr before midnight and 1 = 24hr after midnight
    #     - For example, 9pm = -0.125 and 3am = 0.125.
    # - `stages` (only required for training):
    #   - 2D array of floats (size: epoch_count x 1):
    #     - Stage mapping: 0=Wake, 1=N1/S1, 2=N2/S2, 3=N3/S3/S4, 4=REM.
    #       - It is not uncommon to find REM mapped to 5. However, the network was trained with data with both AASM and R&K scorings, so a unified "deep sleep" score was mapped to 3. And because it's inconvenient to have a gap, REM was remapped to 4.
    #     - All "unscored" epochs should be mapped to 0 (also see weight below).
    # - `weights` (only required for training):
    #   - 2D array of floats (size: epoch_count x 1):
    #     - 0 (no weight) to 1 (full weight)
    #     - All "unscored" epochs should be given a weight of 0.

    # Check if 'ecgs' is in correct format
    print("ECG size should be: \t(epoch_count x 7680)")
    print(f"ECG size is: \t\t{ecgs.shape}")

    print("\nECG type should be a: \t2D numpy array of floats")
    print(f"ECG types are: \t\t{type(ecgs)} \n\t\t\t{type(ecgs[1][1])}")

    # Calculate demographics
    def get_demographics(gender, age):
        # Convert gender in binary number
        if gender.lower() == "Male" or "M" or "Man":
            sex = 1
        elif gender.lower() == "Female" or "F" or "Woman":
            sex = 0
        else:
            return "Gender is not entered correctly. Please insert Male or Female"
        
        # Convert age in Floating number
        age = age/100
        
        # Return calculated demographics
        return np.array([sex, age])
    
    # Get demographics
    gender = "Male"
    age = 40
    demographics = get_demographics(gender, age)

    # Check if 'demographics' is in correct format
    print("Demographics size should be: \t(2 x 1)")
    print(f"Demographics size is: \t\t{demographics.shape}")

    print("\nDemographics should be contain a 2D array of floats representing:")
    print(" - Sex: Binary digit (0=female or 1=male)")
    print(" - Age: Floating number (age/100)")

    print("\nDemographics variable contains:")
    print(" - Sex:", demographics[0])
    print(" - Age:", demographics[1])

    from tabulate import tabulate

    table = [[gender, age]]
    headers = ["Gender", "Age"]
    print(tabulate(table, headers, tablefmt="pretty"))

    # #### Midnight offset

    # Bereken de offset ten opzichte van middernacht
    def get_midnight_offset(start_timestamp):
        # Verkrijg de tijd van de start in het gewenste formaat (HH:MM:SS)
        start_time = start_timestamp.strftime("%H:%M:%S")
        print(f"Starttijd: {start_time}")

        # Verkrijg de uren, minuten en seconden uit de start_timestamp
        hours = int(start_timestamp.strftime("%H"))
        minutes = int(start_timestamp.strftime("%M"))
        seconds = int(start_timestamp.strftime("%S"))
        
        # Bereken de kloktijd offset ten opzichte van middernacht
        offset = ((seconds / 60 + minutes) / 60 + hours) / 24

        # Vergelijk de starttijd met 12:00:00 (omgezet naar een datetime.time object)
        if start_timestamp.time() > time(12, 0, 0):  # Gebruik datetime.time(12, 0, 0) om de tijd te vergelijken
            offset = -1 + offset

        return np.array([offset])

    # Voorbeeld hoe je de functie kunt gebruiken:
    start_timestamp = f.getStartdatetime()  # Verkrijg de starttijd van je EDF-bestand
    midnight_offset = get_midnight_offset(start_timestamp)  # Bereken de offset
    print("Middernacht offset:", midnight_offset)

    datetime.datetime.now()

    # Get midnight offset
    # start_date_time = datetime.datetime(2024, 11, 19, 12, 0, 1) # test
    start_date_time = datetime.datetime.now()
    midnight_offset = get_midnight_offset(start_timestamp)

    # Check if 'midnight_offset' is in correct format
    print("Midnight offset should contain a float between the range [-1, 1] representing the clocktime offset of when the recording began.")
    print("Midnight offset value:", midnight_offset[0])

    # ## Create HDF5 file
    # import h5py
    # import os

    # Folder for ECG data
    # Haal bestandsnaam uit het pad
    filename = os.path.basename(edf_file)  # bijvoorbeeld 'shhs1-200001.edf'
    name_without_ext = os.path.splitext(filename)[0]  # 'shhs1-200001'
    dataset, participant_id = name_without_ext.split('-')

    # Definieer en maak hoofdfolder aan als die nog niet bestaat
    results_base_path = r"C:/shhs/Results_validatie"
    os.makedirs(results_base_path, exist_ok=True)

    # Maak submapnaam aan
    result_folder_name = f"Results_{dataset}_{participant_id}"
    full_result_path = os.path.join(results_base_path, result_folder_name)

    # Maak submap aan
    os.makedirs(full_result_path, exist_ok=True)

    # Stel de bestandsnaam samen
    result_filename = f"preprocessed_ecg_data_{dataset}_{participant_id}.h5"
    result_file_path = os.path.join(full_result_path, result_filename)
    print("result_file_path:", result_file_path)

    # Print pad als check
    print(f"Resultaat wordt opgeslagen in: {result_file_path}")

    # ## Read HDF5 file
    # Create or overwrite HDF5 file
    with h5py.File(result_file_path, 'w') as hdf5_file:
        ecgs_data = hdf5_file.create_dataset("ecgs", data=ecgs)
        demographics_data = hdf5_file.create_dataset("demographics", data=demographics)
        midnight_offset_data = hdf5_file.create_dataset("midnight_offset", data=midnight_offset)

    # Open HDF5 file
    with h5py.File(result_file_path, 'r') as hdf5_file:
        ecgs_data = hdf5_file["ecgs"]
        demographics_data = hdf5_file["demographics"]
        midnight_offset_data = hdf5_file["midnight_offset"]
        print(ecgs_data[()])
        print(demographics_data[()])
        print(midnight_offset_data[()])

        print(hdf5_file)

    
    return result_file_path, df_ecg

    # Folder for ECG data VANAF HIER NIEUW
    # Folder for ECG data
    # results_base_path = r"C:/shhs/Results_validatie"
    # os.makedirs(results_base_path, exist_ok=True)

    # # Maak submapnaam aan
    # result_folder_name = f"Results_{dataset}_{participant_id}"
    # full_result_path = os.path.join(results_base_path, result_folder_name)

    # # Verwijder de bestaande map als die er is, en maak hem opnieuw aan
    # if os.path.exists(full_result_path):
    #     print(f"Verwijder bestaande map: {full_result_path}")
    #     # Verwijder alle bestanden en submappen in de map
    #     for filename in os.listdir(full_result_path):
    #         file_path = os.path.join(full_result_path, filename)
    #         try:
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)  # Verwijder bestand
    #             elif os.path.isdir(file_path):
    #                 shutil.rmtree(file_path)  # Verwijder submap
    #         except Exception as e:
    #             print(f"Kon bestand {file_path} niet verwijderen: {e}")
    # else:
    #     print(f"Maak nieuwe map aan: {full_result_path}")

    # os.makedirs(full_result_path, exist_ok=True)

    # # Controleer of de map echt bestaat
    # if not os.path.isdir(full_result_path):
    #     print(f"Fout: de map {full_result_path} is niet aangemaakt of niet toegankelijk.")
    # else:
    #     print(f"Map bestaat: {full_result_path}")

    # # Stel de bestandsnaam samen
    # result_filename = f"preprocessed_ecg_data_{dataset}_{participant_id}.h5"
    # result_file_path = os.path.join(full_result_path, result_filename)
    # print("result_file_path:", result_file_path)

    # # Print pad als check
    # print(f"Resultaat wordt opgeslagen in: {result_file_path}")

    # # Create or overwrite HDF5 file
    # try:
    #     with h5py.File(result_file_path, 'w') as hdf5_file:
    #         ecgs_data = hdf5_file.create_dataset("ecgs", data=ecgs)
    #         demographics_data = hdf5_file.create_dataset("demographics", data=demographics)
    #         midnight_offset_data = hdf5_file.create_dataset("midnight_offset", data=midnight_offset)

    #     print(f"Bestand succesvol opgeslagen op: {result_file_path}")
    # except Exception as e:
    #     print(f"Fout bij opslaan van HDF5 bestand: {e}")
    #     result_file_path = None

    # # Open HDF5 file en lees de data
    # if result_file_path is not None:
    #     try:
    #         with h5py.File(result_file_path, 'r') as hdf5_file:
    #             ecgs_data = hdf5_file["ecgs"]
    #             demographics_data = hdf5_file["demographics"]
    #             midnight_offset_data = hdf5_file["midnight_offset"]
    #             print("ECG Data:", ecgs_data[()])
    #             print("Demographics Data:", demographics_data[()])
    #             print("Midnight Offset Data:", midnight_offset_data[()])
    #             print(hdf5_file)
    #     except Exception as e:
    #         print(f"Fout bij openen van HDF5 bestand: {e}")
    # else:
    #     print("HDF5 bestand is niet aangemaakt, geen gegevens om te lezen.")

    # # Return het pad van het bestand en andere gegevens als nodig
    # return result_file_path, df_ecg

    
