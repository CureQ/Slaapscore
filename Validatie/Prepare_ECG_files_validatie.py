# Bestandsnaam: Prepare_ECG_files_validatie.py
# Naam: Esmee Springer
# Voor het laatst bewerkt op: 03-06-2025

# Importeren van benodigde pakkages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    # De preprocessing van de ECG data is nodig om de data geschikt te maken als input voor het neurale netwerk
    # 

    # Open het bestand
    f = pyedflib.EdfReader(edf_file)

    # Haal het aantal signalen, de signaallabels en de samplefrequenties op
    n_signals = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sampling_rates = f.getSampleFrequencies()

    # Maak een lege lijst om de eerste 10 waarden van elk signaal in op te slaan
    data = []

    # Loop door elk signaal en sla de eerste 10 waarden op
    for i in range(n_signals):
        signal_data = f.readSignal(i)  # Lees het volledige signaal
        data.append([signal_labels[i], sampling_rates[i]] + list(signal_data[:10]))

    # Sluit het EDF-bestand
    f.close()

    # Zet de data in een DataFrame
    columns = ["Signaalnaam", "Samplefrequentie"] + [f"Punt {i+1}" for i in range(10)]
    df = pd.DataFrame(data, columns=columns)

    # Print het DataFrame met de verzamelde signaalinformatie
    print(df)

    # Open het EDF-bestand om specifieke ECG-gegevens te verwerken
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


    # ### Visualiseer ruwe data

    if show_plots:
        plt.close()
        # Plot het volledige ECG-signaal van de nacht
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
    # ### Verzamelen van metadata (belangrijke gegevens) over de metingen
    
    # Haal de bestandsnaam uit het pad
    filename = os.path.basename(edf_file)  # 'shhs1-200001.edf'

    # Verwijder extensie en splits op '-'
    name_without_ext = os.path.splitext(filename)[0]  # 'shhs1-200001'
    dataset, participant_id = name_without_ext.split('-')

    # Print de datasetnaam en het participant ID
    print("Dataset:", dataset)           # 'shhs1'
    print("Participant ID:", participant_id)  # '200001'

    # Haal de ECG-waarden uit je dataframe en zet ze om naar een numpy array
    raw_ecg_samples = df_ecg["sample"].to_numpy()  

    # Haal de startdatum en starttijd op (datetime-object)
    start_timestamp = f.getStartdatetime()  

    # Verkrijg de startdatum en starttijd 
    start_date = start_timestamp.date()  # Haalt alleen de datum op
    start_time = start_timestamp.time()  # Haalt alleen de tijd op

    # Print de startdatum en starttijd apart
    print(f"Measurement start date: {start_date}")
    print(f"Measurement start time: {start_time}")
    print(f"Start date and start time: {start_timestamp}")

    # Haal de eerste en laatste timestamp uit het ECG DataFrame
    first_timestamp = df_ecg["timestamp"].iloc[0]
    last_timestamp = df_ecg["timestamp"].iloc[-1]

    # Bereken de duur in milliseconden (tijd is in seconden, dus vermenigvuldig met 1000)
    duration_milliseconds = (last_timestamp - first_timestamp)*1000 
    print("Measurement took {0} milliseconds.".format(duration_milliseconds))

    # Bereken de duur in seconden
    duration_seconds = int(duration_milliseconds / 1000)

    # Bereken het aantal uren, minuten en seconden
    minutes, seconds = divmod(duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # print de meetduur
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

# Voeg extra kolom met timestamps toe aan de ECG data
    def add_timestamps(df_ecg, start_timestamp, end_timestamp):
        # Print de starttijd en eindtijd van de meting
        print("Starting timestamp: ", start_timestamp)
        print("Ending timestamp:", end_timestamp, "\n\n")
        
        # Bereken het totaal aantal minuten tussen begin- en eindtijd
        total_minutes = (end_timestamp - start_timestamp).total_seconds() / 60

        # Bereken hoeveel metingen er per minuut zijn gedaan
        measurements_per_minute = len(df_ecg) / total_minutes
        
        # Genereer voor elke rij in het ECG-DataFrame een timestamp
        df_ecg["Timestamp"] = [start_timestamp + timedelta(minutes=i/measurements_per_minute) for i in range(len(df_ecg))]
        # Geef het DataFrame terug
        return df_ecg
    
    # Voeg timestamps toe aan elk meetpunt
    df_ecg = add_timestamps(df_ecg, start_timestamp, end_timestamp)
    df_ecg

    # Visualiseer ECG data 
    # Functie om ECG-data te plotten
    def plot_data(df_ecg, ecg_samples=np.array([]), title="Raw data ECG measurement", y_range=[0,0], x_range=[0,0]):
        # Sluit eerder geopende plots
        plt.close()

        # Als er een aparte lijst met ECG-samples wordt meegegeven, gebruik die om te plotten
        if ecg_samples.any():
            ecg_samples = pd.Series(ecg_samples)
            plt.plot(df_ecg["Timestamp"], ecg_samples)
        else:
            # Zo niet, gebruik de ECG-kolom uit het DataFrame
            plt.plot(df_ecg["Timestamp"], df_ecg["sample"])

        # Stel y-as in
        if y_range != [0,0]:
            plt.ylim(y_range[0], y_range[1])
        # Stel x-as in
        if x_range != [0,0]:
            plt.xlim(x_range[0], x_range[1])

        # Stel de grafiektitel en labels in
        plt.title(title)
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    # Als show_plots=True, toon de plot
    if show_plots:
        plot_data(df_ecg)

        # <br><br>
    # 
    # ---
    # 
    # ## Filter ruis
    # 
    # ### High pass filter
    # 
    # Pas een high-pass filter toe om baseline wander te verwijderen

    # Importeer signaalverwerkingsfunctie voor het filteren van de data
    from scipy.signal import butter, filtfilt, iirnotch, resample

    # Functie om een high-pass filter toe te passen
    def highpass_filter(data, cutoff=0.5, fs=hertz, order=4):
        # Bepaal de Nyquist-frequentie, helft van de samplefrequentie
        nyquist = 0.5 * fs
        # Normaliseer de cutoff frequentie
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        # Pas het filter toe
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    # Pas het high-pass filter toe op de ruwe ECG-samples
    highpass_filtered_ecg = highpass_filter(raw_ecg_samples)
    highpass_filtered_ecg

    # ### Verwijder lijnruis
    # 
    # Lijnruis (50/60 Hertz) en andere frequentieruis worden verwijderd met het notch-filter.

    # Notch-filter functei om netspanningsruis te verwijderen (bijv., 50/60 Hz)
    def notch_filter(data, freq=50, fs=hertz, quality_factor=30):
        # Bereken Nyquist-frequentie
        nyquist = 0.5 * fs
        # Normaliseer de frequentie ten opzichte van Nyquist-frequentie
        freq = freq / nyquist
        b, a = iirnotch(freq, quality_factor)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    # Pas het 50 Hz notch filter toe om netspanningsruis te verwijderen
    filtered_ecg = notch_filter(highpass_filtered_ecg, freq=50, fs=hertz)  # 50 Hz filter
    filtered_ecg

    # Pas indien nodig ook het 60 Hz filter toe
    filtered_ecg = notch_filter(filtered_ecg, freq=60, fs=hertz)  # 60 Hz filter
    filtered_ecg

    df_ecg["Timestamp"][0]

    # Definieer een specifiek tijdsinterval voor visualisatie
    x_range = [df_ecg["Timestamp"][998000], df_ecg["Timestamp"][1000000]]
    y_range = [-1, 1]

    # Visualiseer de gefilterde ECG data in het gekozen tijdsinterval
    if show_plots:
        plot_data(df_ecg, y_range=y_range, x_range=x_range)

    title = "Filtered ECG data on noise"
    # Visualiseer de volledig gefilterde ECG data
    if show_plots:
        plot_data(df_ecg, filtered_ecg, title)

    # <br><br>
    # 
    # ---
    # 
    # ## Resample de data naar 256 Hertz
    # 
    # Resample de data van de oorspronkelijke gemeten 125 Hz naar 256 Hz
    # 
    # Dit is nodig omdat het neurale netwerk getraind is met ECG-datasets op 256 Hz
    # 
    # #### Sample ECG data

    new_hertz = 256 # Nieuwe samplefrequentie in Hertz
    # Resample het ECG-signaal naar 256 Hz
    resampled_ecg = resample(filtered_ecg, int(len(filtered_ecg) * (new_hertz / hertz)))
    resampled_ecg

    # #### Resample tijdas

    # Haal de originele timestamps op in seconden
    original_timestamps = df_ecg["timestamp"].to_numpy() 
    original_timestamps

    # Bereken originele tijdstappen in seconden
    time_original = np.arange(len(original_timestamps)) / hertz  
    time_resampled = np.linspace(0, time_original[-1], len(resampled_ecg))  

    # Toon originele en nieuwe tijdstippen
    print("Original timesteps: ", time_original[:5], "...", time_original[-5:])
    print("Resampled timesteps:", time_resampled[:5], "...", time_resampled[-5:])

    # Interpoleer originele tijdstempels naar nieuwe tijdas
    resampled_timestamps = np.interp(time_resampled, time_original, original_timestamps)

    # Toon eerste en laate tijdstempels
    print("Original timestamps: ", original_timestamps[:5], "...", original_timestamps[-5:])
    print("Resampled timestamps:", resampled_timestamps[:5], "...", resampled_timestamps[-5:])

    # Maak een nieuw DataFrame aan met de resamplde data
    resampled_df = pd.DataFrame({
        'timestamp': resampled_timestamps,
        'sample': resampled_ecg
    })

    # Voeg extra timestamp-kolom toe voor betere visualisatie
    resampled_df = add_timestamps(resampled_df, start_timestamp, end_timestamp)
    resampled_df

    # Titel voor visualisatie
    title = f"Resampled ECG data from {hertz} Hz to {new_hertz} Hz"
    # Plot de resamplde ECG-gegevens als show_plots=True
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

    # Bereken de mediaan van de resamplde ECG-data
    median_ecg = np.median(resampled_ecg)
    median_ecg

    # Haal de mediaan ervan af om de data rond 0 te centreren
    centered_ecg = resampled_ecg - median_ecg

    # Controleer of de mediaan (bijna) nul is
    np.abs(np.median(centered_ecg))

    # Vervang de originele waarden in het DataFrame door de gecentreerde ECG-waarden
    resampled_df["sample"] = centered_ecg
    resampled_df

    # Plot de gecentreerde ECG-data als show_plots=True
    title = "Median of ECG data is now equal to 0"
    if show_plots:
        plot_data(resampled_df, centered_ecg, title, y_range=y_range, x_range=x_range)

    # ---
    # 
    # ## Spikes & Hartslag
    # 
    # ### Bepaal de thresholdwaarde voor hartslagpieken

    # Bepaal de thresholdwaarde voor het detecteren van pieken met behulp van de standaardafwijking
    spike_threshold = np.std(resampled_df["sample"])
    spike_threshold

    # Visualiseer de ECG-waarden met de thresholdlijn
    if show_plots:
        plt.close()
        # Plot ECG waarden met thresholdlijn
        plt.plot(df_ecg["Timestamp"][998000:1000000], df_ecg["sample"][998000:1000000])
        plt.ylim([-1, 1])
        # Plot thresholdlijn
        plt.axhline(spike_threshold, color="r")
        plt.title("Spike threshold = {:.2f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    # Gebruik find_peaks om pieken te detecteren
    peaks, properties = find_peaks(df_ecg["sample"], height=spike_threshold, distance=fs*0.6)

    # Visualiseer de gedetecteerde pieken
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

    # Print het aantal gedetecteerde pieken
    print(f"Aantal gedetecteerde pieken: {len(peaks)}")

    # Haal de tijdstempels van de gedetecteerde pieken
    peak_timestamps = df_ecg["Timestamp"].iloc[peaks]

    # Bereken de tijdsverschillen tussen de pieken (RR-intervals) in seconden
    time_differences = peak_timestamps.diff().dt.total_seconds()

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
        # Plot een deel van het ECG-signaal met de thresholdlijn en gedetecteerde pieken
        plt.plot(resampled_df["Timestamp"][resampled_x_range[0]:resampled_x_range[1]], resampled_df["sample"][resampled_x_range[0]:resampled_x_range[1]])
        # Stel y-as limieten in
        plt.ylim([-2000, 2000])
        # Plot de thresholdlijn
        plt.axhline(spike_threshold, color="r")
        # Plot alle spikes
        plt.scatter(spike_value_df["Timestamp"][spikes_x_range[0]:spikes_x_range[1]], spike_value_df["sample"][spikes_x_range[0]:spikes_x_range[1]], c="r", s=10)
        # Lay-out
        plt.title("Spike threshold = {:.2f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

        plt.close()
        # Plot het volledig resamplede ECG-signaal met alle gedetecteerde pieken
        plt.plot(resampled_df["Timestamp"], resampled_df["sample"])
        
        plt.scatter(spike_value_df["Timestamp"], spike_value_df["sample"], c="r", s=10)
        
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("ECG values")
        plt.show()

        # ### Bereken InterBeat Interval (tijd tussen hartslagen)

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
    # Meet eerst de minimale en maximale waarden van elke hartslag (dus niet van alle ECG-data, alleen de hartslagen).<br>
    # De data moet zo geschaald worden dat het 90e percentiel (of hoger) van de minimale en maximale hasrtalgwaarden binnen het beriek [-0.5, 0.5] valt.
    # 
    # Beweging en andere ruis kunnen de amplitude van de meeste hartslagen overschrijden. <br>
    # Ruis kan binnen het bereik van [-1.0, -0.5] en [0.5, 1.0] vallen. 

    # Haal alleen alle hartslagwaarden op (pieken)
    spike_values = spike_value_df["sample"]
    spike_values

    # Bereken het 90e percentiel van de minimale en maximale hartslagwaarden
    min_value = np.percentile(spike_values, 10)
    max_value = np.percentile(spike_values, 90)
    print("The 90th percentile of the minimum heartbeat values:", min_value)
    print("The 90th percentile of the maximum heartbeat values:", max_value)

    # Bepaal de schaalfactor op basis van de grootste absolute waarde
    scale_factor = 0.5 / max(abs(min_value), abs(max_value))
    print("Scale factor:", scale_factor)

    # Schaal de ECG-waarden met de schaalfactor, zonder correctie
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
    # Alle datapunten met ruis en zelfs de 'hoge' hartslagen moeten binnen het bereik [-1.0, 1.0] vallen.<br>
    # Uitschieter moeten allemaal worden begrensd [-1.0, 1.0].

    # Begrens waarden tot [-1.0, 1.0] om ruis en hoge hartslagen te verwerken
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
    # ## Reshape naar 30-seconde epochs
    # 
    # In de laatste preprocessing stap moeten de ECG-waarden worden opgedeeld in 30 seconde epochs <br>
    # Deze 30 seconde epochs worden toegevoegd aan een nieuwe dataset als afzonderlijke rijen <br>
    # De vorm van de dataset wordt dan [epoch_count * 7680]. De 7680 kolommen ontstaan uit 30 seconds * 256 Hertz. 

    # Bereken de nieuwe vorm van de 2D-array
    epoch_length = 30 * new_hertz # 30 seconde * 256 Hertz = 7680 datapunten per epoch
    epoch_count = len(clamped_ecg) // epoch_length
    print("Dataset shape: ({}, {})".format(epoch_count, epoch_length))

    # Knip ECG-data af naar het dichtsbijzijnde 30 seconde epoch
    trimmed_ecg = clamped_ecg[:epoch_count * epoch_length] 
    print("Amount of measurements taken into account:", len(trimmed_ecg))

    # Vorm de ECG-data om naar een 2D array
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

    # Controleer of 'ecgs' het juiste formaat heeft
    print("ECG size should be: \t(epoch_count x 7680)")
    print(f"ECG size is: \t\t{ecgs.shape}")

    print("\nECG type should be a: \t2D numpy array of floats")
    print(f"ECG types are: \t\t{type(ecgs)} \n\t\t\t{type(ecgs[1][1])}")

    # #### Demografische gegevens

    # Bereken demografische gegevens
    def get_demographics(gender, age):
        # Zet geslacht om naar een binair getal
        if gender.lower() == "Male" or "M" or "Man":
            sex = 1
        elif gender.lower() == "Female" or "F" or "Woman":
            sex = 0
        else:
            return "Gender is not entered correctly. Please insert Male or Female"
        
        # Zet leeftijd om naar een float (komma getal)
        age = age/100
        
        # Geef berekende demografische gegevens terug
        return np.array([sex, age])
    
    # Haal demografische gegevens op
    gender = "Male"
    age = 40
    demographics = get_demographics(gender, age)

    # Controleer of 'demographics' het juiste formaat heeft
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

    # #### Midnight offset berekenen

    # Functie om midnight-offset te berekenen
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
        if start_timestamp.time() > time(12, 0, 0):  
            offset = -1 + offset

        return np.array([offset])

    # Voorbeeld hoe je de functie kunt gebruiken:
    start_timestamp = f.getStartdatetime()  # Verkrijg de starttijd van je EDF-bestand
    midnight_offset = get_midnight_offset(start_timestamp)  # Bereken de offset
    print("Middernacht offset:", midnight_offset)

    # Huidige datum en tijd ophalen
    datetime.datetime.now()

    # Verkrijg midnight offset

    start_date_time = datetime.datetime.now()
    # Bereken de midnight offset op basis van de starttijd van de opname
    midnight_offset = get_midnight_offset(start_timestamp)

    # Controleer of 'midnight_offset' het juiste formaat heeft
    print("Midnight offset should contain a float between the range [-1, 1] representing the clocktime offset of when the recording began.")
    print("Midnight offset value:", midnight_offset[0])

    # ## Create HDF5 file
    
    # Folder voor ECG data
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
    # Maak het HDF5 bestand aan of overschrijf wanneer deze al bestaat
    with h5py.File(result_file_path, 'w') as hdf5_file:
        # Voeg de ECG data toe aan het bestand
        ecgs_data = hdf5_file.create_dataset("ecgs", data=ecgs)
        # Voeg demografische data toe aan het bestand
        demographics_data = hdf5_file.create_dataset("demographics", data=demographics)
        # Voeg midnight offset toe aan het bestand
        midnight_offset_data = hdf5_file.create_dataset("midnight_offset", data=midnight_offset)

    # ## Read HDF5 file

    # Open HDF5 bestand in read modus
    with h5py.File(result_file_path, 'r') as hdf5_file:
        # Lees de ECG-data uit het bestand
        ecgs_data = hdf5_file["ecgs"]
        # Lees de demografische data uit het bestand
        demographics_data = hdf5_file["demographics"]
        # Lees de midnight offset uit het bestand
        midnight_offset_data = hdf5_file["midnight_offset"]
        # Print de inhoud
        print(ecgs_data[()])
        print(demographics_data[()])
        print(midnight_offset_data[()])

        print(hdf5_file)

    # Geef result_file_path en df_ecg terug
    return result_file_path, df_ecg

