# Bestandsnaam: Prepare_ECG_files_gui.py
# Geschreven door: Esmee Springer
# Voor het laatst bewerkt op: 23-05-2025

# Importeren van benodigde pakkages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import find_peaks
import re
import os
import h5py


def extract_start_timestamp(ecg_file):
    # Loop door alle bestanden in de opgegeven ECG map
    for file in os.listdir(ecg_file):
        print("Bestand gevonden:", file) # Print de naam van elk gevonden bestand
        # Controleer of de bestandsnaam voldoet aan het verwachte patroon
        match = re.match(r"(\d{8}T\d{6}Z)_\w+_ecg_stream\.csv", file)
        if match:
            print("Bestandsnaam matched:", file)  

            # Extraheer de datum en tijd uit de bestandsnaam
            date_time_str = match.group(1)  
            date_str, time_str = date_time_str.split("T")
            time_str = time_str.replace("Z", "")
            try:
                # Zet de datum en tijd om naar een datetime object
                start_timestamp = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
                return start_timestamp
            except ValueError:
                # Bij een error, ga door naar het volgende bestand
                pass
    # Als er geen geldige timestamp gevonden is, geef een foutmelding
    raise ValueError("Start timestamp kon niet worden geëxtraheerd uit bestandsnamen in deze map.")



def get_hdf5_file(ecg_file, file_counter, show_plots=False):
    # # Cardiosomnography data preprocessing
    # 
    # De preprocessing van de ECG data is nodig om de data geschikt te maken als input voor het neurale netwerk
    # 
 
    # Lees het CSV-bestand met ECG-data in als DataFrame
    df = pd.read_csv(ecg_file)
    # Zet de kolom 'sample' om naar een array
    raw_ecg_samples = df["sample"].to_numpy()
    print("Raw ECG samples:", raw_ecg_samples)
    hertz = 125
    print("Amount of measured ECG values:", len(raw_ecg_samples))
    print("\nFirst and last measured ECG values:\n", raw_ecg_samples)
    print("Df:",df)

    # ### Visualiseer de ruwe ECG signalen

    if show_plots:
        plt.close()
        # Plot the ECG values of a whole night
        plt.plot(df["timestamp"], df["sample"])
        plt.title("Raw data ECG measurement")
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("ECG values")
        plt.show()

    # ##### Wat te doen met alle ECG-waarden boven de 2500? Verwijderen, clampen of iets anders? 

    # <br><br>
    # 
    # ---
    # 
 
    try:
        # Pak de bovenliggende mapnaam van het .csv-bestand
        folder_path = os.path.dirname(ecg_file)
        folder_name = os.path.basename(folder_path)

        # Probeer het participantnummer te extraheren uit de mapnaam
        match = re.search(r"participant[_\-]?(\d+)", folder_name, re.IGNORECASE)
        if match:
            # Zet het participantnummer om naar een integer 
            participant_number = int(match.group(1))
            print("Measurement of participant {0}".format(participant_number))
        else:
            # Als er geen match is, geef een foutmelding
            raise ValueError
    except:
        # Geef een foutmelding als het participantnummer niet geëxtraheerd kan worden
        print("Participant number kon niet worden geëxtraheerd uit de mapnaam.")
        participant_number = "onbekend"

    # Starttimestamp halen uit bestandsnaam in de geselecteerde map
    try:
        folder_path = os.path.dirname(ecg_file)  # pad naar de map waarin het csv-besatnd zich bevindt
        start_timestamp = extract_start_timestamp(folder_path) # starttijdstip extraheren uit bestandsnaam
        print("Start timestamp:", start_timestamp)
    except ValueError as e:
        # Geef een foutmelding als het niet lukt om de starttijd te extraheren
        print(e)
        return None


    # #### Bepaal de duur van de meting

    # Haal de eerste en laatste timestamp op uit het DataFrame
    first_timestamp = df["timestamp"].iloc[0]
    last_timestamp = df["timestamp"].iloc[-1]
    
    # Bereken de duur van de meting in milliseconden
    duration_milliseconds = last_timestamp - first_timestamp
    
    # Print de duur van de meting
    print("Measurement took {0} milliseconds.".format(duration_milliseconds))

    # Bepaal de duur van de meting in seconden
    duration_seconds = int(duration_milliseconds / 1000)
    # Bereken het aantal minuten en seconden van de duur
    minutes, seconds = divmod(duration_seconds, 60)
    # Bereken het aantal uren en minuten van de duur
    hours, minutes = divmod(minutes, 60)

    # Print de duur van de meting in seconden 
    print("The measurement duration took {0} seconds.".format(duration_seconds))
    # Print de duur van de meting in uren, minuten en seconden
    print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

    # #### Bepaal het eindtijdstip van de meting

    # Voeg de duur van de meting toe aan het starttijdstip om het eindtijdstip te krijgen
    duration_timestamp = datetime.timedelta(seconds=duration_seconds)
    end_timestamp = start_timestamp + duration_timestamp
    # Print het starttijdstip van de meting
    print("Measurement started on {0}".format(start_timestamp))
    # Print het eindtijdstip van de meting
    print("Measurement  ended  on {0}".format(end_timestamp))

    # #### Bereken interval tussen metingen

    # Bereken de gemiddelde tijd tussen elke meting
    measurement_interval = duration_seconds / len(df)
    # Bereken de frquentie (Hertz)
    hertz = round(1/measurement_interval)

    # Print de gemiddelde tijd tussen metingen in seconden en milliseconden
    print("Average time interval between each measurement is: ")
    print(" - {0:.3f} seconds.".format(measurement_interval))
    print(" - {0:.3f} milliseconds.".format(measurement_interval*1000))

    # Print de samplefrequentie in Hertz
    print("\nSample rate: {0} Hertz.".format(hertz))

    # ### Voeg extra timestamps toe voor betere visualisatie

    # Functie om een extra kolom met timestamps toe te voegen aan het DataFrame
    def add_timestamps(df, start_timestamp, end_timestamp):
        # Print het starttimestamp en eindtimestamp van de meting
        print("Starting timestamp: ", start_timestamp)
        print("Ending timestamp:", end_timestamp, "\n\n")
        
        # Bereken het totaal aantal minuten tussen de starttijd en eindtijd
        total_minutes = (end_timestamp - start_timestamp).total_seconds() / 60
        # Bereken het gemiddeld aantal metingen per minuut
        measurements_per_minute = len(df) / total_minutes
        
        # Maak een lijst met timestamps verspreid over de meetduur
        df["Timestamp"] = [start_timestamp + datetime.timedelta(minutes=i/measurements_per_minute) for i in range(len(df))]
        return df

    # Voeg timestamps toe aan elke meting in het DataFrame
    df = add_timestamps(df, start_timestamp, end_timestamp)
    # Print het DataFrame
    print("df:",df)

    # ### Visualiseer ruwe data

    # Functie om ECG-data te visualiseren op elk gewenst moment
    def plot_data(df, ecg_samples=np.array([]), title="Raw data ECG measurement", y_range=[0,0], x_range=[0,0]):
        plt.close()
        # als er specifieke ECG-samples zijn meegegeven, plot deze dan
        if ecg_samples.any():
            ecg_samples = pd.Series(ecg_samples)
            plt.plot(df["Timestamp"], ecg_samples)
        else:
            # anders plot de standaard ECG-samples uit het DataFrame
            plt.plot(df["Timestamp"], df["sample"])

        # Stel de y-as range in
        if y_range != [0,0]:
            plt.ylim(y_range[0], y_range[1])
        # Stel de x-as range in
        if x_range != [0,0]:
            plt.xlim(x_range[0], x_range[1])

        # Voeg titels en labels toe aan de grafiek
        plt.title(title)
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    # Toon de plot als show_plots=True
    if show_plots:
        plot_data(df)

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
    def highpass_filter(data, cutoff=0.5, fs=125, order=4):
        # bepaald de Nyquist-frequentie, helft van de samplefrequentie
        nyquist = 0.5 * fs
        # Normaliseer de cutoff frequentie
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        print("data",data)
        # Pas het filter toe
        filtered_data = filtfilt(b, a, data)
        print("filtered data:",filtered_data)
        return filtered_data

    # Pas het high-pass filter toe op de ruwe ECG-samples
    filtered_ecg = highpass_filter(raw_ecg_samples)
    print("Filtered ECG:",filtered_ecg)

    # ### Verwijder lijnruis
    # 
    # Lijnruis (50/60 Hertz) en andere frequentieruis worden verwijderd met het notch-filter.

    # Notch-filter functei om netspanningsruis te verwijderen (bijv., 50/60 Hz)
    def notch_filter(data, freq=50, fs=125, quality_factor=30):
        # Bereken Nyquist-frequentie
        nyquist = 0.5 * fs
        # Normaliseer de frequentie ten opzichte van Nyquist-frequentie
        freq = freq / nyquist
        b, a = iirnotch(freq, quality_factor)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    # Pas het 50 Hz notch filter toe om netspanningsruis te verwijderen
    filtered_ecg = notch_filter(filtered_ecg, freq=50, fs=125)  # 50 Hz filter
    filtered_ecg

    # pas indien nodig ook het 60 Hz filter toe
    filtered_ecg = notch_filter(filtered_ecg, freq=60, fs=125)  # 60 Hz filter
    filtered_ecg

    df["Timestamp"][0]

    # Definieer een spcifiek tijdsinterval voor visualisatie
    x_range = [df["Timestamp"][998000], df["Timestamp"][1000000]]
    y_range = [-2000, 2000]

    # Visualiseerd de gefilterde ECG data in het gekozen tijdsinterval
    if show_plots:
        plot_data(df, y_range=y_range, x_range=x_range)

    title = "Filtered ECG data on noise"
    # Visualiseer de volledig gefilterde ECG data
    if show_plots:
        plot_data(df, filtered_ecg, title)

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
    # Resample het ECG-signaal van 125 Hz naar 256 Hz
    resampled_ecg = resample(filtered_ecg, int(len(filtered_ecg) * (new_hertz / hertz)))
    print("Resampled ECG", resampled_ecg)

    # #### Ressample tijdas

    # Haal de originele timestamps op in seconden
    original_timestamps = df["timestamp"].to_numpy() 
    original_timestamps

    # Bereken originele tijdstappen in seconden
    time_original = np.arange(len(original_timestamps)) / hertz 
    # Bereken nieuwe tijdstappen op basis van het aantal nieuwe samples
    time_resampled = np.linspace(0, time_original[-1], len(resampled_ecg))  

    # Toon eerste en laatste tijdstappen
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
    print("Resampled df:", resampled_df)

    df.shape # Toon de originele shape van het DataFrame

    # Titel voor visualisatie
    title = f"Resampled ECG data from {hertz} Hz to {new_hertz} Hz"
    # plot de resamplde ECG-gegevens als show_plots=True
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
    print("Spike threshold:", spike_threshold)
    print(resampled_df)

    # Visualiseer de ECG-waarden met de thresholdlijn
    if show_plots:
        plt.close()
        # Plot ECG values with threshold line
        plt.plot(df["Timestamp"][998000:1000000], df["sample"][998000:1000000])
        plt.ylim([-2000, 2000])
        # Plot threshold line
        plt.axhline(spike_threshold, color="r")
        plt.title("Spike threshold = {:.0f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    # ### Detecteer alle mogelijke pieken
    # Gebruik find_peaks om pieken te detecteren
    fs=hertz # Samplefrequentie
    peaks, properties = find_peaks(df["sample"], height=spike_threshold, distance=fs*0.6)

    # Visualiseer de gedetecteerde pieken
    if show_plots:
        # Plot ECG met gedetecteerde pieken
        plt.figure(figsize=(10, 5))
        plt.plot(df["Timestamp"], df["sample"], label="ECG Signaal")
        plt.ylim([-2000, 2000])
        plt.axhline(spike_threshold, color="r", linestyle="--", label="Threshold")

        # Plot pieken
        plt.scatter(df["Timestamp"].iloc[peaks], df["sample"].iloc[peaks], color='red', marker='o', label="Gedetecteerde pieken")

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
    peak_timestamps = df["Timestamp"].iloc[peaks]

    # Bereken de tijdsverschillen tussen opeenvolgende pieken (RR-intervals) in seconden
    time_differences = peak_timestamps.diff().dt.total_seconds()

    # Bekijk de eerste paar tijdsverschillen
    print(time_differences.head(20))

    # Bereken de hartslag per piek (BPM)
    bpm_values = 60 / time_differences

    # Toon de BPM-waarden voor de eerste paar pieken
    print(bpm_values.head(20))

    # Maak een nieuw DataFrame voor de gedetecteerde pieken
    spike_value_df = df.iloc[peaks]  # Gebruik de indices van de pieken om de relevante rijen te selecteren

    # Voeg de tijdsverschillen toe aan dit nieuwe DataFrame 
    spike_value_df["time_differences"] = time_differences.values
    # Bereken de hartslag voor elke piek (in bpm)
    spike_value_df["heartrate"] = 60 / spike_value_df["time_differences"]

    # Bekijk het nieuwe DataFrame
    print(spike_value_df.head(20))

    # ### Plot spikes in het ECG signaal

    # Definieer een bereik van indices voor de x-as van de plots
    resampled_x_range = [2044000, 2048000]
    spikes_x_range = [6851, 6864]

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
        plt.title("Spike threshold = {:.0f}".format(spike_threshold))
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
        # Plot de tijdsintervallen tussen opeenvolgende hartslagen (RR intervallen)
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
    trimmed_ecg = clamped_ecg[:epoch_count * epoch_length] # ECG length is now a multiple of 30 seconds
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
    def get_midnight_offset(start_date_time):
        # Haal de starttijd op uit de startdatum en starttijd
        start_time = start_date_time.strftime("%X")
        print(start_time)

        # Haal uren, minuten en seconden uit de starttijd
        hours = int(start_date_time.strftime("%H"))
        minutes = int(start_date_time.strftime("%M"))
        seconds = int(start_date_time.strftime("%S"))
        # Calculate clock time offset to nearest midnight of when recording began
        offset = ((seconds / 60 + minutes) / 60 + hours) / 24

        # Corrigeer de offset naar een negatieve waarde als de opname vóór middernacht begon
        if start_time > "12:00:00":
            offset = -1 + offset
            
        return np.array([offset])

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

    
    # import os

    # Folder voor ECG data
    preprocessed_ecg_data_base_folder = os.path.dirname(ecg_file)
    preprocessed_ecg_data_folder = "preprocessed_ecg_data_participant_{0}".format(participant_number)
    preprocessed_ecg_data_folder = "{0}/{1}".format(preprocessed_ecg_data_base_folder, preprocessed_ecg_data_folder)

    # Maak de map aan als deze nog niet bestaat
    if not os.path.exists(preprocessed_ecg_data_folder):
        os.makedirs(preprocessed_ecg_data_folder)

    # Maak een unieke bestandsnaam voor de ECG data
    preprocessed_ecg_data_file = "preprocessed_ecg_data_day_{0}.h5".format(file_counter)
    preprocessed_ecg_data_file_name = "{0}/{1}".format(preprocessed_ecg_data_folder, preprocessed_ecg_data_file)
    print(preprocessed_ecg_data_file_name)

    # Maak het HDF5 bestand aan of overschrijf wanneer deze al bestaat
    with h5py.File(preprocessed_ecg_data_file_name, 'w') as hdf5_file:
        # Voeg de ECG data toe aan het bestand
        ecgs_data = hdf5_file.create_dataset("ecgs", data=ecgs)
        # Voeg demografische data toe aan het bestand
        demographics_data = hdf5_file.create_dataset("demographics", data=demographics)
        # Voeg midnight offset toe aan het bestand
        midnight_offset_data = hdf5_file.create_dataset("midnight_offset", data=midnight_offset)

    # ## Read HDF5 file

    # Open HDF5 bestand in read modus
    with h5py.File(preprocessed_ecg_data_file_name, 'r') as hdf5_file:
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

    # Geef de bestandsnaam van het HDF5 bestand terug
    return preprocessed_ecg_data_file_name