# Bestandsnaam: Prepare_ECG_files_android.py
# Naam: Esmee Springer
# Laatst bewerkt op: 02-06-2025

# Importeren van benodigde pakkages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv
import json
import re
from io import StringIO
from scipy.signal import find_peaks


def get_hdf5_file_android(ecg_file, file_counter, show_plots=False):
    # Preprocessing van cardiosomnografie data
    # 
    # De ECG-gegevens moeten worden voorbewerkt voordat ze kunnen worden gebruikt als input voor het neuraal netwerk
    # 
    # Inlezen van de ruwe MoveSense data

    # Open het CSV-bestand en lees alle rijen in
    with open(ecg_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Rij 2 (index 1) bevat de metadata in JSON-formaat
    metadata_row = lines[1].strip()  # Verwijder spaties en nieuwe regels aan het begin en einde

    # Corrigeer het JSON-formaat: verwijder vierkante haken in "id"
    cleaned_metadata_row = re.sub(r'"id":"\[#(.*?)\]"', r'"id":"\1"', metadata_row)

    # Zet de JSON-tekst om naar een dictionary
    try:
        metadata = json.loads(cleaned_metadata_row)
        print("Metadata uit rij 2:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except json.JSONDecodeError as e:
        # Toon foutmelding als JSON niet goed geparsed kon worden
        print("Fout bij het parsen van de JSON:", e)

    
    # Inladen van de data
    # Lees de ECG data vanaf rij 3 (zonder kolommen of metadata)
    data_string = "".join(lines[2:])  # Voeg alle rijen vanaf rij 3 samen tot één string

    # Zet de string om in een DataFrame, zonder header en gebruik ";" als scheidingsteken
    df = pd.read_csv(StringIO(data_string), header=None, sep=";")

    # Haal de ECG ruwe metingen op uit kolom 2 (index 1)
    raw_ecg_samples = df[1].to_numpy()  
    
    # Toon aantal gemeten ECG-waarden
    print("Amount of measured ECG values:", len(raw_ecg_samples))
    # Toon de eerste en laatste waarden
    print("\nFirst and last measured ECG values:\n", raw_ecg_samples)

    # Voeg kolomnamen toe aan het DataFrame
    df.columns = ["timestamp", "sample", "bpm"]

    # Toon de eerste 10 rijen van het DataFrame
    print(df.head(10))
   

    # Visualiseer ECG-waarden van de ruwe data

    if show_plots:
        plt.close()
        # Plot the ECG values of a whole night
        plt.plot(df["timestamp"], df["sample"])
        plt.title("Raw data ECG measurement")
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("ECG values")
        plt.show()

    # ##### Opmerking: Wat moet er gebeuren met ECG-waarden boven 250? verwijderen, begrenzen of iets anders?

    # <br><br>
    # 
    # ---
    # 
    # ### Ophalen van metadata over de meting

    try:
        # Haal het participantnummer op uit het pad naar het ECG-bestand
        participant_number = ecg_file.split("/")[-2]
        participant_number = int(participant_number.split("_")[-1])
        print("Measurement of participant {0}".format(participant_number))
    except:
        print("Participant number could not be extracted.")

    # Haal het volledige MoveSense ID op uit de metadata
    try: 
        print("Movesense ID:", metadata.get("id"))
    except: 
        print("Fout bij het ophalen van de Movesense ID")

    #### Ophalen van de datum en tijd van de meting
    # 
    # #### Probeer de startdatum en starttijd te achterhalen uit de bestandsnaam

    try:
        # Haal de startdatum op uit de bestandsnaam
        # Split de bestandsnaam op 'T' om de datum en tijd van elkaar te scheiden
        start_date_time = ecg_file.split("T")

        # Het gedeelte voor de 'T' bevat de datum. Splits op "-" om jaar, maand, dag te verkrijgen
        start_date = start_date_time[0].split("-")[-3:]  # Haal jaar, maand en dag eruit
        start_date = "-".join(start_date)  # Zet de datum terug in het juiste formaat YYYY-MM-DD

        print("Measurement start date:", start_date)

        # Haal de starttijd uit het gedeelte na "T"
        start_time = start_date_time[1]
        start_time = start_time.split(".")[0]

        # Vervang underscores door dubbele punten om de tijd correct te formatteren
        start_time = start_time.replace("_", ":")
        print("Measurement start time:", start_time)  

        # Combineer datum en tijd tot een volledig tijdstip object
        start_timestamp = datetime.datetime.strptime("{0} {1}".format(start_date, start_time), "%Y-%m-%d %H:%M:%S")
        print(start_timestamp)
    except:
        # Print foutmeldingen indien parsing mislukt
        print("Start timestamp could not be extracted.\n")
        print("Are you sure you have the correct file name?")
        print("Expected file name should end like this: xxxxxxxxTxxxxxxZ_xxxxxxxxxxxx_ecg_stream.csv")
        print("Your file name looks like this:", ecg_file)

    # #### Bepaal de duur van de meting

    # Haal de eerste en laatste timestamp op uit het DataFrame
    first_timestamp = df["timestamp"].iloc[0]
    last_timestamp = df["timestamp"].iloc[-1]

    # Bereken de duur van de meting in milliseconden
    # Timestamps staan in seconden, dus vermenigvuldig met 1000 om naar milliseconden te gaan
    duration_milliseconds = (last_timestamp - first_timestamp)*1000 
    print("Measurement took {0} milliseconds.".format(duration_milliseconds))

    # Zet de tijdsduur om naar seconden
    duration_seconds = int(duration_milliseconds / 1000)

    # Bereken uren, minuten en seconden uit het totaal aantal seconden
    minutes, seconds = divmod(duration_seconds, 60) # eerst naar minuten + restseconden
    hours, minutes = divmod(minutes, 60) # daarna naar uren + restminuten

    print("The measurement duration took {0} seconds.".format(duration_seconds))
    print("That amount equals with {} hours, {} minutes, and {} seconds.".format(hours, minutes, seconds))

    # #### Bepaal het tijdstip waarop de meting eindigt

    # Tel de duur van de meting op bij het starttijdstip
    duration_timestamp = datetime.timedelta(seconds=duration_seconds)
    end_timestamp = start_timestamp + duration_timestamp
    print("Measurement started on {0}".format(start_timestamp))
    print("Measurement  ended  on {0}".format(end_timestamp))

    # #### Bereken het interval tussen metingen

    # Bereken de gemiddelde tijd tussen twee metingen
    measurement_interval = duration_seconds / len(df)
    # Bereken het aantal metingen per seconde (Hz)
    hertz = round(1/measurement_interval)
    print("Average time interval between each measurement is: ")
    print(" - {0:.3f} seconds.".format(measurement_interval))
    print(" - {0:.3f} milliseconds.".format(measurement_interval*1000))
    print("\nSample rate: {0} Hertz.".format(hertz))

    # ### Voeg extra timestamps toe voor betere visualisatie

    # Functie om extra kolom met timestamps toe te voegen aan het DataFrame
    def add_timestamps(df, start_timestamp, end_timestamp):
        # Print de starttijd en de eindtijd van de meting
        print("Starting timestamp: ", start_timestamp)
        print("Ending timestamp:", end_timestamp, "\n\n")
        
        # Bereken het totaal aantal minuten tussen starttijd en eindtijd
        total_minutes = (end_timestamp - start_timestamp).total_seconds() / 60
        # Bereken het aantal metingen per minuut
        measurements_per_minute = len(df) / total_minutes
        
        # Maak een lijst met timestamps, verdeeld over de meetperiode
        df["Timestamp"] = [start_timestamp + datetime.timedelta(minutes=i/measurements_per_minute) for i in range(len(df))]
        return df

    # Voeg de extra timestamps toe aan het DataFrame
    df = add_timestamps(df, start_timestamp, end_timestamp)
    df

    # ### Visualiseer ruwe ECG data

    # Functie om ECG data te plotten
    def plot_data(df, ecg_samples=np.array([]), title="Raw data ECG measurement", y_range=[0,0], x_range=[0,0]):
        plt.close() # Sluit eerdere figuren om geheugen te besparen

        # Als er specifieke ECG-samples zijn meegegeven, plot die dan
        if ecg_samples.any():
            ecg_samples = pd.Series(ecg_samples)
            plt.plot(df["Timestamp"], ecg_samples)
        else:
            # Anders plot de standaard 'sample'kolom uit het DataFrame
            plt.plot(df["Timestamp"], df["sample"])

        # Stel het bereik van de y-as in
        if y_range != [0,0]:
            plt.ylim(y_range[0], y_range[1])
        # Stel het bereik van de x-as in
        if x_range != [0,0]:
            plt.xlim(x_range[0], x_range[1])


        plt.title(title)
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30) # Kantel x-as labels voor betere leesbaarheid
        plt.ylabel("ECG values")
        plt.show()

    # Indien show_plots = True, laat de plot zien
    if show_plots:
        plot_data(df)

    # <br><br>
    # 
    # ---
    # 
    # ## Ruis filteren
    # 
    # ### High pass filter
    # 
    # High-pass filter instellen op 0,5 Hertz om de baseline wander te verwijderen (ademhalingsruis)

    # Importeer signaalverwerkingsfunctie voor het filteren van de data
    from scipy.signal import butter, filtfilt, iirnotch, resample

    # Functie voor high-pass filter
    def highpass_filter(data, cutoff=0.5, fs=hertz, order=4):
        nyquist = 0.5 * fs # Nyquist frequentie is de helft van de samplefrequentie
        normal_cutoff = cutoff / nyquist # Normaliseerd de cutoff frequentie
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        # Pas het filter toe op de data, zonder faseverschuiving (filtfilt)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    # Pas het high-pass filter toe op de ruwe ECG-waarden
    highpass_filtered_ecg = highpass_filter(raw_ecg_samples)
    highpass_filtered_ecg

    # ### Verwijder lijnruis
    # 
    # Lijnruis op 50 of 60 Hertz en andere constante frequenties moeten worden verwijderd met notchfilters

    # Functie voor notch-filter om netspanningsruis te verwijderen (bijvoorbeeld 50 of 60 Hz)
    def notch_filter(data, freq=50, fs=hertz, quality_factor=30):
        nyquist = 0.5 * fs # Nyquist frequentie is de helft van de samplefrequentie
        freq = freq / nyquist # Normaliseer de frequentie
        b, a = iirnotch(freq, quality_factor)
        filtered_data = filtfilt(b, a, data)
        # Pas het filter toe zonder faseverschuiving
        return filtered_data

    # Pas notch-filter toe op de eerde gefilterde ECG data om 50 Hz ruis te verwijderen
    filtered_ecg = notch_filter(highpass_filtered_ecg, freq=50, fs=hertz)  
    filtered_ecg

    # Pas notch-filter toe om ook 60 Hz ruis te verwijderen
    filtered_ecg = notch_filter(filtered_ecg, freq=60, fs=hertz)  # Apply 60 Hz notch filter (if applicable)
    filtered_ecg

    # Pak de eerste timestamp uit het DataFrame
    df["Timestamp"][0]

    # Bepaal het laatste indexnummer in het DataFrame
    max_index = len(df) - 1
    # Beginindex: 20 000 rijen voor het einde, of 0 als het minder is
    start_idx = max(0, max_index - 20000)  
    # Eindindex is de laatste rij
    end_idx = max_index

    # Definieer het x-bereik voor de plot
    x_range = [df["Timestamp"].iloc[start_idx], df["Timestamp"].iloc[end_idx]]
    # y-bereik van de plot (ECG-waarden tussen -1 en 1)
    y_range = [-1, 1]

    # Als show_plots=True toon de figuur
    if show_plots:
        plot_data(df, y_range=y_range, x_range=x_range)

    title = "Filtered ECG data on noise"
    # toon de gefilterde ECG-data met titel
    if show_plots:
        plot_data(df, filtered_ecg, title)

    # <br><br>
    # 
    # ---
    # 
    # ## Resample de data naar 256 Hertz
    # 
    # We resamplen de data naar 256 Hertz omdat het neurale netwerk getraind is op ECG data van 256 Hertz
    # 
    # 
    # 
    # #### Resampling van ECG-data

    new_hertz = 256 # Nieuwe samplefrequentie

    # resample de gefilterde ECG-data van originele samplefrequentie naar 256 Hz
    resampled_ecg = resample(filtered_ecg, int(len(filtered_ecg) * (new_hertz / hertz)))
    resampled_ecg

    # #### Resample de tijdas

    original_timestamps = df["timestamp"].to_numpy() # Original timestamps numpy array
    original_timestamps

    # Lineaire interpolatie voor nieuwe timestamps
    time_original = np.arange(len(original_timestamps)) / hertz  # Originele tijdstappen in seconden
    time_resampled = np.linspace(0, time_original[-1], len(resampled_ecg))  # Nieuwe tijdstappen in seconden

    print("Original timesteps: ", time_original[:5], "...", time_original[-5:])
    print("Resampled timesteps:", time_resampled[:5], "...", time_resampled[-5:])

    # Interpoleer originele timestamps naar de nieuwe tijdas
    resampled_timestamps = np.interp(time_resampled, time_original, original_timestamps)

    print("Original timestamps: ", original_timestamps[:5], "...", original_timestamps[-5:])
    print("Resampled timestamps:", resampled_timestamps[:5], "...", resampled_timestamps[-5:])

    # Maak een niwue DataFrame aan met de resamplede timsestamps en ECG-waarden
    resampled_df = pd.DataFrame({
        'timestamp': resampled_timestamps,
        'sample': resampled_ecg
    })

    # Voeg extra timestamps toe voor betere visualisatie
    resampled_df = add_timestamps(resampled_df, start_timestamp, end_timestamp)
    resampled_df

    # Toon de shape van het oorspronkelijke DataFrame
    df.shape

    # Titel voor de plot
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
    # De mediaan van alle data moet er worden afgehaald. De mediaan van alle ECG-waarden moet gelijk zijn aan 0.

    # Bereken de mediaan van de resampled ECG data
    median_ecg = np.median(resampled_ecg)
    median_ecg

    # Haal de mediaan er van af om de data rond 0 te centreren
    centered_ecg = resampled_ecg - median_ecg

    # Controleer of de mediaan nu (bijna) nul is
    np.abs(np.median(centered_ecg))

    # Werk de kolom 'sample' bij met de gecentreerde data
    resampled_df["sample"] = centered_ecg
    resampled_df

    # Plot titel
    title = "Median of ECG data is now equal to 0"
    if show_plots:
        plot_data(resampled_df, centered_ecg, title, y_range=y_range, x_range=x_range)

    # ---
    # 
    # ## Spikes & HeartRate
    # 
    # ### Bereken thresholdwaarde of hartslagpieken

    # Bereken de threshold als de standaarddeviatie van de ECG data
    spike_threshold = np.std(resampled_df["sample"])
    spike_threshold

    if show_plots:
        plt.close()
        # Plot een gedeelte van de ECG waarden met de thresholdlijn
        plt.plot(df["Timestamp"][998000:1000000], df["sample"][998000:1000000])
        plt.ylim([-1, 1])
        # Plot de thresholdlijn
        plt.axhline(spike_threshold, color="r")
        plt.title("Spike threshold = {:.2f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    # ### Detecteer alle potentiële pieken
    # Detecteer alle mogelijke pieken in het ECG-siganaal
    fs=hertz
    # Vind alle pieken die hoger zijn dan de thresholdwaarde en minstens 0,6 seconden van elkaar verwijderd zijn
    peaks, properties = find_peaks(df["sample"], height=spike_threshold, distance=fs*0.6)

    if show_plots:
        # Plot het ECG-signaal met de gedetecteerde pieken
        plt.figure(figsize=(10, 5))
        plt.plot(df["Timestamp"], df["sample"], label="ECG Signaal")
        plt.ylim([-1, 1])
        # Plot de thresholdlijn
        plt.axhline(spike_threshold, color="r", linestyle="--", label="Threshold")

        # Plot de gedetecteerde pieken als rode cirkels
        plt.scatter(df["Timestamp"].iloc[peaks], df["sample"].iloc[peaks], color='red', marker='o', label="Gedetecteerde pieken")

        # Labels en titel instellen
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

    # Bereken de tijdsverschillen tussen de pieken (RR-intervals) in seconden
    time_differences = peak_timestamps.diff().dt.total_seconds()

    # Toon de eerste 20 tijdsverschillen
    print(time_differences.head(20))

    # Bereken de hartslag per piek (BPM)
    bpm_values = 60 / time_differences

    # Toon de BPM-waarden voor de eerste 20 pieken
    print(bpm_values.head(20))

    # Maak een nieuw DataFrame met alleen de rijen van de gedetecteerde pieken
    spike_value_df = df.iloc[peaks]  # Gebruik de indices van de pieken om de relevante rijen te selecteren

    # Voeg de tijdsverschillen toe als extra kolom aan dit nieuwe DataFrame 
    spike_value_df["time_differences"] = time_differences.values
    # Bereken de hartslag voor elke piek (in bpm)
    spike_value_df["heartrate"] = 60 / spike_value_df["time_differences"]

    # Toon de eerste 20 rijen van het nieuwe DataFrame
    print(spike_value_df.head(20))

    # ### Plot spikes (hartslagpieken)

    # inzoomen voor duidelijke weergave van ECG-signaal
    resampled_x_range = [0, 2048000] # Bereik voor het ECG-signaal
    spikes_x_range = [0, 6864] # Bereik voor pieken

    if show_plots:
        plt.close()
        # Plot ECG waarden binnen het gekozen bereik
        plt.plot(resampled_df["Timestamp"][resampled_x_range[0]:resampled_x_range[1]], resampled_df["sample"][resampled_x_range[0]:resampled_x_range[1]])
       
        plt.ylim([-2000, 2000]) # y-as limieten voor betere zichtbaarheid
        # Plot de thresholslijn voor hartslagpieken
        plt.axhline(spike_threshold, color="r")
        # Plot alle gedetecteerde pieken binnen het kleinere bereik
        plt.scatter(spike_value_df["Timestamp"][spikes_x_range[0]:spikes_x_range[1]], spike_value_df["sample"][spikes_x_range[0]:spikes_x_range[1]], c="r", s=10)
        # Layout aanpassen
        plt.title("Spike threshold = {:.2f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

        plt.close()
        # Plot de ECG data met alle pieken
        plt.plot(resampled_df["Timestamp"], resampled_df["sample"])
        
        # Plot alle spikes over de hele dataset
        plt.scatter(spike_value_df["Timestamp"], spike_value_df["sample"], c="r", s=10)
        
        # Layout aanpassingen
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("ECG values")
        plt.show()

    # ### Bereken InterBeat Interval (tijd tussen hartslagen)

    # # ### Bereken hartslag

    if show_plots:
        plt.close()
        # Plot de berekende hartslag over de hele nacht (afgeleid van de RR intervals)
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
    # ### Scale ECG-signaal naar bereik [-0.5, 0.5]
    # 
    # Eerst meten we de minimum en maximum waarden van elkar hartslagpiek. (Dus niet van alle ECG-data, maar alleen van de pieken).<br>
    # De data wordt geschaal zodat de 90e percentiel (of hoger) van deze min/max waarden binnen het bereik valt [-0.5, 0.5].
    # 
    # Beweging en andere ruis kunnen hogere amplitudes veroorzaken dan de meeste hartslagen. <br>
    # Deze ruis kan dus buiten het bereik [-1.0, -0.5] liggen, bijvoorbeeld tussen [-1.0, -0.5] of [0.5, 1.0]. 

    # Pak alleen de waarden van de hartslagen (pieken)
    spike_values = spike_value_df["sample"]
    spike_values

    # Bereken het 10e (min) en 90e (max) percentiel van de piekwaarden
    min_value = np.percentile(spike_values, 10)
    max_value = np.percentile(spike_values, 90)
    print("The 90th percentile of the minimum heartbeat values:", min_value)
    print("The 90th percentile of the maximum heartbeat values:", max_value)

    # Schaalfactor gebaseerd op de grootste absolute waarde
    scale_factor = 0.5 / max(abs(min_value), abs(max_value))
    print("Scale factor:", scale_factor)

    # Schaal het gecentreerde ECG-signaal met de berekende schaalfactor
    scaled_ecg = centered_ecg * scale_factor
    resampled_df["sample"] = scaled_ecg
    resampled_df

    print("Is the median still equal to 0?")
    print("Median:", np.abs(np.median(scaled_ecg)))

    # Titel voor de visualisatie van het geschaalde ECG signaal
    title = "Scaled ECG data to scale [-0.5, 0.5]"
    if show_plots:
        plot_data(resampled_df, scaled_ecg, title, x_range=x_range, y_range=[-1, 1])

    if show_plots:
        plot_data(resampled_df, scaled_ecg, title)

    # ### Clamp outliers
    # 
    # Na het schalen van de ECG-data naar het bereik [-0.5, 0.5] blijven er mogelijk nog enkele uitschieters over.
    # Dit kunnen bijvoorbeeld hoge pieken van hartslagen zijn of ruis afkomstig van beweging
    # Om te zorgen dat alle waarden binnen een verwachte range vallen, worden de waarden geclamped
    #
    # Alle waarden worden beperkt tot het bereik [-1.0, 1.0]
    # Waarden boven 1.0 worden gelijkgemaakt aan 1.0, waarden onder -1.0 worden gelijkgemaakt aan 1.0

    # Clamp waarden naar het bereik [-1.0, 1.0] om uitschieters en ruis te beperken
    clamped_ecg = np.clip(scaled_ecg, -1.0, 1.0)
    clamped_ecg

    # Controleer of de mediaan nog steeds rond 0 ligt
    print("Is the median still equal to 0?")
    print("Median:", np.abs(np.median(clamped_ecg)))

    # Visualiseer de geclampte ECG-gegevens met specifiek bereik
    title = "Clamped outliers between [-1.0, 1.0]"
    if show_plots:
        plot_data(resampled_df, clamped_ecg, title, x_range=x_range, y_range=[-1, 1])

    # Nogmaals visualiseren zonder expliciet bereik
    if show_plots:
        plot_data(resampled_df, clamped_ecg, title)

    # <br><br>
    # 
    # ---
    # 
    # ## Reshape naar 30-seconde epochs
    # 
    # Laatste stap in de preprocessing pipeline: het opdelen van het ECG-signaal in 30-seconden epochs. <br>
    # Elk fragment bevat een vaste hoeveelheid metingen: 30 seconde * 256 Hertz =7680 waarden per epoch. <br>
    # Deze fragmenten worden georganiseerd in een 2D array waarbij elke rij een afzonderlijke epoch vertegenwoordigt
    # Dit formaat is ideaal voor het trainen van neurale netwerken op tijdsreeksen

    # Bereken de lengte van één epoch
    epoch_length = 30 * new_hertz # 30 seconde * 256 Hertz = 7680 datapunten per epoch
    
    # Bereken het aantal epochs dat geëxtraheerd kan worden uit de clamped ECG data
    epoch_count = len(clamped_ecg) // epoch_length
    print("Dataset shape: ({}, {})".format(epoch_count, epoch_length))

    # Knip de ECG-data af zodat het een veelvoud is van een epoch lengte
    # Hierdoor wordt voorkomen dat er onvolledige epoch zijn op het einde
    trimmed_ecg = clamped_ecg[:epoch_count * epoch_length] 
    print("Amount of measurements taken into account:", len(trimmed_ecg))

    # Reshape the ECG data naar een 2D array
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

    # Controleer of de ECG array de juiste dimensies heeft
    print("ECG size should be: \t(epoch_count x 7680)")
    print(f"ECG size is: \t\t{ecgs.shape}")

    # Controleer of het datatype van de ECG-array correct is
    print("\nECG type should be a: \t2D numpy array of floats")
    print(f"ECG types are: \t\t{type(ecgs)} \n\t\t\t{type(ecgs[1][1])}")

    # #### Demographics

    # Functie om het geslacht en de leeftijd om te zetten naar een array van demografische gegevens
    def get_demographics(gender, age):
        # Zet geslacht om naar binaire waarde
        if gender.lower() == "Male" or "M" or "Man":
            sex = 1
        elif gender.lower() == "Female" or "F" or "Woman":
            sex = 0
        else:
            return "Gender is not entered correctly. Please insert Male or Female"
        
        # Zet leeftijd om naar een float gedeeld door 100
        age = age/100
        
        # Geef numpy array met [geslacht, leeftijd] terug
        return np.array([sex, age])

    # Demografische gegevens invoeren
    gender = "Male"
    age = 40
    demographics = get_demographics(gender, age)

    # Controleer of de array de juiste dimensies en waardes heeft
    print("Demographics size should be: \t(2 x 1)")
    print(f"Demographics size is: \t\t{demographics.shape}")

    print("\nDemographics should be contain a 2D array of floats representing:")
    print(" - Sex: Binary digit (0=female or 1=male)")
    print(" - Age: Floating number (age/100)")

    # Toon inhoud van de demografische array
    print("\nDemographics variable contains:")
    print(" - Sex:", demographics[0])
    print(" - Age:", demographics[1])

    from tabulate import tabulate

    # Toon demografische gegevens netjes in tabelvorm
    table = [[gender, age]]
    headers = ["Gender", "Age"]
    print(tabulate(table, headers, tablefmt="pretty"))

    # #### Midnight offset

    # Deze functie berekent het tijdsverschil tussen het begin van de opname en middernacht
    def get_midnight_offset(start_date_time):
        # Verkrijg het tijdsstip van de starttijd
        start_time = start_date_time.strftime("%X")
        print(start_time)

        # haal uren, minuten en seconden uit de starttijd
        hours = int(start_date_time.strftime("%H"))
        minutes = int(start_date_time.strftime("%M"))
        seconds = int(start_date_time.strftime("%S"))
        # Bereken de offset in uren
        offset = ((seconds / 60 + minutes) / 60 + hours) / 24

        # Corrigeer de offset naar een negatieve waarde als de opname vóór middernacht begon
        if start_time > "12:00:00":
            offset = -1 + offset
            
        return np.array([offset])
    # Print de huidige datum en tijd om te controleren wanneer de code wordt uitgevoerd
    datetime.datetime.now()

    # Bereken de midnight offset

    start_date_time = datetime.datetime.now()
    # Gebruik de eerder gedefinieerde functie om de offset te berekenen
    midnight_offset = get_midnight_offset(start_timestamp)

    # Controleer of 'midnight_offset' in het juiste formaat staat
    print("Midnight offset should contain a float between the range [-1, 1] representing the clocktime offset of when the recording began.")
    print("Midnight offset value:", midnight_offset[0])

    # ## Create HDF5 file
    # Doel: Sla alle voorbewerkte ECG-gegevens op in een HDF5-bestand dat kan worden gebruikt als input voor een neuraal netwerk

    import h5py
    import os

    # Stel de naam samen voor de map waarin de voorbewerkte ECG-data zal worden opgeslagen per participant
    preprocessed_ecg_data_base_folder = os.path.dirname(ecg_file)
    preprocessed_ecg_data_folder = "preprocessed_ecg_data_participant_{0}".format(participant_number)
    preprocessed_ecg_data_folder = "{0}/{1}".format(preprocessed_ecg_data_base_folder, preprocessed_ecg_data_folder)

    # Maak de map aan als deze nog niet bestaat
    if not os.path.exists(preprocessed_ecg_data_folder):
        os.makedirs(preprocessed_ecg_data_folder)

    # Stel de bestandsnaam samen voor het uiteindelijke h5 bestand per dag
    preprocessed_ecg_data_file = "preprocessed_ecg_data_day_{0}.h5".format(file_counter)
    preprocessed_ecg_data_file_name = "{0}/{1}".format(preprocessed_ecg_data_folder, preprocessed_ecg_data_file)
    print(preprocessed_ecg_data_file_name)

    # Maak een nieuw HDF5 bestand of overschrijf het bestaande bestand
    with h5py.File(preprocessed_ecg_data_file_name, 'w') as hdf5_file:
        # Voeg de ECG waarden toe
        ecgs_data = hdf5_file.create_dataset("ecgs", data=ecgs)
        # Voeg demografische gegevens toe
        demographics_data = hdf5_file.create_dataset("demographics", data=demographics)
        # Voeg de midnight offset toe
        midnight_offset_data = hdf5_file.create_dataset("midnight_offset", data=midnight_offset)

    # ## Read HDF5 file

    # Open het HDF5 bestand in read modus
    with h5py.File(preprocessed_ecg_data_file_name, 'r') as hdf5_file:
        # Lees de ECG data
        ecgs_data = hdf5_file["ecgs"]
        # Lees de demografische gegevens
        demographics_data = hdf5_file["demographics"]
        # Lees de offset ten opzichte van middernacht
        midnight_offset_data = hdf5_file["midnight_offset"]
        # Print de inhoud
        print(ecgs_data[()])
        print(demographics_data[()])
        print(midnight_offset_data[()])

    # geef de bestandsnaam en het oorspronkelijke DataFrame terug
    return preprocessed_ecg_data_file_name, df