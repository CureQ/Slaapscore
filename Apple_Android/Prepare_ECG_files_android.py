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
    # # Cardiosomnography data preprocessing
    # 
    # The ECG data needs to be preprocessed, before the ECG data can be given as input for the neural network.
    # 
    # ### Read the raw MoveSense ECG data

    # Open het CSV-bestand en lees alle rijen in
    with open(ecg_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Rij 2 (index 1) bevat de metadata als JSON
    metadata_row = lines[1].strip()  # Verwijder spaties en newline

    # Fix: JSON opschonen (verwijder vierkante haken in "id")
    cleaned_metadata_row = re.sub(r'"id":"\[#(.*?)\]"', r'"id":"\1"', metadata_row)

    # JSON omzetten naar een dictionary
    try:
        metadata = json.loads(cleaned_metadata_row)
        print("Metadata uit rij 2:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except json.JSONDecodeError as e:
        print("Fout bij het parsen van de JSON:", e)

    
    # Inladen data
    # Overige data inlezen (vanaf rij 3), zonder kolomnamen toe te voegen
    data_string = "".join(lines[2:])  # Pak alles vanaf rij 3

    # Lees de rest van het bestand als een dataframe zonder kolomnamen
    df = pd.read_csv(StringIO(data_string), header=None, sep=";")
    raw_ecg_samples = df[1].to_numpy()  
    #hertz = 191
    print("Amount of measured ECG values:", len(raw_ecg_samples))
    print("\nFirst and last measured ECG values:\n", raw_ecg_samples)

    df.columns = ["timestamp", "sample", "bpm"]
    # Toon de eerste 10 rijen
    print(df.head(10))
   

    # ### Visualize ECG values of raw data

    if show_plots:
        plt.close()
        # Plot the ECG values of a whole night
        plt.plot(df["timestamp"], df["sample"])
        plt.title("Raw data ECG measurement")
        plt.xlabel("Timestamps")
        plt.xticks(rotation=45)
        plt.ylabel("ECG values")
        plt.show()

    # ##### What to do with all ECG values above 2500? Delete, clamp, or?

    # <br><br>
    # 
    # ---
    # 
    # ### First, get all provided information about the measurement

    try:
        # Get the participant number
        participant_number = ecg_file.split("/")[-2]
        participant_number = int(participant_number.split("_")[-1])
        print("Measurement of participant {0}".format(participant_number))
    except:
        print("Participant number could not be extracted.")

    # Get the full file name
    try: 
        print("Movesense ID:", metadata.get("id"))
    except: 
        print("Fout bij het ophalen van de Movesense ID")

    # #### Get the date and time of the sleep session
    # 
    # #### Get start timestamp

    try:
        # Get the start date out of the file name
        # Split de naam op 'T' om de datum en tijd te scheiden
        start_date_time = ecg_file.split("T")
        # Het deel voor de 'T' bevat de datum, we nemen het en splitsen de onderdelen
        start_date = start_date_time[0].split("-")[-3:]  # Haal jaar, maand en dag eruit
        start_date = "-".join(start_date)  # Zet de datum terug in het juiste formaat YYYY-MM-DD

        print("Measurement start date:", start_date)

        # Get the start time out of the file name
        start_time = start_date_time[1]
        start_time = start_time.split(".")[0]
        # Vervang underscores door dubbele punten om de tijd correct te formatteren
        start_time = start_time.replace("_", ":")
        print("Measurement start time:", start_time)  

        # Get the start date and time in a nice format
        start_timestamp = datetime.datetime.strptime("{0} {1}".format(start_date, start_time), "%Y-%m-%d %H:%M:%S")
        print(start_timestamp)
    except:
        print("Start timestamp could not be extracted.\n")
        print("Are you sure you have the correct file name?")
        print("Expected file name should end like this: xxxxxxxxTxxxxxxZ_xxxxxxxxxxxx_ecg_stream.csv")
        print("Your file name looks like this:", ecg_file)

    # #### Get the measurement duration

    # Get timestamps
    first_timestamp = df["timestamp"].iloc[0]
    last_timestamp = df["timestamp"].iloc[-1]
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

    # #### Get end timestamp

    # Add measurement duration to start timestamp
    duration_timestamp = datetime.timedelta(seconds=duration_seconds)
    end_timestamp = start_timestamp + duration_timestamp
    print("Measurement started on {0}".format(start_timestamp))
    print("Measurement  ended  on {0}".format(end_timestamp))

    # #### Get interval between measurements

    # Get average time between each measurement
    measurement_interval = duration_seconds / len(df)
    hertz = round(1/measurement_interval)
    print("Average time interval between each measurement is: ")
    print(" - {0:.3f} seconds.".format(measurement_interval))
    print(" - {0:.3f} milliseconds.".format(measurement_interval*1000))
    print("\nSample rate: {0} Hertz.".format(hertz))

    # ### Add extra Timestamps for better visualization

    # Add extra column with timestamps
    def add_timestamps(df, start_timestamp, end_timestamp):
        # Start timestamp and end timestamp
        print("Starting timestamp: ", start_timestamp)
        print("Ending timestamp:", end_timestamp, "\n\n")
        
        # Calculate the total amount of minutes between start- and end time
        total_minutes = (end_timestamp - start_timestamp).total_seconds() / 60
        # Calculate the total amount of measurement per minute
        measurements_per_minute = len(df) / total_minutes
        
        # Create a list of timestamps
        df["Timestamp"] = [start_timestamp + datetime.timedelta(minutes=i/measurements_per_minute) for i in range(len(df))]
        return df

    # Add timestamps for each measurement
    df = add_timestamps(df, start_timestamp, end_timestamp)
    df

    # ### Visualize raw data

    # Visualize ecg data at any time
    def plot_data(df, ecg_samples=np.array([]), title="Raw data ECG measurement", y_range=[0,0], x_range=[0,0]):
        plt.close()
        # Plot the ECG values of a whole night
        if ecg_samples.any():
            ecg_samples = pd.Series(ecg_samples)
            plt.plot(df["Timestamp"], ecg_samples)
        else:
            #print(df["Timestamp"])
            # print(df.columns)
            # print(df)
            plt.plot(df["Timestamp"], df["sample"])

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
        plot_data(df)

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
    def highpass_filter(data, cutoff=0.5, fs=hertz, order=4):
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

    df["Timestamp"][0]

    max_index = len(df) - 1
    start_idx = max(0, max_index - 20000)  # 20.000 rijen voor het einde, of begin als kleiner
    end_idx = max_index

    x_range = [df["Timestamp"].iloc[start_idx], df["Timestamp"].iloc[end_idx]]
    # x_range = [df["Timestamp"][998000], df["Timestamp"][1000000]]
    y_range = [-1, 1]
    if show_plots:
        plot_data(df, y_range=y_range, x_range=x_range)

    title = "Filtered ECG data on noise"
    if show_plots:
        plot_data(df, filtered_ecg, title)

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

    # #### Resample time axis

    original_timestamps = df["timestamp"].to_numpy() # Original timestamps numpy array
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

    df.shape

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
        plt.plot(df["Timestamp"][998000:1000000], df["sample"][998000:1000000])
        plt.ylim([-1, 1])
        # Plot threshold line
        plt.axhline(spike_threshold, color="r")
        plt.title("Spike threshold = {:.2f}".format(spike_threshold))
        plt.xlabel("Timestamps")
        plt.xticks(rotation=30)
        plt.ylabel("ECG values")
        plt.show()

    # ### Detect all potential spikes
    # Detect all potential spikes
    fs=hertz
    peaks, properties = find_peaks(df["sample"], height=spike_threshold, distance=fs*0.6)

    if show_plots:
        # Plot ECG met gedetecteerde pieken
        plt.figure(figsize=(10, 5))
        plt.plot(df["Timestamp"], df["sample"], label="ECG Signaal")
        plt.ylim([-1, 1])
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
    # Toon aantal pieken
    print(f"Aantal gedetecteerde pieken: {len(peaks)}")

    # Haal de tijdstempels van de gedetecteerde pieken
    peak_timestamps = df["Timestamp"].iloc[peaks]

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
    spike_value_df = df.iloc[peaks]  # Gebruik de indices van de pieken om de relevante rijen te selecteren

    # Voeg de tijdsverschillen toe aan dit nieuwe DataFrame (al eerder berekend)
    spike_value_df["time_differences"] = time_differences.values
    # Bereken de hartslag voor elke piek (in bpm)
    spike_value_df["heartrate"] = 60 / spike_value_df["time_differences"]

    # Bekijk het nieuwe DataFrame
    print(spike_value_df.head(20))

    # # Detect all points above the threshold
    # above_threshold = resampled_df.copy()
    # above_threshold = above_threshold[above_threshold["sample"] > spike_threshold]
    # above_threshold_samples = above_threshold.shape[0]
    # print("Samples in cleaned dataset:", resampled_df.shape[0])
    # print("Samples above threshold:   ", above_threshold_samples)
    # above_threshold[:10]

    # # Variable how many seconds one spike lasts at max (200 milliseconds)
    # spike_duration = 0.2
    # # Calculate how many samples 1 spike lasts at max
    # spike_duration_samples = int(spike_duration * hertz)
    # print("1 spike is always happening in less than {0} samples.".format(spike_duration_samples))

    # # ### Detect only the real spikes

    # # Calculate time differences between sequenced measurements
    # above_threshold["time_differences"] = above_threshold['Timestamp'].diff().dt.total_seconds() * 1000  # Difference in milliseconds
    # above_threshold["time_differences"][:3]

    # # Fill first value of time difference with 1000 instead of NaN, because there is no previous value (This detects the first spike)
    # first_index = above_threshold.index[0]
    # above_threshold.loc[first_index, "time_differences"] = 1000
    # above_threshold[:5]

    # # Group potential spikes in heartbeats
    # minimum_spike_time_difference = 200
    # above_threshold['heartbeat_group'] = (above_threshold['time_differences'] > minimum_spike_time_difference).cumsum()
    # above_threshold[:20]

    # # Select only the rows with the highest sample value in each group (The spike)
    # # Add spike values to spike values dataset
    # spike_value_df = above_threshold.loc[above_threshold.groupby('heartbeat_group')['sample'].idxmax()]
    # # Keep al relevant columns
    # spike_value_df = spike_value_df[["timestamp", "sample", "Timestamp"]]
    # spike_value_df

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

    # #### Demographics

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

    # Calculate midnight offset
    def get_midnight_offset(start_date_time):
        # Get start time from start date time
        start_time = start_date_time.strftime("%X")
        print(start_time)

        # Get sleep time
        hours = int(start_date_time.strftime("%H"))
        minutes = int(start_date_time.strftime("%M"))
        seconds = int(start_date_time.strftime("%S"))
        # Calculate clock time offset to nearest midnight of when recording began
        offset = ((seconds / 60 + minutes) / 60 + hours) / 24

        # Calculate negative offset, if sleep started before midnight
        if start_time > "12:00:00":
            offset = -1 + offset
            
        return np.array([offset])

    datetime.datetime.now()

    # Get midnight offset
    # start_date_time = datetime.datetime(2024, 11, 19, 12, 0, 1) # test
    start_date_time = datetime.datetime.now()
    midnight_offset = get_midnight_offset(start_timestamp)

    # Check if 'midnight_offset' is in correct format
    print("Midnight offset should contain a float between the range [-1, 1] representing the clocktime offset of when the recording began.")
    print("Midnight offset value:", midnight_offset[0])


    # ## Create HDF5 file

    import h5py
    import os

    # Folder for ECG data
    preprocessed_ecg_data_base_folder = os.path.dirname(ecg_file)
    preprocessed_ecg_data_folder = "preprocessed_ecg_data_participant_{0}".format(participant_number)
    preprocessed_ecg_data_folder = "{0}/{1}".format(preprocessed_ecg_data_base_folder, preprocessed_ecg_data_folder)

    # Create folder for ECG data if not exists
    if not os.path.exists(preprocessed_ecg_data_folder):
        os.makedirs(preprocessed_ecg_data_folder)

    # Create unique and easy to understand filename for ECG data
    preprocessed_ecg_data_file = "preprocessed_ecg_data_day_{0}.h5".format(file_counter)
    preprocessed_ecg_data_file_name = "{0}/{1}".format(preprocessed_ecg_data_folder, preprocessed_ecg_data_file)
    print(preprocessed_ecg_data_file_name)

    # Create or overwrite HDF5 file
    with h5py.File(preprocessed_ecg_data_file_name, 'w') as hdf5_file:
        ecgs_data = hdf5_file.create_dataset("ecgs", data=ecgs)
        demographics_data = hdf5_file.create_dataset("demographics", data=demographics)
        midnight_offset_data = hdf5_file.create_dataset("midnight_offset", data=midnight_offset)

    # ## Read HDF5 file

    # Open HDF5 file
    with h5py.File(preprocessed_ecg_data_file_name, 'r') as hdf5_file:
        ecgs_data = hdf5_file["ecgs"]
        demographics_data = hdf5_file["demographics"]
        midnight_offset_data = hdf5_file["midnight_offset"]
        print(ecgs_data[()])
        print(demographics_data[()])
        print(midnight_offset_data[()])

    return preprocessed_ecg_data_file_name, df