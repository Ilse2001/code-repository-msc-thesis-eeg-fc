## This code contains custom functions to load EEG data in for further analysis. The functions are
#   1. load_dpa: loads dpa files as mne.RawArray
#   2. event_code_generator: generate an event convert dict with all possible combinations of event codes (specifically for the eventfiles used by the CBL)
#   3. tobii2events: convert Tobii logfile to an EEG eventlist
#   4. add_markers: adds annotations to the EEG from an eventfile
#   5. load_data_with_markers: loads data from dpa file as an MNE.RawArray with annotations from an eventfile
#   6. extract_segments: extracts and concatenate segments from raw data based on start/end times
#   7. identify_periods: identifies eyes open and eyes closed periods from annotations 
#   8. create_cropped_raw: creates a cropped EEG file only containing test conditions, excludes explanation and practice conditions 
#   9. plot_clinical_eeg: plots clinical representation of the eeg with 19 electrodes 
# The functions can be used in other scripts by using "from Functions_load_data import {name function}"

# import required toolboxes 

import mne 
import datetime
import os
import numpy as np
import pandas as pd

# electrode dictionary 

elec_dict = {
    "Electrode_1020": [
        "Fp1",
        "Fp2",
        "F8",
        "T8",
        "P8",
        "O2",
        "F4",
        "C4",
        "P4",
        "F7",
        "T7",
        "P7",
        "O1",
        "F3",
        "C3",
        "P3",
        "Fz",
        "Cz",
        "Pz",
        "A1",
        "A2",
        "F9",
        "F10",
    ],
    "Nearest_GSN128_Electrode": [
        "E25",
        "E8",
        "E122",
        "E108",
        "E96",
        "E83",
        "E124",
        "E93",
        "E85",
        "E33",
        "E45",
        "E58",
        "E70",
        "E24",
        "E42",
        "E60",
        "E5",
        "E55",
        "E62",
        "E56",
        "E107",
        "E128",
        "E125",
    ],
    "Euclidean_distance": [
        0.0181938809601634,
        0.0176452325197104,
        0.0089669908600607,
        0.0083647875506258,
        0.0135919495402834,
        0.0199487986966936,
        0.0131432801580241,
        0.0147307838528649,
        0.0171061405883185,
        0.0079340633237886,
        0.0070493993629348,
        0.0135507499479835,
        0.0202399853227704,
        0.0132476077609949,
        0.0137353713471133,
        0.0157772903771991,
        0.0175867303133565,
        0.014875549884968,
        0.0154002610447424,
        0.0205018854175188,
        0.020251510805766,
        0.0085717364085546,
        0.0094661620206614,
    ],
}
elec = pd.DataFrame.from_dict(elec_dict)

# Functions

def load_dpa(selected_eeg, EEG_FOLDER = 'EEG', montage = mne.channels.make_standard_montage("GSN-HydroCel-128"), preload=False):
    # loading dpa files as mne.RawArray
    raw = mne.io.read_raw_curry(EEG_FOLDER + "/" + selected_eeg, preload=preload, verbose='CRITICAL')
    raw = raw.pick(picks = "eeg")
    raw = raw.rename_channels(dict(zip(raw.ch_names, montage.ch_names)))
    raw = raw.set_montage("GSN-HydroCel-128")
    raw.info["bads"].append("E55")
    return raw

def event_code_generator():
    # generate an event convert dict with all possible combinations of event codes
    event_convert_dict = {}
    # Facial processing dict
    # Syntax AA00_BB
    fp_aa = ['AF', 'AM', 'BF', 'BM', 'HF', 'HM', 'WF', 'WM']
    fp_bb = ['AO', 'AC', 'HO', 'HC', 'HE', 'SC', 'SO', 'FC', 'FO', 'NC', 'NO']
    # Number can be up to 12, a (1) can be added
    fp_dict = {}
    for fp_idx in range(13):
        for aa in fp_aa:
            for bb in fp_bb:
                temp_key = f'{aa}{str(fp_idx).zfill(2)}_{bb}'
                temp_key1 = f'{aa}{str(fp_idx).zfill(2)}_{bb} (1)'
                fp_dict[temp_key] = f'Facial processing | {temp_key}'
                fp_dict[temp_key1] = f'Facial processing | {temp_key1}'

    # Update event dict
    event_convert_dict = {**event_convert_dict, **fp_dict}

    gng_dict = {}
    for gng_aa in ['a2', 'a3']:
        for gng_x in ['L', 'R']:
            temp_key = f'GNG_{gng_aa}_{gng_x}'
            gng_dict[temp_key] = f'Go - No Go | {temp_key}'
            gng_dict[temp_key + ' - 2a'] = f'Go - No Go | {temp_key+ ' - 2a'}'
            gng_dict[temp_key + ' - 2b'] = f'Go - No Go | {temp_key+ ' - 2b'}'

    # Update event dict
    event_convert_dict = {**event_convert_dict, **gng_dict}

    sa_dict = {}
    for sa_aa in range(8):
            temp_key = f"POPOUT{str(sa_aa+1)}_newphone"
            sa_dict[temp_key] = f'Social Attention | {temp_key}'
    
    # Update event dict
    event_convert_dict = {**event_convert_dict, **sa_dict}

    sa_dict2 = {}
    for xx_aa in range(18):
            for xx_bb in ['_l', '_r']:
                for xx_cc in ['', ' (1)', ' (2)', ' (3)']:
                    temp_key = f"m{str(xx_aa+1)}{xx_bb}{xx_cc}"
                    sa_dict2[temp_key] = f'Socal attention | Movie {temp_key}'
    
    # Update event dict
    event_convert_dict = {**event_convert_dict, **sa_dict2}

    sl_dict = {}
    for sl_aa in range(8):
            for sl_bb in ['SL1.0', 'SL2.0', 'SL3.0']:
                temp_key = f"empty_ans{str(sl_aa+1)}_{sl_bb}"
                temp_key1 = f"empty_ans_pr{str(sl_aa+1)}_{sl_bb}"
                sl_dict[temp_key] = f'Sequence Learning | {temp_key}'
                sl_dict[temp_key1] = f'Sequence Learning | {temp_key1}'

    for sl_aa in range(9):
        temp_key = f"stip_{str(sl_aa+1)}"
        sl_dict[temp_key] = f'Sequence Learning | {temp_key}'

    # Update event dict
    event_convert_dict = {**event_convert_dict, **sl_dict}

    tf_dict = {}
    for xx_aa in range(3):
            for xx_bb in range(81):
                for xx_cc in range(4):
                    for xx_dd in ['md', 'pd']:
                        temp_key = f"tf{str(xx_aa+1)}_{str(xx_bb+1)}_ca{str(xx_cc+1)}_{xx_dd}"
                        tf_dict[temp_key] = f'Matrix Reasoning | Movie {temp_key}'
    
    # Update event dict
    event_convert_dict = {**event_convert_dict, **tf_dict}

    # Events that follow no pattern, but repeat for Wavy, Breeny and Neuro
    avatar_dict = {}
    for avatar in ["Wavy", "Neuro", "Breeny"]:
        avatar_dict[f"{avatar}_321"] =  f"{avatar} | Countdown to start"
        avatar_dict[f"{avatar}_321_lastframe"] = f"{avatar} | Countdown to start finished"
        avatar_dict[f"{avatar}_321_trimmed"] = f"{avatar} | Countdown to start trimmed"
        avatar_dict[f"{avatar}_datwashet_lastframe"] = f"{avatar} | Final frame"
        avatar_dict[f"{avatar}_datwashet_trimmed"] = f"{avatar} | Final frame trimmed"
        avatar_dict[f"{avatar}_duimen"] = f"{avatar} | Thumbs up"
        avatar_dict[f"{avatar}_focus_boven"] = f"{avatar} | Focus up"
        avatar_dict[f"{avatar}_focus_links"] = f"{avatar} | Focus left"
        avatar_dict[f"{avatar}_focus_midden_1s"] = f"{avatar} | Focus center 1 sec"
        avatar_dict[f"{avatar}_focus_midden_2s"] = f"{avatar} | Focus center 2 sec"
        avatar_dict[f"{avatar}_focus_midden_4s"] = f"{avatar} | Focus center 4 sec"
        avatar_dict[f"{avatar}_focus_onder"] = f"{avatar} | Focus down"
        avatar_dict[f"{avatar}_focus_rechts"] = f"{avatar} | Focus right"
        avatar_dict[f"{avatar}_goedjebestgedaan"] = f"{avatar} | Well done"
        avatar_dict[f"{avatar}_goedjebestgedaan_lastframe"] = f"{avatar} | Well done last frame"
        avatar_dict[f"{avatar}_oefenen"] = f"{avatar} | Practice"
        avatar_dict[f"{avatar}_oefenen_lastframe"] = f"{avatar} | Practice last frame"
        avatar_dict[f"{avatar}_ogen_dicht"] = f"{avatar} | Eyes closed"
        avatar_dict[f"{avatar}_ogen_open"] = f"{avatar} | Eyes open"
        avatar_dict[f"{avatar}_opendicht"] = f"{avatar} | Closing and opening eyes"
        avatar_dict[f"{avatar}_opendicht_lastframe"] = f"{avatar} | Closing and opening eyes last frame"
        avatar_dict[f"{avatar}_snippers"] = f"{avatar} | Snippers"
        avatar_dict[f"{avatar}_testje_klaarvoor"] = f"{avatar} | Ready for the start"
        avatar_dict[f"{avatar}_testje_klaarvoor_lastframe"] = f"{avatar} | Ready for the start last frame"
        avatar_dict[f"{avatar}_uitrusten"] = f"{avatar} | Time to rest"
        avatar_dict[f"{avatar}_uitrusten_lastframe"] = f"{avatar} | Time to rest eyes last frame"
        avatar_dict[f"{avatar}_weertestje"] = f"{avatar} | Time for another test"
        avatar_dict[f"{avatar}_weertestje_lastframe"] = f"{avatar} | Time for another test last frame"
        avatar_dict[f"{avatar}__titelslide_leeg"] = f"{avatar} | Empty title slide",
        avatar_dict[f"{avatar}_titelslide_leeg"] = f"{avatar} | Empty title slide"
    
    # Update event dict
    event_convert_dict = {**event_convert_dict, **avatar_dict}

    # Rest, no logical rule to events
    manual_event_dict = {
        "Eyetracker Calibration": "Eyetracker calibration",
        "FP2.0AnswerSheet": "Facial processing | Answer sheet",
        "GNG_1_expl_fox_hedgehog (1)": "Go - No Go | Explanation rule 1",
        "GNG_2a_expl_fox_hedgehog (1)": "Go - No Go | Explanation rule 2a",
        "GNG_2b_expl_fox_hedgehog (1)": "Go - No Go | Explanation rule 2b",
        "RS2.0 Text to Continue - 5-7 years (1)": "Protocol RS2.0 | 5-7 years",
        "RS2.0 Text to Continue - 9-13 years": "Protocol RS2.0 | 9-13 years",
        "RS2.0 Text to Continue - 30-42 months": "Protocol RS2.0 | 30-42 months",
        "RS2.0 Text to Continue - 15-18 years": "Protocol RS2.0 | 15-18 years",
        
        "leeg": "Empty",
        'nan' :  'Empty',  
        np.nan : 'Empty',          
        "wit": "White screen",
        "Text (1)": "Text",
        "Text (2)": "Text",
        "Text (4)": "Text",
        "Text (5)": "Text",
        "Text (13)": "Text",
    }
    # Update event dict
    event_convert_dict = {**event_convert_dict, **manual_event_dict}

    return event_convert_dict

def tobii2events(eventlist, startEEG, endEEG):
    # converting Tobii logfile to an EEG eventlist

    # Define required columns
    eventcols = [
        "Recording timestamp",
        "Computer timestamp",
        "Recording date",
        "Recording start time",
        "Event",
        "Presented Stimulus name",
        "Presented Media name",
    ]

    event_convert_dict = event_code_generator()

    # Load the log file exported by Tobii
    tobiilog = pd.read_csv(eventlist, sep="\t", usecols=eventcols)

    # Create a timestamp per row
    eventfile_RAW = tobiilog.copy()

    # Extract moment of start of experiment
    start_experiment = (
        eventfile_RAW["Recording date"].iloc[0]
        + " "
        + eventfile_RAW["Recording start time"].iloc[0]
    )

    # Datetime expects microseconds as 6 digit zero padded, convert
    start_exp_conv = (
        start_experiment.split(".")[0] + "." + start_experiment.split(".")[1].zfill(6)
    )

    start_exp_ts = pd.to_datetime(start_exp_conv, format="%d/%m/%Y %H:%M:%S.%f")
    eventfile_RAW['Computer timestamp delta'] = [x - eventfile_RAW.at[0, 'Computer timestamp'] for x in eventfile_RAW['Computer timestamp']]

    eventfile_RAW["Time"] = pd.to_datetime(
        eventfile_RAW["Computer timestamp delta"], unit="us", origin=start_exp_ts,
        utc=True
    )
    event_start = eventfile_RAW["Time"].min()

    # Feedback: print start, end and duration of EEG and events
    print("Start, end and duration of event list and EEG.")
    print("\t Start \t \t End \t \t Duration (min)")
    print(f"EEG \t {startEEG.strftime('%H:%M:%S')} \t {endEEG.strftime('%H:%M:%S')} \t {round((endEEG - startEEG).total_seconds() / 60, 1)}")
    print(f"Events \t {event_start.strftime('%H:%M:%S')} \t {eventfile_RAW["Time"].max().strftime('%H:%M:%S')} \t {round((eventfile_RAW["Time"].max() - eventfile_RAW["Time"].min()).total_seconds()/60, 1)}")

    # Select only events within the limits of the EEG
    events_in_eeg = eventfile_RAW[(eventfile_RAW['Time'] > startEEG) & \
                                     (eventfile_RAW['Time'] < endEEG)].copy()

    # Create a timestamp in microseconds relative to start of EEG
    events_in_eeg['Timestamp in EEG (microsec)'] = [(x - startEEG) / datetime.timedelta(microseconds=1) for x in events_in_eeg['Time']]
    # Convert to usable eventfile for EEG
    eventfile_eeg = events_in_eeg[
        ["Timestamp in EEG (microsec)", "Presented Stimulus name"]
    ].copy()

    # Compress to first entry per changing stimulus
    eventfile_eeg_copy = eventfile_eeg.copy()
    valid_stimuli = ~eventfile_eeg_copy['Presented Stimulus name'].isin(['leeg', 'nan'])
    valid_stimuli &= ~eventfile_eeg_copy['Presented Stimulus name'].isna()
    eventfile_filtered = eventfile_eeg_copy[valid_stimuli].copy()

    eventfile_filtered['stimulus_changed'] = eventfile_filtered['Presented Stimulus name'].shift() != eventfile_filtered['Presented Stimulus name']
    eventfile_filtered.loc[eventfile_filtered.index[0], 'stimulus_changed'] = True
    eventfile = eventfile_filtered[eventfile_filtered['stimulus_changed']].copy()
    eventfile = eventfile.set_index("Presented Stimulus name").sort_values('Timestamp in EEG (microsec)')

    # Create dictionary of event codes
    eventcodes = eventfile.index.unique().to_list()
    eventdict = dict(zip(eventcodes, list(range(len(eventcodes)))))

    eventfile["Eventdescription"] = [str(x) for x in eventfile.index.to_list()]
    eventfile["Eventcode"] = [eventdict[x] for x in eventfile.index.to_list()]
    eventfile["Empty"] = [0] * len(eventfile)
    events_new_index = [event_convert_dict[x] for x in eventfile.index.to_list()]
    eventfile["FullDescription"] = events_new_index
    eventfile = eventfile.set_index("FullDescription")
    eventfile['Timestamp (microsec)'] = [int(x) for x in eventfile["Timestamp in EEG (microsec)"]]

    eventsframe = eventfile[["Timestamp (microsec)", "Empty", "Eventcode"]].copy()

    return eventsframe, eventdict, event_start

def add_markers(raw, eventfile, EVENT_FOLDER = 'Eyetracker'):
    #adds annotations to the EEG from an eventfile
    startEEG = raw.info['meas_date']
    endEEG = startEEG + datetime.timedelta(seconds = len(raw)/raw.info['sfreq'])
    if os.path.exists(EVENT_FOLDER + "/" + eventfile):
        try: 
            event_raw, event_dict, event_start = tobii2events(EVENT_FOLDER + "/" + eventfile, startEEG, endEEG)
            event_raw["Time_sec"] = [x / 1e6 for x in event_raw["Timestamp (microsec)"]]

            # Create annotations
            annot = mne.Annotations(onset=event_raw["Time_sec"].to_numpy(), duration=0, description=event_raw.index.to_numpy())
            raw = raw.set_annotations(annot, verbose="ERROR")

            if event_raw.empty:
                print("no events found")
                return raw, None, None, False 
            
            return raw, event_raw, event_dict, True

        except Exception as e:
            print(f"Error in tobii2events: {e}")
            return raw, None, None, False 
    
    else:
        print("eventfile does not exists")
        return raw, None, None, False

def load_data_with_markers(selected_eeg, EEG_FOLDER= 'EEG', EVENT_FOLDER = 'Eyetracker'):
    # loads data from dpa file as an MNE.RawArray with annotations from an eventfile
    eventfile = selected_eeg.replace("_convert.cdt.dpa", ".tsv")
    raw = load_dpa(selected_eeg, EEG_FOLDER = EEG_FOLDER)
    raw, event_raw, event_dict, markers = add_markers(raw, eventfile, EVENT_FOLDER=EVENT_FOLDER)
    return raw, event_raw, event_dict, markers

def extract_segments(raw, periods):
    # Extract and concatenate segments from raw data based on start/end times.
    epochs = []
    for start, stop in periods:
        segment = raw.copy().crop(tmin=max(0, start), tmax=min(stop, raw.times[-1]))
        epochs.append(segment.get_data())
    return np.concatenate(epochs, axis=1)

def identify_periods(raw):
    # Identify eyes open and eyes closed periods from annotations 
    annotations = raw.annotations
    starts = annotations.onset
    labels = annotations.description

    open_periods = []
    closed_periods = []
    
    for i in range(len(starts)-1):
        onset = starts[i]
        offset = starts[i + 1]
        label = labels[i].lower()

        if 'eyes open' in label: 
            open_periods.append((onset, offset))
        elif 'eyes closed' in label:
            closed_periods.append((onset, offset))

    if not open_periods and not closed_periods:
        raise ValueError("No eyes open or eyes closed periods found in annotations")
        
    return open_periods, closed_periods

def create_cropped_raw(raw, events, event_dict, log = False):
    # creates a cropped EEG file only containing test conditions, excludes explanation and practice conditions
    
    # initiate lists and variables
    reverse_event_dict = {code: name for name, code in event_dict.items()}
    segments_to_keep = []
    current_start = None
    in_practice_section = False
    sorted_events = events[events[:, 0].argsort()]
    recording_start = None
    events_to_exclude = []
    
    for i in range(len(sorted_events)):
        event_code = sorted_events[i, 2]
        event_name = reverse_event_dict.get(event_code, '')
        event_time = (sorted_events[i, 0] / raw.info['sfreq'])/1000 

        avatars = ["Wavy", "Neuro", "Breeny"]
        excluded_suffixes = ['_ogen_open', '_ogen_dicht']
        
        # check if event needs to be excluded 
        is_avatar_event = any(
                            avatar in event_name and all(f"{avatar}{suffix}" not in event_name for suffix in excluded_suffixes)
                            for avatar in avatars
                            )
        is_calibration = "Eyetracker Calibration" in event_name
        is_protocol = "RS2.0" in event_name
        is_explanation = "expl_fox_hedgehog" in event_name 
        
        # find begin and end of practive sessions
        if "oefenen" in event_name and any(avatar in event_name for avatar in ["Wavy", "Neuro", "Breeny"]):
            in_practice_section = True
        elif "321" in event_name and any(avatar in event_name for avatar in ["Wavy", "Neuro", "Breeny"]):
            in_practice_section = False

        # save segments to exclude
        if is_avatar_event or is_calibration or is_protocol or is_explanation or in_practice_section: 
            events_to_exclude.append((event_time, event_name))

        # start recording on first relevant event
        if recording_start is None and not (is_avatar_event or is_calibration or is_protocol or 
                                          is_explanation or in_practice_section):
            recording_start = event_time
        
        # define segments to keep 
        if not (is_avatar_event or is_calibration or is_protocol or is_explanation or in_practice_section) and current_start is None:
            current_start = event_time
        elif (is_avatar_event or is_calibration or is_protocol or is_explanation or in_practice_section) and current_start is not None:
            segment_end = event_time
            segments_to_keep.append((current_start, segment_end))
            current_start = None
    
    if current_start is not None:
        segments_to_keep.append((current_start, raw.times[-1]))
    
    # Make new MNE.RawArray with only the test conditions
    if segments_to_keep:
        raw_filtered = mne.io.concatenate_raws([raw.copy().crop(tmin=start, tmax=end) 
                                               for start, end in segments_to_keep])

        if log:
            # give summary
            total_kept_duration = sum(end-start for start, end in segments_to_keep)
            print(f"Original recording length: {raw.times[-1]:.2f} seconds")
            print(f"length of cropped recording: {total_kept_duration:.2f} seconds ({total_kept_duration/raw.times[-1]*100:.1f}%)")
        
        return raw_filtered
    else:
        print("No test conditions found, returning full recording")
        return raw.copy()

def plot_clinical_eeg(raw, start, title, elloc=elec):
    # Function to plot the average EEG data (23 channels), with a bandpass filter applied.
    %matplotlib Tk
    clinselect = raw.copy()
    clinselect.pick(picks=elloc["Nearest_GSN128_Electrode"].to_list())

    # Rename the electrodes
    elnames = dict(
        zip(
            elloc["Nearest_GSN128_Electrode"].to_list(),
            elloc["Electrode_1020"].to_list(),
        )
    )

    clinselect.rename_channels(elnames, verbose='CRITICAL')

    # Invert amplitudes
    raw_info = clinselect.info
    raw_array = clinselect[:, :]
    raw_data = raw_array[0]
    raw_invert = -1 * raw_data

    raw_inv = mne.io.RawArray(raw_invert, raw_info) 

    raw_inv.plot(duration=15, n_channels=23, start=start, clipping=None, splash=False, overview_mode='hidden', theme='light', title = title)
