## Code for preprocessing EEG files 
# This code will preprocess EEG measurements with the following steps: 
#   1. Prepare data to fit ICA on
#   2. Artifact removal with IClabel
#   3. Standard filters 
# This code can be applied to multiple measurements and the results will be saved as a .fif file. The measurements to 
# include can be given by an assigned folder, also with the option to only analyze the new files. 

#%% 
# Import required toolboxes
import os
import mne
import pyprep
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
from Functions_load_data import load_data_with_markers, create_cropped_raw #custom functions

mne.set_log_level('CRITICAL')

#%%  
# Input 

EEG_folder = 'EEG'
EVENT_folder = 'Eyetracker'
output_dir = 'Preprocessed_EEGs'

montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

#%% 
# Main analysis

eeg_files = os.listdir(EEG_folder)
eeg_files = [x for x in eeg_files if x.endswith("_convert.cdt.dpa")]

for eeg_file in eeg_files:
    print(f"Processing file: {eeg_file}")
    measurement_id = eeg_file.split('_')[0] + '_' + eeg_file.split('_')[3]  #change if the eeg files names are build differently 

    ''' uncomment to only process new files
    if os.path.exists(output_dir + "/" + measurement_id + "_prepped.fif"):
        print('Preprocessed file already exists, continue with next measurement')
        continue
    '''

    # load data
    raw, event_raw, event_dict, markers = load_data_with_markers(eeg_file, EEG_FOLDER=EEG_folder, EVENT_FOLDER=EVENT_folder)
    raw.load_data()

    if markers==False:
        print("Preprocessing not possible because no markers available, continue with the next measurement")
        continue 
    
    events = event_raw.values.astype(int)
    
    ## 1. Prepare data to fit ICA on
    raw_cropped = raw.copy() # prepare data for fitting as a copy to keep raw intact to apply the ICA to later
    raw_cropped = create_cropped_raw(raw_cropped, events, event_dict)
    
    # reject bad channels for fitting 
    noisy_channels1 = pyprep.NoisyChannels(raw_cropped)
    noisy_channels1.find_all_bads()
    bad1 = noisy_channels1.get_bads()
    bad1 = list(map(str, bad1))

    raw_cropped.info["bads"] += bad1
    raw_filtered = raw_cropped.copy()
   
    raw_filtered.filter(l_freq=1,h_freq=100, verbose='CRITICAL')
    raw_filtered.set_eeg_reference("average", verbose='CRITICAL')

    # reject bad epochs for fitting
    segment_dur = 1
    threshold = 100
    sfreq = raw_filtered.info['sfreq']
    segment_samples = int(segment_dur * sfreq)
    n_samples = raw_filtered.n_times
    n_segments = n_samples // segment_samples 
    bad_times = []

    raw_prepped = raw_filtered.copy()

    for i in range(n_segments): 
        start = i * segment_samples
        stop = start + segment_samples
        data_segment, _ = raw_filtered[:, start:stop]
        max_amplitude = np.max(np.abs(data_segment*1e3))
        if max_amplitude > threshold:
            onset = start/sfreq
            bad_times.append((onset, segment_dur))

    for onset, duration in bad_times:
        raw_prepped.annotations.append(onset, duration, "bad artifact")

    ## 2. Artifact removal with IClabel
    filt_raw = raw_prepped.copy() 

    ica = ICA(n_components=None, max_iter='auto', method='infomax', random_state=97, fit_params=dict(extended=True))
    print('fitting ICA') 
    ica.fit(filt_raw)

    ic_labels = label_components(filt_raw, ica, method='iclabel')
    labels = ic_labels['labels']
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ['brain', 'other']]

    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)
    
    print('ica done')

    ## 3. Standard filters
    raw_filtered = reconst_raw.copy() # important to initiate raw_filtered and not continue on the raw_filtered defined earlier
    raw_filtered.filter(l_freq=None, h_freq=80, verbose='CRITICAL') 
    raw_filtered.resample(250, npad="auto", verbose='CRITICAL') 
    linefreq = (50, 100)
    raw_filtered.notch_filter(freqs=linefreq, verbose='CRITICAL') 
    raw_filtered.filter(l_freq=0.1, h_freq=None, verbose='CRITICAL') 

    # identify bad channels, re-reference and interpolate
    noisy_channels2 = pyprep.NoisyChannels(raw_filtered)
    noisy_channels2.find_all_bads()
    bad2 = noisy_channels2.get_bads()
    bad2 = list(map(str, bad2))

    raw_preprocessed = raw_filtered.copy()
    raw_preprocessed.info["bads"] += bad2

    raw_preprocessed.set_eeg_reference(ref_channels="average", verbose='CRITICAL')
    raw_preprocessed.interpolate_bads(verbose='CRITICAL')

    # save preprocessed EEG 
    raw_preprocessed.save(output_dir + "/" + measurement_id + "_prepped.fif") 

    # clean variables to process next file
    del raw, filt_raw, raw_prepped, noisy_channels1, bad1, event_raw, event_dict, markers, raw_cropped, ica, exclude_idx, ic_labels, labels, reconst_raw, raw_filtered, noisy_channels2, bad2, raw_preprocessed

print('all files preprocessed and saved in ouput folder')
# %%
