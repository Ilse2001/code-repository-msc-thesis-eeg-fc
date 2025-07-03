## Code for quality check of functional connectivity analysis by Berger effect
# This code will make three figures per measurement to inspect the differences in functional connectivity between eyes open 
# and eyes closed resting state. This code can be applied to multiple measurements and the figures will be showed/saved as 
# one figure with three subplots per measurement:
#   1. Alpha power spectra for both conditions
#   2. Difference between te conditions in alpha connectivity on lobe level 
#   3. Difference between te conditions in alpha connectivity on hemisphere level 

# all functional connectivity measurements will be extracted from the results file created by the FC_analysis.py script. 
# The measurements to include can be given by a list (the correct results files will be searched for in the assigned folder)
# or all the results files in the assigned folder can be used (also with the option to only analyze the new files). 
# The spectra are calculated from the preprocessed EEG files with the corresponding participant number. 

#%% 
# import required toolboxes

import mne
import os
import numpy as np
import matplotlib.pyplot as plt

from Functions_load_data import extract_segments, identify_periods # custom functions

%matplotlib Tk

#%% 
# input 
EEG_folder = 'Preprocessed_EEGs'
FC_results_folder = 'FC_results'
output_dir = FC_results_folder

measurements = ['123456_01012025', '234567_02012025', '345678_03012025'] # measurements to analyze (include both participant number and date of the exam combined with _)

''' uncomment if all measurements in the folder should be analyzed
all_files_from_folder = os.listdir(FC_results_folder)
measurements = [x for x in all_files_from_folder if x.endswith("functional_connectivity_results.npz")]
'''

montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

#%% 
# main analysis

for measurement in measurements:
    print(f"Processing measurement of {measurement.split('_')[0]} on {measurement.split('_')[1]}")

    # Extract subject ID from filename
    subject_id = measurement.split('_')[0] + '_' + measurement.split('_')[1]

    ''' uncomment if only new files in the folder should be analyzed
    if os.path.exists(ouput_dir + "/" + subject_id + "_difference_open_closed.png"):
        print('Measurement is already analysed, continue with the next one')
        continue
    '''

    # Load data
    EEG_filename = subject_id + '_prepped.fif'
    directory = os.path.join(EEG_folder, EEG_filename)
    eeg = mne.io.read_raw_fif(directory, preload=True)
    eeg.set_eeg_reference(projection=True)

    # Identify eyes open and eyes closed periods
    open_periods, closed_periods = identify_periods(eeg)

    # Extract data segments
    data_open = extract_segments(eeg, open_periods)
    data_closed = extract_segments(eeg, closed_periods)
    
    # Create raw arrays from segments
    raw_open = mne.io.RawArray(data_open, eeg.info)
    raw_closed = mne.io.RawArray(data_closed, eeg.info)

    # compute spectra 
    occipital_picks = ['E75', 'E70', 'E83']
    psd_closed = raw_closed.compute_psd(fmin=4, fmax=20, picks=occipital_picks, n_fft = 1024)
    psd_open = raw_open.compute_psd(fmin=4, fmax=20, picks=occipital_picks, n_fft = 1024)

    psd_array_open = psd_open.get_data()
    freqs = psd_open.freqs
    ch_names = psd_open.info.ch_names
    psd_array_closed = psd_closed.get_data()

    # get FC data
    path = os.path.join(FC_results_folder + "/" + subject_id + "_functional_connectivity_results.npz")
    data = np.load(path, allow_pickle=True)

    lobes = data['lobe_connectivity'].item()['lobes']
    lobe_conn_diff = data['lobe_connectivity'].item()["alpha_open"] -data['lobe_connectivity'].item()["alpha_closed"]

    hem_conn_open = np.array([[data['hemisphere_connectivity'].item()['alpha_open']['left_within'], 0], [data['hemisphere_connectivity'].item()['alpha_open']['between'], data['hemisphere_connectivity'].item()['alpha_open']['right_within']]])
    hem_conn_closed = np.array([[data['hemisphere_connectivity'].item()['alpha_closed']['left_within'], 0], [data['hemisphere_connectivity'].item()['alpha_closed']['between'], data['hemisphere_connectivity'].item()['alpha_closed']['right_within']]])
    hem_conn_diff = hem_conn_open - hem_conn_closed

    # make figure
    plt.figure(figsize=(18, 5))
    vmax = 0.2
    vmin = -vmax

    plt.subplot(131) # 1. Alpha power spectra for both conditions
    plt.title(f'Alpha power spectrum in occipital electrodes')
    plt.plot(freqs, np.mean(psd_array_open, axis=0), label='Eyes open', color='blue')
    plt.plot(freqs, np.mean(psd_array_closed, axis=0), label='Eyes closed', color='orange')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Power Spectral Density ($V^2$/Hz)')
    plt.legend()

    plt.subplot(132) # 2. Difference between te conditions in alpha connectivity on lobe level 

    im3 = plt.imshow(lobe_conn_diff, cmap='RdBu_r', vmin = vmin, vmax = vmax)
    plt.colorbar(im3, label='Difference in Phase lag index (PLI)')
    plt.title('Lobe-level alpha connectivity difference \n (Eyes open - eyes closed)')
    plt.xticks(range(len(lobes)), lobes, rotation=45)
    plt.yticks(range(len(lobes)), lobes)

    plt.subplot(133) # 3. Difference between te conditions in alpha connectivity on hemisphere level 
    im4 = plt.imshow(hem_conn_diff, cmap='RdBu_r', vmin = vmin, vmax = vmax)
    plt.colorbar(im4, label='Difference in Phase lag index (PLI)')
    plt.title('Hemispheric alpha connectivity difference \n (Eyes open - eyes closed)')
    plt.xticks([0,1], ['left', 'right'], rotation=45)
    plt.yticks([0,1], ['left', 'right'])

    plt.tight_layout()
    plt.show() #comment if not required
    plt.savefig(os.path.join(output_dir, f"{subject_id}_open_dicht_new.png")) #comment if not required

