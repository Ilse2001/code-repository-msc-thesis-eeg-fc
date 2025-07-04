## Code for functional connectivity analysis in the source space 
# This code will perform a functional connectivity analysis in the source space on EEG measurements with the following steps: 
#   1. Source reconstruction using eLORETA
#   2. Extraction of time courses for anatomical regions (Desikan-Killiany atlas)
#   3. Connectivity analysis using Phase Lag Index (PLI)
#   4. Lobe-to-lobe connectivity analysis
#   5. Hemispheric connectivity analysis
#   6. Minimum Spanning Tree (MST) analysis with diameter and leaf fraction
#   7. Individual Alpha Peak Frequency (iAPF) detection
# This code can be applied to multiple participants and the results will be saved as a .npz file with the following metrics:
#   * subject_id 
#   * connectivity: the full connectivity matrices for both eyes open and closed conditions for each frequency band  
#   * lobe_connectivity: the lobe level connectivity matrices for both eyes open and closed conditions for each frequency band 
#   * hemisphere_connectivity: the left and right intrahemispheric and the interhemishperic PLI for both eyes open and closed conditions for each frequency band  
#   * minimum_spanning_tree: the MST with its diameter and leaf fraction for both eyes open and closed conditions for each frequency band  
#   * iapf
# The measurements to include can be given by assigning a folder, also with the option to only analyze the new files in the folder. 

#%% 
# import required toolboxes

import os
import mne
import numpy as np
import pandas as pd
import networkx as nx
import mne_connectivity
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne_connectivity import spectral_connectivity_epochs

from Functions_load_data import extract_segments, identify_periods # custum functions 

#%% 
# input

output_dir = 'FC_results'
EEG_folder = 'Preprocessed_EEGs'

montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

# Define frequency bands
freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 25),
    'gamma': (25, 50),
    'broad': (1, 50)
}

mne.set_log_level(verbose = 'critical')

#%% 
# Define required functions

def calculate_iapf(raw, picks=['E75', 'E70', 'E83'], fmin=4, fmax=20):
    """Calculate individual alpha peak frequency from raw data."""
    # Create PSD
    psd_object = raw.compute_psd(fmin=fmin, fmax=fmax, picks=picks, n_fft = 1024)
    psd = psd_object.get_data()
    freqs = psd_object.freqs

    # Average across selected channels
    psd_mean = psd.mean(axis=0)
    
    # Find peak frequency
    peak_idx = np.argmax(psd_mean)
    peak_freq = freqs[peak_idx]
    
    ''' uncomment to create figure to show the peak
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_mean)
    plt.axvline(peak_freq, color='r', linestyle='--', 
                label=f'Peak: {peak_freq:.2f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Individual Alpha Peak Frequency')
    plt.legend()
    '''

    return peak_freq

def compute_mst_metrics(con_matrix):
    """Compute MST and extract metrics (diameter, leaf fraction)."""
    # Create distance matrix (1 - connectivity)
    np.fill_diagonal(con_matrix, 0)
    distance_matrix = 1 - con_matrix
    
    # Create network graph
    G = nx.from_numpy_array(distance_matrix)
    
    # Compute MST
    mst = nx.minimum_spanning_tree(G)
    
    # Calculate metrics
    diameter = nx.diameter(mst)
    
    # Calculate leaf fraction (nodes with degree 1 / total nodes)
    leaf_nodes = [node for node, degree in dict(mst.degree()).items() if degree == 1]
    leaf_fraction = len(leaf_nodes) / len(mst.nodes())
    
    return mst, diameter, leaf_fraction

def group_labels_by_lobe(labels):
    """Group cortical labels by anatomical lobe."""
    lobe_mapping = {
        'frontal': ['frontal', 'orbitofrontal', 'paracentral', 'pars', 'precentral'],
        'temporal': ['temporal', 'entorhinal', 'fusiform', 'transverse', 'bankssts'],
        'parietal': ['parietal', 'postcentral', 'precuneus', 'supramarginal'],
        'occipital': ['occipital', 'cuneus', 'lingual', 'pericalcarine'],
        'limbic': ['cingulate', 'hippocampal'],
        'insula': ['insula']
    }
    
    # Create a dictionary to store lobe assignments
    label_to_lobe = {}
    
    # Assign each label to a lobe
    for label in labels:
        name = label.name.lower()
        assigned = False
        for lobe, keywords in lobe_mapping.items():
            if any(keyword in name for keyword in keywords):
                label_to_lobe[label.name] = lobe
                assigned = True
                break
        if not assigned:
            label_to_lobe[label.name] = 'other'
    
    return label_to_lobe

def compute_lobe_connectivity(con_matrix, labels, label_to_lobe):
    """Compute lobe-to-lobe connectivity from label connectivity matrix."""
    # Get unique lobes
    lobes = sorted(set(label_to_lobe.values()))
    
    # Create empty lobe connectivity matrix
    lobe_con = np.zeros((len(lobes), len(lobes)))
    lobe_counts = np.zeros((len(lobes), len(lobes)))
    
    # Map label indices to lobe indices
    label_names = [label.name for label in labels]
    
    # Fill lobe connectivity matrix
    for i, label_i in enumerate(label_names):
        for j, label_j in enumerate(label_names):
            if i != j:  # Skip self-connections
                lobe_i = label_to_lobe[label_i]
                lobe_j = label_to_lobe[label_j]
                
                lobe_i_idx = lobes.index(lobe_i)
                lobe_j_idx = lobes.index(lobe_j)
                
                
                lobe_con[lobe_i_idx, lobe_j_idx] += con_matrix[i, j]
                lobe_counts[lobe_i_idx, lobe_j_idx] += 1
    
    # Average by counts (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        lobe_con = np.divide(lobe_con, (0.5*lobe_counts), 
                           where=lobe_counts > 0)
    
    # Replace NaNs with zeros
    lobe_con = np.nan_to_num(lobe_con, 0)

    # add same combinations together and make triangular matrix
    lobe_con_sym = lobe_con + lobe_con.T
    np.fill_diagonal(lobe_con_sym, np.diagonal(lobe_con))
    lobe_con_tri = np.tril(lobe_con_sym, k=0)

    return lobe_con_tri, lobes

def compute_hemisphere_connectivity(con_matrix, labels):
    """Compute within and between hemisphere connectivity."""
    # Determine hemisphere for each label
    label_names = [label.name for label in labels]
    hemispheres = []
    
    for name in label_names:
        if 'lh' in name:
            hemispheres.append('left')
        elif 'rh' in name:
            hemispheres.append('right')
        else:
            hemispheres.append('unknown')
    
    # Create masks for left and right hemisphere
    left_mask = np.array(hemispheres) == 'left'
    right_mask = np.array(hemispheres) == 'right'
    
    # Compute mean connectivity
    left_within = con_matrix[np.ix_(left_mask, left_mask)]
    right_within = con_matrix[np.ix_(right_mask, right_mask)]
    between = np.vstack((con_matrix[np.ix_(left_mask, right_mask)], con_matrix[np.ix_(right_mask, left_mask)]))
    
    # Calculate mean values (excluding diagonal for within-hemisphere)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)
    
    left_within_mean = (np.nansum(left_within) - np.trace(left_within)) / (0.5 * (n_left * (n_left - 1)))
    right_within_mean = (np.nansum(right_within) - np.trace(right_within)) / (0.5 * (n_right * (n_right - 1)))
    between_mean = np.nansum(between) / ( 0.5 * (n_right*n_left))
    
    return {
        'left_within': left_within_mean,
        'right_within': right_within_mean,
        'between': between_mean
    }

# %% Main analysis

eeg_files = os.listdir(EEG_folder)

for eeg_file in eeg_files:

    # Extract subject ID from filename
    subject_id = eeg_file.split('_')[0] + '_' + eeg_file.split('_')[1]
    print(f"Processing file: {subject_id}")

    ''' uncomment to only process new files
    if os.path.exists('output_dir' + "/" + subject_id + "_functional_connectivity_results.npz"):
        print('FC results already exist, continue with next file')
        continue
    '''

    # Load data
    directory = os.path.join(EEG_folder, eeg_file)
    eeg = mne.io.read_raw_fif(directory, preload=True)
    eeg.set_eeg_reference(projection=True)

    ##  1. Source reconstruction using eLORETA
    # Setup source space (using fsaverage) (choose to fetch from MNE website or local copy)
    fs_dir = r"z:\MNE_sample_data\fsaverage" 
    subjects_dir = r"z:\MNE_sample_data"

    subject = "fsaverage"
    trans = "fsaverage"

    src = fs_dir + "/fsaverage-ico-5-src.fif"
    bem = fs_dir + "/fsaverage-5120-5120-5120-bem-sol.fif"

    ''' uncomment if MNE templates are not locally availabel (then comment lines above)
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)

    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    '''

    # Create forward solution
    fwd = mne.make_forward_solution(
        eeg.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
    )
        
    # Compute noise covariance matrix
    noise_cov = mne.compute_raw_covariance(eeg, tmin=0, tmax=None, method='shrunk')

    # Create inverse operator
    inverse_operator = make_inverse_operator(
        eeg.info, fwd, noise_cov, loose=0.2, depth=0.8
    )

    # Identify eyes open and eyes closed periods
    open_periods, closed_periods = identify_periods(eeg)

    # Extract data segments
    data_open = extract_segments(eeg, open_periods)
    data_closed = extract_segments(eeg, closed_periods)

    # Create raw arrays from segments
    raw_open = mne.io.RawArray(data_open, eeg.info)
    raw_closed = mne.io.RawArray(data_closed, eeg.info)

    # Create epochs
    epochs_open = mne.make_fixed_length_epochs(raw_open, duration=1.0, preload=True)
    epochs_closed = mne.make_fixed_length_epochs(raw_closed, duration=1.0, preload=True)

    # Apply inverse solution to get source estimates
    stc_open = apply_inverse_epochs(epochs_open, inverse_operator, lambda2=1./9., 
                                    method='eLORETA', pick_ori=None)
    stc_closed = apply_inverse_epochs(epochs_closed, inverse_operator, lambda2=1./9., 
                                        method='eLORETA', pick_ori=None)

    # Load source space
    src = mne.read_source_spaces(src)

    ##  2. Extraction of time courses for anatomical regions (Desikan-Killiany atlas)
    # Get cortical labels from Desikan-Killiany atlas
    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    labels = [label for label in labels if 
                np.any([v in s['vertno'] for s in src for v in label.vertices])]
    label_names = [label.name for label in labels]

    # Extract time courses for each label
    label_ts_open = mne.extract_label_time_course(
        stc_open, labels, src, mode='mean_flip', return_generator=False
    )
    label_ts_closed = mne.extract_label_time_course(
        stc_closed, labels, src, mode='mean_flip', return_generator=False
    )

    ##  3. Connectivity analysis using Phase Lag Index (PLI)
    # Calculate PLI connectivity for both conditions and all frequency bands
    con_results = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        # Eyes open
        con_open = spectral_connectivity_epochs(
            label_ts_open, method='pli', mode='multitaper', 
            sfreq=eeg.info["sfreq"], fmin=fmin, fmax=fmax, 
            faverage=True, mt_adaptive=True, n_jobs=1, verbose='critical'
        )
        con_data_open = con_open.get_data(output='dense')[:, :, 0]  # Get the first (only) freq band
        
        # Eyes closed
        con_closed = spectral_connectivity_epochs(
            label_ts_closed, method='pli', mode='multitaper', 
            sfreq=eeg.info["sfreq"], fmin=fmin, fmax=fmax, 
            faverage=True, mt_adaptive=True, n_jobs=1, verbose='critical'
        )
        con_data_closed = con_closed.get_data(output='dense')[:, :, 0]  # Get the first (only) freq band

        con_results[f"{band_name}_open"] = con_data_open
        con_results[f"{band_name}_closed"] = con_data_closed

    ##  4. Lobe-to-lobe connectivity analysis
    # Group labels by lobe
    label_to_lobe = group_labels_by_lobe(labels)

    # Compute lobe connectivity for all bands (open and closed)
    lobe_conn_results = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        
        # Eyes open
        lobe_conn_open, lobes = compute_lobe_connectivity(
            con_results[f"{band_name}_open"], labels, label_to_lobe
        )
        
        # Eyes closed
        lobe_conn_closed, lobes = compute_lobe_connectivity(
            con_results[f"{band_name}_closed"], labels, label_to_lobe
        )
        
        lobe_conn_results[f"{band_name}_open"] = lobe_conn_open
        lobe_conn_results[f"{band_name}_closed"] = lobe_conn_closed

    lobe_conn_results['lobes'] = lobes

    ##  5. Hemispheric connectivity analysis
    # Compute hemisphere connectivity for all bands
    hemi_conn_results = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        
        # Eyes open
        hemi_conn_open = compute_hemisphere_connectivity(
            con_results[f"{band_name}_open"], labels
        )
        
        # Eyes closed
        hemi_conn_closed = compute_hemisphere_connectivity(
            con_results[f"{band_name}_closed"], labels
        )

        hemi_conn_results[f"{band_name}_open"] = hemi_conn_open
        hemi_conn_results[f"{band_name}_closed"] = hemi_conn_closed

    ##  6. Minimum Spanning Tree (MST) analysis with diameter and leaf fraction
    # Compute MST metrics for all bands
    mst_results = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        
        # Eyes open
        mst_open, dm_open, lf_open = compute_mst_metrics(
            con_results[f"{band_name}_open"].copy()
        )
        
        # Eyes closed
        mst_closed, dm_closed, lf_closed = compute_mst_metrics(
            con_results[f"{band_name}_closed"].copy()
        )
        
        mst_results[f"{band_name}_open"] = {'mst':mst_open, 'diameter':dm_open, 'leaf_fraction':lf_open}
        mst_results[f"{band_name}_closed"] = {'mst':mst_closed, 'diameter':dm_closed, 'leaf_fraction':lf_closed}

    ##  7. Individual Alpha Peak Frequency (iAPF) detection
    # Calculate iAPF
    iapf = calculate_iapf(raw_closed)

    # Save results
    output_file = os.path.join(output_dir, f"{subject_id}_functional_connectivity_results.npz")
    np.savez(
        output_file,
        subject_id=subject_id,
        connectivity={k: v for k, v in con_results.items()},
        lobe_connectivity={k: v for k, v in lobe_conn_results.items()},
        hemisphere_connectivity={k: v for k, v in hemi_conn_results.items()},
        minimum_spanning_tree={k: v for k, v in mst_results.items()},
        iapf=iapf,
    )

    # Create results summary for display
    summary = {
        'Subject ID': subject_id,
        'Individual Alpha Peak Frequency': f"{iapf:.2f} Hz",
        'MST Diameter (Eyes Open) (1 - 50 Hz)': mst_results['broad_open']['diameter'],
        'MST Diameter (Eyes Closed) (1 - 50 Hz)': mst_results['broad_closed']['diameter'],
        'MST Leaf Fraction (Eyes Open) (1 - 50 Hz)': f"{mst_results['broad_open']['leaf_fraction']:.2f}",
        'MST Leaf Fraction (Eyes Closed) (1 - 50 Hz)': f"{mst_results['broad_closed']['leaf_fraction']:.2f}",
        'Interhemispheric Connectivity (Eyes Open) (Alpha)': f"{hemi_conn_results['alpha_open']['between']:.4f}",
        'Interhemispheric Connectivity (Eyes Closed) (Alpha)': f"{hemi_conn_results['alpha_closed']['between']:.4f}"
    } # adjustable to preferences

    print("\nResults Summary:") # comment if not required
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_file}")

print("\nAnalysis completed successfully!")
# %%
