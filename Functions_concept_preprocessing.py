## ## This code contains custom functions to preprocess EEG data. The functions are
#   1. base_filters: applies standard filters including band pass, notch and resampling
#   2. wavelet_enhanced_ica: applies a preprocessing pipeline with wavelet enhanced ica using the functions:
#       a. apply_ica: applies ICA to decompose the signal in sources
#       b. remove_artifact_components: cleans all components with a wavelet filter and inverse transforms the data to electrodes 
#           I. wavelet_decomposition: applies wavelet decomposition on ICA_components
#          II. wavelet_thresholding: determines and applies threshold for wavelet filter per component
#         III. wavelet_reconstruction: applies inverse wavelet decomposition to reconstruct filtered ICA components
#   3. apply_MWF: applies a preprocessing pipeline with multichannel wiener filter using the functions:
#       a. detect_muscle_artifacts: detects epochs with muscle artifacts by the log power slope
#           I. compute_log_power_slope: computes the log power slope
#       b. make_template: makes template of the recording indicating epochs with muscle artifacts
#       c. optimize_template: optimizes artifact template for calculating wiener filter by smoothing and equal distribution
#       d. include_time_lag: includes time lag versions of the signal to capture temporal dynamics for better brain signal reconstruction 
#       e. calculate_cov: calculates the covariance matrices of the brain signal and the noise
#       f. GEVD: applies generalized eigenvalue decomposition to optimize the spatial filter
#       g. make_wiener filter: calculates the multichannel wiener filter based on the signal and noise characteristics 
#       h. apply_wiener: applies the multichannel wiener filter on the EEG data for a cleaned result
# The functions can be used in other scripts by using "from Functions_concept_preprocessing import {name function}"

# Import required toolboxes
import numpy as np
import mne
import pywt
import matplotlib.pyplot as plt
from mne_icalabel import label_components
from mne.preprocessing import ICA
from sklearn.decomposition import FastICA
from mne import EpochsArray
import datetime
import pandas as pd
import os
from scipy.signal import welch
from scipy.linalg import eig 

# Define functions
def base_filters(raw):
    # applies standard filters including band pass, notch and resampling
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=None, h_freq=80, verbose='CRITICAL') #lowpass filter

    raw_filtered.resample(250, npad="auto", verbose='CRITICAL') #downsamplen

    linefreq = (50, 100)
    raw_filtered.notch_filter(freqs=linefreq, verbose='CRITICAL') #notch filter for line noise

    raw_filtered.filter(l_freq=0.1, h_freq=None, verbose='CRITICAL') #highpass filter

    return raw_filtered

def apply_ica(eeg_data, n_components=25):
    # applies ICA to decompose the signal in sources
    ica = FastICA(n_components=n_components, random_state=42)
    sources = ica.fit_transform(eeg_data.T).T
    
    return ica, sources

def wavelet_decomposition(sources, wavelet='db4', level=3):
    # applies wavelet decomposition on ICA_components
    
    coeffs = [pywt.wavedec(sources[i, :], wavelet, level=level) for i in range(sources.shape[0])]
    return coeffs

def wavelet_thresholding(coeffs, threshold_factor=0.5):
    # determines and applies threshold for wavelet filter per component
    def soft_threshold(coeff, threshold):
        return np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)
    
    thresholded_coeffs = []
    for c in coeffs:
        detail_coeffs = c[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # robust estimator
        K = 2 * np.log(len(detail_coeffs)) * sigma

        thresholded_component = [soft_threshold(subband, K) for subband in c]
        thresholded_coeffs.append(thresholded_component)
    
    return thresholded_coeffs

def wavelet_reconstruction(coeffs, wavelet='db4'):
    # applies inverse wavelet decomposition to reconstruct filtered ICA components
    return np.array([pywt.waverec(c, wavelet) for c in coeffs])

def remove_artifact_components(ica, sources, wavelet, level=3):
    # cleans all components with a wavelet filter and inverse transforms the data to electrodes 
    coeffs = wavelet_decomposition(sources, wavelet, level)
    thresholded_coeffs = wavelet_thresholding(coeffs)
    filtered_sources = wavelet_reconstruction(thresholded_coeffs, wavelet)
    
    min_len = min(sources.shape[1], filtered_sources.shape[1])
    sources = sources[:, :min_len]
    filtered_sources = filtered_sources[:, :min_len]
    clean_sources = sources - filtered_sources

    ''' optional visualisation
    info = mne.create_info(ch_names=[f'IC{i}' for i in range(sources.shape[0])],
                        sfreq = 250, 
                        ch_types = 'misc')
    
    ica_sources_raw = mne.io.RawArray(sources, info)
    ica_sources_filtered = mne.io.RawArray(filtered_sources, info)
    ica_sources_clean = mne.io.RawArray(clean_sources, info)

    ica_sources_raw.plot()
    ica_sources_filtered.plot()
    ica_sources_clean.plot()
    '''
    
    return  ica.inverse_transform(clean_sources.T).T

def wavelet_enhanced_ica(raw_eeg, wavelet='db4', level=3, n_components=25):
    # applies a preprocessing pipeline with wavelet enhanced ica combining ICA with wavelet filtering to improve signal preservation

    if isinstance(raw_eeg, mne.Epochs):
        ica_results = []
        epoch_data = raw_eeg.get_data()
        for i, epoch in enumerate(epoch_data):
            ica, sources = apply_ica(epoch, n_components)
            clean_data = remove_artifact_components(ica, sources, wavelet, level)
            ica_results.append(clean_data)
        ica_results = np.array(ica_results)
        info = raw_eeg.info
        cleaned_epochs = EpochsArray(ica_results, info, events=raw_eeg.events, tmin=raw_eeg.tmin)
        return cleaned_epochs


    eeg_data = raw_eeg.get_data()
    ica, sources = apply_ica(eeg_data, n_components)
    clean_data = remove_artifact_components(ica, sources, wavelet, level)
    raw_ica = raw_eeg.copy()
    raw_ica._data = clean_data
    return raw_ica

def compute_log_power_slope(epoch_data, sfreq, fmin=7, fmax=75):
    # computes the log power slope
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=len(epoch_data))
    valid_idx = (freqs >= fmin) & (freqs <=fmax)
    log_freqs = np.log10(freqs[valid_idx])
    log_psd = np.log10(psd[valid_idx])
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    return slope

def detect_muscle_artifacts(data, sfreq):
    # detects epochs with muscle artifacts by the log power slope
    slopes = np.array([[compute_log_power_slope(data[e, ch], sfreq)
                        for ch in range(data.shape[1])]
                        for e in range(data.shape[0])])
    return slopes

def make_template(length, epochs, slopes, sfreq):
    # makes template of the recording indicating epochs with muscle artifacts
    slopes[slopes < -0.59] = np.nan
    slopes = slopes - (-0.59)

    epoch_activity = np.nansum(slopes, axis=1)

    mwf_template_odd = np.zeros(length)
    mwf_template_even = np.zeros(length)

    epoch_times = (epochs.events[:, 0] / sfreq)-(epochs.events[0,0]/sfreq)

    for i, start in enumerate(epoch_times):
        start_idx = int(start*sfreq)
        end_idx = int((start + 1)*sfreq)
        if i % 2 == 0:
            mwf_template_even[start_idx:end_idx] = epoch_activity[i]
        else:
            mwf_template_odd[start_idx:end_idx] = epoch_activity[i]
    
    mwf_template = (mwf_template_even + mwf_template_odd)/2
    
    return mwf_template

def optimize_template(mwf_template, sfreq):
    # optimizes artifact template for calculating wiener filter by smoothing and equal distribution
    artifact_indices = np.where(mwf_template > 0)[0]

    # testing if there is enough clean data to calculate the filter on
    if len(artifact_indices) > 0.5 * len(mwf_template):
        print(len(artifact_indices)/len(mwf_template))
        print('only taking most severe 50 precent as artifact')
        threshold = np.percentile(mwf_template, 50)
        artifact_mask = (mwf_template>=threshold).astype(int)
    else:
        artifact_mask = mwf_template != 0

    # elongating artifact segments if too short to calculate the filter on
    min_samples = int(1.2 * sfreq)
    diff_mask = np.diff(np.concatenate(([0], artifact_mask.astype(int), [0])))
    artifact_starts = np.where(diff_mask == 1)[0]
    artifact_ends = np.where(diff_mask == -1)[0]
    start = artifact_starts
    end = artifact_ends

    for i, start_i in enumerate(artifact_starts):
        if end[i] - start[i] < min_samples:
            pad = (min_samples - (end[i]-start[i])) //2
            if start[i] - pad < 0 | end[i]+pad > len(artifact_mask):
                p = 1
            else:
                artifact_mask[(start[i] - pad):(end[i]+pad)] = 1

    # filling up clean segments as artifact if too short to calculate the filter on
    diff_mask_2 = np.diff(np.concatenate(([0], artifact_mask.astype(int), [0])))
    artifact_starts = np.where(diff_mask_2 == 1)[0]
    artifact_ends = np.where(diff_mask_2 == -1)[0]
    start = artifact_starts
    end = artifact_ends

    for i, start_i in enumerate(artifact_starts):
        if i != len(artifact_starts)-1:
            if start[i+1] - end[i] < min_samples:
                artifact_mask[end[i]:start[i+1]] = 1

    # removing beginning and end if these segments are too short to calculate the filter on
    diff_mask_3 = np.diff(np.concatenate(([0], artifact_mask.astype(int), [0])))
    artifact_starts = np.where(diff_mask_3 == 1)[0]
    artifact_ends = np.where(diff_mask_3 == -1)[0]

    artifact_mask = artifact_mask.astype(float)
    if artifact_starts[0] == 0:
        if artifact_ends[0] < min_samples:
            artifact_mask[0:artifact_ends[0]] = np.nan
    elif artifact_starts[0] < min_samples:
        artifact_mask[0:artifact_starts[0]] = np.nan

    if artifact_ends[-1] == len(artifact_mask):
        if artifact_ends[-1] - artifact_starts[-1] < min_samples:
            artifact_mask[artifact_starts[-1]:] = np.nan
    elif len(artifact_mask) - artifact_ends[-1] < min_samples:
        artifact_mask[artifact_ends[-1]:] = np.nan
    
    return artifact_mask

def include_time_lag(eegdata, M, T, delay):
    # includes time lag versions of the signal to capture temporal dynamics for better brain signal reconstruction 
    M_s = (2*delay + 1) * M
    y_s = np.zeros((M_s, T))

    for tau in range(-delay,delay + 1):
        y_shift = np.roll(eegdata, shift=tau, axis=1)
        if tau > 0:
            y_shift[:,:tau] = 0
        elif tau < 0:
            y_shift[:, tau:] = 0
        start_idx = (tau+delay) * M
        end_idx = start_idx + M
        y_s[start_idx:end_idx, :] = y_shift

    return M_s, y_s

def calculate_cov(y_s, artifact_mask):
    # calculates the covariance matrices of the brain signal and the noise
    Ryy = np.cov(y_s[:, artifact_mask == 0], rowvar=True)
    Rnn = np.cov(y_s[:, artifact_mask == 1], rowvar=True)

    if not np.allclose(Ryy, Ryy.T):
        Ryy = (Ryy + Ryy.T) / 2
    if not np.allclose(Rnn, Rnn.T):
        Rnn = (Rnn + Rnn.T) / 2

    return Ryy, Rnn

def GEVD(Ryy, Rnn):
    # applies generalized eigenvalue decomposition to optimize the spatial filter
    eigenvalues, eigenvectors = eig(Ryy, Rnn)
    real_parts = np.real(eigenvalues)
    sorted_indices = np.argsort(real_parts)[::-1]
    Lambda = eigenvalues[sorted_indices]
    V = eigenvectors[:, sorted_indices]

    Lambda_y = np.dot(V.T.conj(), np.dot(Ryy, V))
    Lambda_n = np.dot(V.T.conj(), np.dot(Rnn, V))

    Delta = Lambda_y - Lambda_n
    return V, Delta, Lambda
 
def make_wiener_filter(M_s, Delta, V, Lambda):
    # calculates the multichannel wiener filter based on the signal and noise characteristics 
    rank_w = M_s - np.sum(np.diag(Delta) < 0)
    Delta.flat[rank_w * (M_s + 1) + np.arange(0, M_s - rank_w) * (M_s + 1)] = 0
    W = V @ np.linalg.inv(Lambda + np.eye(M_s)) @ Delta @ np.linalg.inv(V)
    return W

def apply_wiener(raw, eegdata, W, M_s, delay, M, T):
    # applies the multichannel wiener filter on the EEG data for a cleaned result
    channelmeans = np.mean(eegdata, axis=1, keepdims=True)
    y = eegdata - np.tile(channelmeans, (1,T))

    y_s = np.zeros((M_s, y.shape[1]))

    for tau in range(-delay,delay + 1):
        y_shift = np.roll(y, shift=tau, axis=1)
        if tau > 0:
            y_shift[:,:tau] = 0
        elif tau < 0:
            y_shift[:, tau:] = 0
        start_idx = (tau+delay) * M
        end_idx = start_idx + M
        y_s[start_idx:end_idx, :] = y_shift

    orig_chans = np.arange(M)

    d = np.dot(W[:, orig_chans].T, y_s)
    n = y - d
    n = n + np.tile(channelmeans, (1,T))

    RS_MWF = raw.copy()
    RS_MWF._data = n.real

    return RS_MWF

def apply_MWF(raw):
    # applies a preprocessing pipeline with multichannel wiener filter
    sfreq = raw.info['sfreq']
    epochs = mne.make_fixed_length_epochs(raw, duration=1, overlap=0.5, preload=True)
    data = epochs.get_data()
    eegdata = raw.get_data()
    [M, T] = eegdata.shape
    delay = 8
    slopes = detect_muscle_artifacts(data, sfreq)
    MWF_template = make_template(len(raw.times), epochs, slopes, sfreq)
    artifact_mask = optimize_template(MWF_template, sfreq)
    M_s, y_s = include_time_lag(eegdata, M, T, delay)
    Ryy, Rnn = calculate_cov(y_s, artifact_mask)
    V, Delta, Lambda = GEVD(Ryy, Rnn)
    W = make_wiener_filter(M_s, Delta, V, Lambda)
    RS_MWF = apply_wiener(raw, eegdata, W, M_s, delay, M, T)
    return RS_MWF
