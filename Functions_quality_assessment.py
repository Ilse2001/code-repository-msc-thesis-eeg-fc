## This code contains custom functions to assess quality of data and cleaning performance. The functions are
#   1. segment_data: segments data from a MNE.RawArray into segments of 1 second per channel
#   2. detect_constant_or_nan: indicates segments with a constant signal or Inf/NaN values in a boolean mask
#   3. detect_extreme_amplitudes: indicates segments with extreme amplitudes in a boolean mask
#   4. detect_high_noise: indicates segments with high noise levels in a boolean mask based on Cristian Kothe's method 
#   5. detect_low_correlation: indicates segments with low correlation (with other channels) in a boolean mask 
#   6. assess_eeg_quality: combines the four criteria to calculate a quality ratio of good segments and a boolean mask of bad segments
#   7. compute_ser_arr: calculates the cleaning performance based on a preprocesed and original file with the metrics: 
#       a. SER (signal to error ratio): power ratio between the original signal and the removed signal based on segments indicated as clean before preprocessing
#       b. ARR (artifact to residue ratio): power ratio between the removed signal and the cleaned signal based on segments indicated as bad before preprocessing
#       c. correlation between the original and the cleaned signal based on the segments indicated as clean before preprocessing
#       d. MI (mutual information) between the removed and the clean signal based on the segments indicated as bad before preprocessing
# The functions can be used in other scripts by using "from Functions_quality_assessment import {name function}"

# import required toolboxes
import mne 
import numpy as np
from scipy.signal import welch
from scipy.stats import median_abs_deviation
from sklearn.feature_selection import mutual_info_regression

# define functions
def segment_data(raw, segment_length =1):
    # segments data from a MNE.RawArray into segments of 1 second per channel
    sfreq = raw.info['sfreq']
    n_samples = int(sfreq * segment_length)
    data = raw.get_data(units='V')
    n_segments = data.shape[1] // n_samples
    segments = data[:, :n_segments * n_samples].reshape(data.shape[0], n_segments, n_samples)
    return segments, sfreq

def detect_constant_or_nan(segments):
    # indicates segments with a constant signal or Inf/NaN values in a boolean mask
    bad_segments = np.zeros(segments.shape[:2], dtype=bool)
    for ch in range(segments.shape[0]):
        for i in range(segments.shape[1]):
            seg = segments[ch, i]
            if np.any(np.isnan(seg)) or np.any(np.isinf(seg)):
                bad_segments[ch, i] = True
            elif np.std(seg) < 1e-10 or median_abs_deviation(seg) < 1e-10:
                bad_segments[ch, i] = True
    return bad_segments

def detect_extreme_amplitudes(segments, runs = 1):
    # indicates segments with extreme amplitudes in a boolean mask
    rSD = median_abs_deviation(segments, axis=2)
    rSD_median = np.median(rSD)
    rSD_mad = median_abs_deviation(rSD)
    z_scores = (rSD - rSD_median)/rSD_mad
    bad_segments = (np.abs(z_scores) > 3) | (np.abs(segments) > 150e-6).any(axis=2)
    
    while runs > 1:
        rSD_median = np.median(rSD[bad_segments==0]) 
        rSD_mad = median_abs_deviation(rSD[bad_segments==0])
        z_scores = (rSD - rSD_median)/rSD_mad
        bad_segments = (np.abs(z_scores) > 3) | (np.abs(segments) > 150e-6).any(axis=2)
        runs = runs - 1
    
    return bad_segments

def detect_high_noise(segments, sfreq, runs = 1):
    # indicates segments with high noise levels in a boolean mask based on Cristian Kothe's method 
    def compute_rNSR(segment):
        f, Pxx = welch(segment, sfreq, nperseg=len(segment))
        noise_power = np.sum(Pxx[f>40])
        signal_power = np.sum(Pxx[f<40])
        return noise_power/(signal_power + 1e-10)

    rNSR = np.apply_along_axis(compute_rNSR, 2, segments)
    rNSR_medain = np.median(rNSR)
    rNSR_mad = median_abs_deviation(rNSR)
    z_scores = (rNSR - rNSR_medain)/rNSR_mad
    bad_segments = (rNSR > 0.5) | (z_scores > 5)

    while runs > 1:
        rNSR_medain = np.median(rNSR[bad_segments==0])
        rNSR_mad = median_abs_deviation(rNSR[bad_segments==0])
        z_scores = (rNSR - rNSR_medain)/rNSR_mad
        bad_segments = (rNSR > 0.5) | (z_scores > 5)
        runs = runs -1

    return bad_segments

def detect_low_correlation(segments):
    # indicates segments with low correlation (with other channels) in a boolean mask 
    bad_segments = np.zeros(segments.shape[:2], dtype=bool)
    for i in range(segments.shape[1]):
        corr_matrix = np.corrcoef(segments[:,i,:])
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, np.nan)
        max_corr = np.nanpercentile(abs_corr, 98, axis=1)
        bad_segments[:,i] = max_corr < 0.6
    return bad_segments

def assess_eeg_quality(raw, runs = 1):
    # combines the four criteria to calculate a quality ratio of good segments and a boolean mask of bad segments
    segments, sfreq = segment_data(raw)
    bad_constant = detect_constant_or_nan(segments)
    bad_amplitude = detect_extreme_amplitudes(segments, runs = runs)
    bad_noise = detect_high_noise(segments, sfreq, runs = runs)
    bad_correlation = detect_low_correlation(segments)
    bad_segments = np.logical_or.reduce([bad_constant, bad_amplitude, bad_noise, bad_correlation])
    total_segments = segments.shape[1] * segments.shape[0]
    quality_ratio = 1 - np.sum(bad_segments)/total_segments
    return{
        'bad_constant': bad_constant, 
        'bad_amplitude': bad_amplitude, 
        'bad_noise': bad_noise, 
        'bad_correlation': bad_correlation, 
        'bad_segments': bad_segments, 
        'quality_ratio': quality_ratio
    }
    target_sfreq = min(raw_eeg.info['sfreq'], cleaned_eeg.info['sfreq'])

    if raw_eeg.info['sfreq'] > target_sfreq:
        raw_eeg = raw_eeg.resample(target_sfreq)

    raw_segments, _ = segment_data(raw_eeg)
    cleaned_segments, _ = segment_data(cleaned_eeg)
    raw_segments = raw_segments.real
    cleaned_segments = cleaned_segments.real
    removed_signal = raw_segments - cleaned_segments

    clean_periods = ~bad_segments
    artifact_periods = bad_segments

    SER_channels = []
    power_noise_channels = []
    ARR_channels = []
    SER_corr = []
    ARR_mi = []

    for ch in range(raw_segments.shape[0]):
        SER = []
        ARR = []
        power_cs = []
        power_as = []
        for i in range(raw_segments.shape[1]):
            power_clean = np.mean(cleaned_segments[ch, i] ** 2)
            power_noise = np.mean(removed_signal[ch, i] ** 2)
            power_raw = np.mean(raw_segments[ch, i] ** 2)
            if clean_periods[ch, i] == True:
                #bereken orginele SER
                ser_value = 10 * np.log10(power_raw / power_noise)
                SER.append(ser_value)
                #bereken SER met correlatie
                corr_value = np.corrcoef(raw_segments[ch, i], cleaned_segments[ch, i])[0,1]
                SER_corr.append(corr_value)
                #power
                power_cs.append(power_raw)
            if artifact_periods[ch, i] == True:
                #bereken orginele ARR
                arr_value = 10 * np.log10(power_noise / power_clean)
                ARR.append(arr_value)
                #bereken ARR met mutual information
                vector = removed_signal[ch, i].reshape(-1,1)
                mi_value = mutual_info_regression(vector, cleaned_segments[ch, i])
                ARR_mi.append(mi_value)
                #power
                power_as.append(power_raw)
        SER_channels.append(np.nanmean(SER))
        ARR_channels.append(np.nanmean(ARR))
        if len(power_as) > 0:
            if len(power_cs) > 0:
                power_noise_channels.append(np.mean(power_as) - np.mean(power_cs))
            else:
                power_noise_channels.append(np.mean(power_as))
        else:
            power_noise_channels.append(0)
    
    total_artifact_power = np.sum(power_noise_channels)
    weigths = power_noise_channels/total_artifact_power

    weigthed_ser = np.nansum(SER_channels * weigths)
    weigthed_arr = np.nansum(ARR_channels * weigths)

    return weigthed_ser, weigthed_arr, np.mean(SER_corr), np.mean(ARR_mi)

def compute_ser_arr(raw_eeg, cleaned_eeg, bad_segments):
    #calculates the cleaning performance based on a preprocesed and original file with the metrics: SER, ARR, correlation and MI
    target_sfreq = min(raw_eeg.info['sfreq'], cleaned_eeg.info['sfreq'])

    if raw_eeg.info['sfreq'] > target_sfreq:
        raw_eeg = raw_eeg.resample(target_sfreq)

    raw_data = raw_eeg.get_data()
    cleaned_data = cleaned_eeg.get_data()

    min_len = min(raw_data.shape[1], cleaned_data.shape[1])
    raw_data = raw_data[:, :min_len]
    cleaned_data = cleaned_data[:, :min_len]

    raw_eeg = mne.io.RawArray(raw_data, raw_eeg.info)
    cleaned_eeg = mne.io.RawArray(cleaned_data, cleaned_eeg.info)

    raw_segments, _ = segment_data(raw_eeg)
    cleaned_segments, _ = segment_data(cleaned_eeg)
    raw_segments = raw_segments.real
    cleaned_segments = cleaned_segments.real
    removed_signal = raw_segments - cleaned_segments
 

    clean_periods = ~bad_segments
    artifact_periods = bad_segments

    SER_channels = []
    ARR_channels = []
    power_noise_channels = []
    clean_powers = []
    SER_corr = []
    ARR_mi = []

    # calculate metrics per channel
    for ch in range(raw_segments.shape[0]):
        # seperate clean and bad segments
        clean_idx = clean_periods[ch]
        artf_idx = artifact_periods[ch]
        clean_idxs = np.where(clean_periods[ch, :])[0]

        raw_clean = raw_segments[ch][clean_idx]
        cleaned_clean = cleaned_segments[ch][clean_idx]
        removed_clean = removed_signal[ch][clean_idx]

        raw_artf = raw_segments[ch][artf_idx]
        cleaned_artf = cleaned_segments[ch][artf_idx]
        removed_artf = removed_signal[ch][artf_idx]

        

        # signal preservation metrics on clean segments
        if len(raw_clean) > 0:
            power_clean = np.mean(raw_clean**2)
            power_removed = np.mean(removed_clean**2)
            SER_value = 10 * np.log10(power_clean / power_removed) if power_removed > 0 else np.nan
            SER_channels.append(SER_value)

            for i in range(len(raw_clean)):
                corr = np.corrcoef(raw_clean[i], cleaned_clean[i])[0, 1]
                SER_corr.append(corr)
        else:
            SER_channels.append(np.nan)

        # artifact removal metrics on bad segments
        if len(cleaned_artf) > 0:
            power_clean_artf = np.mean(cleaned_artf**2)
            power_removed_artf = np.mean(removed_artf**2)
            ARR_value = 10 * np.log10(power_removed_artf / power_clean_artf) if power_clean_artf > 0 else np.nan
            ARR_channels.append(ARR_value)

            for i in range(len(removed_artf)):
                vector = removed_artf[i].reshape(-1, 1)
                mi = mutual_info_regression(vector, cleaned_artf[i])
                ARR_mi.append(mi)
        else:
            ARR_channels.append(np.nan)

        # calculate weigths for weigthed mean SER and ARR
        power_artf = np.mean(raw_artf**2) if len(raw_artf) > 0 else 0
        power_clean = np.mean(raw_clean**2) if len(raw_clean) > 0 else 0
        noise_contribution = power_artf - power_clean
        power_noise_channels.append(max(noise_contribution, 0)) 

        if len(clean_idxs) > 0:
            clean_power = np.mean([np.mean(raw_segments[ch, i]**2) for i in clean_idxs])
        else:
            clean_power = 0
        clean_powers.append(clean_power)

    # calculate weigthed SER and ARR
    power_noise_channels = np.array(power_noise_channels)
    total_artifact_power = np.sum(power_noise_channels)
    weights = power_noise_channels / total_artifact_power if total_artifact_power > 0 else np.zeros_like(power_noise_channels)

    SER_channels = np.array(SER_channels)
    ARR_channels = np.array(ARR_channels)
    clean_powers = np.array(clean_powers)

    weighted_arr = np.nansum(ARR_channels * weights)
    weights_clean_power = clean_powers / np.sum(clean_powers)
    weighted_ser = np.nansum(SER_channels * weights_clean_power)

    return weighted_ser, weighted_arr, np.nanmean(SER_corr), np.nanmean(ARR_mi)
