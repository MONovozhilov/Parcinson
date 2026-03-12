import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, SEGMENT_DURATION, TARGET_FRAMES

def extract_acoustic_features_for_segment(segment, sr=SAMPLE_RATE):
    features = []
    try:
        sound = parselmouth.Sound(segment, sampling_frequency=sr)
        
        if hasattr(sound, 'to_mfcc'):
            mfcc = sound.to_mfcc(number_of_coefficients=13)
            if hasattr(mfcc, 'to_array'):
                mfcc_matrix = mfcc.to_array()
                for i in range(min(13, mfcc_matrix.shape[0])):
                    coef_values = mfcc_matrix[i, :]
                    if len(coef_values) > 0 and not np.all(np.isnan(coef_values)):
                        coef_values = coef_values[~np.isnan(coef_values)]
                        features.extend([np.mean(coef_values), np.std(coef_values),
                                         np.min(coef_values), np.max(coef_values),
                                         np.max(coef_values) - np.min(coef_values),
                                         np.median(coef_values)])
                    else:
                        features.extend([0.0] * 6)
                if mfcc_matrix.size > 0:
                    features.extend([np.mean(mfcc_matrix), np.std(mfcc_matrix)])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0] * 80)
        else:
            features.extend([0.0] * 80)

        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) > 0:
            features.extend([np.mean(pitch_values), np.std(pitch_values),
                             np.min(pitch_values), np.max(pitch_values),
                             np.max(pitch_values) - np.min(pitch_values),
                             np.median(pitch_values)])
        else:
            features.extend([0.0] * 6)

        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        intensity_values = intensity_values[~np.isnan(intensity_values)]
        if len(intensity_values) > 0:
            features.extend([np.mean(intensity_values), np.std(intensity_values),
                             np.min(intensity_values), np.max(intensity_values),
                             np.max(intensity_values) - np.min(intensity_values)])
        else:
            features.extend([0.0] * 5)

        formants = sound.to_formant_burg()
        for formant_num in [1, 2, 3]:
            f_values = []
            for t in formants.xs():
                try:
                    f_val = formants.get_value_at_time(formant_num, t)
                    if not np.isnan(f_val) and f_val > 0:
                        f_values.append(f_val)
                except:
                    continue
            if f_values:
                features.extend([np.mean(f_values), np.std(f_values),
                                 np.min(f_values), np.max(f_values)])
            else:
                features.extend([0.0] * 4)

        spectrum = sound.to_spectrum()
        features.extend([spectrum.get_centre_of_gravity(),
                         spectrum.get_standard_deviation(),
                         spectrum.get_skewness(),
                         spectrum.get_kurtosis()])
        spectrum_values = spectrum.values[0]
        n = len(spectrum_values)
        max_freq = sr / 2
        frequencies = np.linspace(0, max_freq, n)
        bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000)]
        total_energy = np.sum(spectrum_values ** 2) + 1e-8
        for low_freq, high_freq in bands:
            band_mask = (frequencies >= low_freq) & (frequencies < high_freq)
            if np.any(band_mask):
                band_energy = np.sum(spectrum_values[band_mask] ** 2)
                features.append(band_energy / total_energy)
            else:
                features.append(0.0)

        hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_values = hnr.values[0]
        hnr_values = hnr_values[~np.isnan(hnr_values)]
        if len(hnr_values) > 0:
            features.extend([np.mean(hnr_values), np.std(hnr_values),
                             np.min(hnr_values), np.max(hnr_values)])
        else:
            features.extend([0.0] * 4)

        features.extend([np.sum(segment ** 2),
                         np.sqrt(np.mean(segment ** 2)),
                         np.sum(np.abs(np.diff(np.signbit(segment)))) / len(segment),
                         np.max(np.abs(segment)),
                         np.mean(np.abs(segment)),
                         np.std(segment)])
        hist, _ = np.histogram(segment, bins=50, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            features.append(-np.sum(hist * np.log2(hist + 1e-10)))
        else:
            features.append(0.0)

        if len(features) != 123:
            if len(features) < 123:
                features = np.pad(features, (0, 123 - len(features)), mode='constant', constant_values=0.0)
            else:
                features = features[:123]
    except Exception:
        features = [0.0] * 123
    
    return np.array(features, dtype=np.float32)

def split_audio_into_segments(filepath, segment_duration=SEGMENT_DURATION, sr_target=SAMPLE_RATE):
    y, sr = librosa.load(filepath, sr=sr_target)
    if y.ndim > 1:
        y = y[0]
    total_duration = len(y) / sr
    num_segments = int(total_duration // segment_duration)
    total_samples = num_segments * int(segment_duration * sr)
    y = y[:total_samples]
    segment_samples = int(segment_duration * sr)
    segments = []
    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]
        if len(segment) == segment_samples:
            segments.append(segment)
    return segments

def extract_spectrogram_for_segment(segment, sr=SAMPLE_RATE, n_mels=N_MELS,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, 
                                              n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    if log_mel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
    return log_mel

def preprocess_all_files_once(filepaths, labels, patient_ids, language_ids):
    segments_specs = []
    segments_acoustics = []
    segments_labels = []
    segments_file_indices = []
    segments_patient_ids = []
    segments_language_ids = []
    
    for file_idx, (filepath, label, pid, lid) in enumerate(zip(filepaths, labels, patient_ids, language_ids)):
        segments = split_audio_into_segments(filepath)
        if len(segments) == 0:
            continue
        for segment in segments:
            spec = extract_spectrogram_for_segment(segment)
            segments_specs.append(spec)
            acoustic_feat = extract_acoustic_features_for_segment(segment)
            segments_acoustics.append(acoustic_feat)
            segments_labels.append(label)
            segments_file_indices.append(file_idx)
            segments_patient_ids.append(pid)
            segments_language_ids.append(lid)
    
    return {
        'segments_specs': np.array(segments_specs, dtype=np.float32),
        'segments_acoustics': np.array(segments_acoustics, dtype=np.float32),
        'segments_labels': np.array(segments_labels, dtype=np.int64),
        'segments_file_indices': np.array(segments_file_indices, dtype=np.int64),
        'segments_patient_ids': np.array(segments_patient_ids, dtype=np.int64),
        'segments_language_ids': np.array(segments_language_ids, dtype=np.int64),
        'original_file_labels': np.array(labels),
        'original_file_paths': np.array(filepaths),
        'original_patient_ids': np.array(patient_ids),
        'original_language_ids': np.array(language_ids)
    }