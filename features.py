import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from best.config import *

def split_audio_into_segments(filepath, segment_duration=SEGMENT_DURATION, sr_target=SAMPLE_RATE):
    y, sr = librosa.load(filepath, sr=sr_target)
    if y.ndim > 1: y = y[0]
    total_samples = int((len(y) / sr // segment_duration) * segment_duration * sr)
    y = y[:total_samples]
    segment_samples = int(segment_duration * sr)
    return [y[i:i + segment_samples] for i in range(0, len(y), segment_samples) if len(y[i:i + segment_samples]) == segment_samples]

def extract_spectrogram_for_segment(segment, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])), mode='constant')
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]
    return (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

def extract_acoustic_features_for_segment(segment, sr=SAMPLE_RATE):
    features =[]
    try:
        sound = parselmouth.Sound(segment, sampling_frequency=sr)
        
        # MFCC
        if hasattr(sound, 'to_mfcc') and hasattr((mfcc := sound.to_mfcc(number_of_coefficients=13)), 'to_array'):
            mfcc_matrix = mfcc.to_array()
            for i in range(min(13, mfcc_matrix.shape[0])):
                coefs = mfcc_matrix[i, :][~np.isnan(mfcc_matrix[i, :])]
                if len(coefs) > 0:
                    features.extend([np.mean(coefs), np.std(coefs), np.min(coefs), np.max(coefs), np.max(coefs)-np.min(coefs), np.median(coefs)])
                else:
                    features.extend([0.0] * 6)
            features.extend([np.mean(mfcc_matrix), np.std(mfcc_matrix)] if mfcc_matrix.size > 0 else [0.0, 0.0])
        else: features.extend([0.0] * 80)
        
        # Pitch
        pitch_vals = call(sound, "To Pitch", 0.0, 75, 500).selected_array['frequency']
        pitch_vals = pitch_vals[pitch_vals > 0]
        features.extend([np.mean(pitch_vals), np.std(pitch_vals), np.min(pitch_vals), np.max(pitch_vals), np.max(pitch_vals)-np.min(pitch_vals), np.median(pitch_vals)] if len(pitch_vals) > 0 else[0.0]*6)
        
        int_vals = sound.to_intensity().values[0][~np.isnan(sound.to_intensity().values[0])]
        features.extend([np.mean(int_vals), np.std(int_vals), np.min(int_vals), np.max(int_vals), np.max(int_vals)-np.min(int_vals)] if len(int_vals) > 0 else [0.0]*5)
        
        formants = sound.to_formant_burg()
        for f_num in [1, 2, 3]:
            f_vals =[f for t in formants.xs() if not np.isnan(f := formants.get_value_at_time(f_num, t)) and f > 0]
            features.extend([np.mean(f_vals), np.std(f_vals), np.min(f_vals), np.max(f_vals)] if f_vals else [0.0]*4)
        
        spectrum = sound.to_spectrum()
        features.extend([spectrum.get_centre_of_gravity(), spectrum.get_standard_deviation(), spectrum.get_skewness(), spectrum.get_kurtosis()])
        spec_vals, freqs = spectrum.values[0], np.linspace(0, sr/2, len(spectrum.values[0]))
        total_energy = np.sum(spec_vals**2) + 1e-8
        for low, high in[(0,250), (250,500), (500,1000), (1000,2000), (2000,4000)]:
            mask = (freqs >= low) & (freqs < high)
            features.append(np.sum(spec_vals[mask]**2) / total_energy if np.any(mask) else 0.0)
            
        hnr_vals = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0).values[0]
        hnr_vals = hnr_vals[~np.isnan(hnr_vals)]
        features.extend([np.mean(hnr_vals), np.std(hnr_vals), np.min(hnr_vals), np.max(hnr_vals)] if len(hnr_vals) > 0 else [0.0]*4)
        
        features.extend([np.sum(segment**2), np.sqrt(np.mean(segment**2)), np.sum(np.abs(np.diff(np.signbit(segment))))/len(segment), np.max(np.abs(segment)), np.mean(np.abs(segment)), np.std(segment)])
        hist = np.histogram(segment, bins=50, density=True)[0]
        features.append(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10)) if len(hist[hist > 0]) > 0 else 0.0)
        
        features = np.pad(features, (0, max(0, 123 - len(features))), constant_values=0.0)[:123]
    except Exception:
        features = [0.0] * 123
        
    return np.array(features, dtype=np.float32)