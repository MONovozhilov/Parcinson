import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import warnings
warnings.filterwarnings('ignore')

def extract_acoustic_features_for_segment(segment, sr=16000):
    """
    Извлекает 123 акустических признака из аудио сегмента
    
    Аргументы:
        segment: numpy array, аудио сегмент
        sr: int, частота дискретизации
    
    Возвращает:
        numpy array (123,), вектор признаков
    """
    features = []
    
    try:
        sound = parselmouth.Sound(segment, sampling_frequency=sr)
        
        # ===== MFCC (80 признаков) =====
        if hasattr(sound, 'to_mfcc'):
            mfcc = sound.to_mfcc(number_of_coefficients=13)
            if hasattr(mfcc, 'to_array'):
                mfcc_matrix = mfcc.to_array()
                for i in range(min(13, mfcc_matrix.shape[0])):
                    coef_values = mfcc_matrix[i, :]
                    if len(coef_values) > 0 and not np.all(np.isnan(coef_values)):
                        coef_values = coef_values[~np.isnan(coef_values)]
                        features.extend([
                            np.mean(coef_values), np.std(coef_values),
                            np.min(coef_values), np.max(coef_values),
                            np.max(coef_values) - np.min(coef_values),
                            np.median(coef_values)
                        ])
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
        
        # ===== Pitch (6 признаков) =====
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) > 0:
            features.extend([
                np.mean(pitch_values), np.std(pitch_values),
                np.min(pitch_values), np.max(pitch_values),
                np.max(pitch_values) - np.min(pitch_values),
                np.median(pitch_values)
            ])
        else:
            features.extend([0.0] * 6)
        
        # ===== Intensity (5 признаков) =====
        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        intensity_values = intensity_values[~np.isnan(intensity_values)]
        if len(intensity_values) > 0:
            features.extend([
                np.mean(intensity_values), np.std(intensity_values),
                np.min(intensity_values), np.max(intensity_values),
                np.max(intensity_values) - np.min(intensity_values)
            ])
        else:
            features.extend([0.0] * 5)
        
        # ===== Formants F1-F3 (12 признаков) =====
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
                features.extend([
                    np.mean(f_values), np.std(f_values),
                    np.min(f_values), np.max(f_values)
                ])
            else:
                features.extend([0.0] * 4)
        
        # ===== Spectral features (9 признаков) =====
        spectrum = sound.to_spectrum()
        features.extend([
            spectrum.get_centre_of_gravity(),
            spectrum.get_standard_deviation(),
            spectrum.get_skewness(),
            spectrum.get_kurtosis()
        ])
        
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
        
        # ===== HNR (4 признака) =====
        hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_values = hnr.values[0]
        hnr_values = hnr_values[~np.isnan(hnr_values)]
        if len(hnr_values) > 0:
            features.extend([
                np.mean(hnr_values), np.std(hnr_values),
                np.min(hnr_values), np.max(hnr_values)
            ])
        else:
            features.extend([0.0] * 4)
        
        # ===== Temporal features (7 признаков) =====
        features.extend([
            np.sum(segment ** 2),
            np.sqrt(np.mean(segment ** 2)),
            np.sum(np.abs(np.diff(np.signbit(segment)))) / len(segment),
            np.max(np.abs(segment)),
            np.mean(np.abs(segment)),
            np.std(segment),
        ])
        
        # Signal entropy
        hist, _ = np.histogram(segment, bins=50, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            features.append(-np.sum(hist * np.log2(hist + 1e-10)))
        else:
            features.append(0.0)
        
        # Защита от несоответствия размерности
        if len(features) != 123:
            if len(features) < 123:
                features = np.pad(features, (0, 123 - len(features)), mode='constant', constant_values=0.0)
            else:
                features = features[:123]
    
    except Exception as e:
        features = [0.0] * 123
    
    return np.array(features, dtype=np.float32)


def extract_spectrogram_for_segment(segment, sr=16000, n_mels=128,
                                     n_fft=1024, hop_length=256, target_frames=437):
    """
    Извлекает и нормализует мел-спектрограмму из аудио сегмента
    
    Аргументы:
        segment: numpy array, аудио сегмент
        sr: int, частота дискретизации
        n_mels: int, количество мел-фильтров
        n_fft: int, размер окна FFT
        hop_length: int, шаг окна
        target_frames: int, целевое количество фреймов
    
    Возвращает:
        numpy array (n_mels, target_frames), нормализованная мел-спектрограмма
    """
    mel_spec = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    if log_mel.shape[1] < target_frames:
        pad = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')
    else:
        log_mel = log_mel[:, :target_frames]
    
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
    return log_mel