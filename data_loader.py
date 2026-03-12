import pandas as pd
import numpy as np
import librosa
import re
from pathlib import Path
import time
from feature_extractor import extract_acoustic_features_for_segment, extract_spectrogram_for_segment
from config import DATA_ROOT, SEGMENT_DURATION, SAMPLE_RATE, TARGET_FRAMES

def detect_language_group_from_filename(filename):
    """Определение языковой группы по имени файла"""
    from config import LANGUAGE_KEYWORDS
    
    filename_lower = filename.lower()
    for lang_id, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return lang_id
    return 0


def split_audio_into_segments(filepath, segment_duration=SEGMENT_DURATION, sr_target=SAMPLE_RATE):
    """
    Разбивает аудио файл на сегменты фиксированной длительности
    
    Аргументы:
        filepath: str, путь к файлу
        segment_duration: float, длительность сегмента в секундах
        sr_target: int, целевая частота дискретизации
    
    Возвращает:
        list of numpy arrays, список сегментов
    """
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


def load_and_prepare_dataframe():
    """
    Загружает данные из директории и создает датафрейм
    
    Возвращает:
        pandas.DataFrame с информацией о файлах
    """
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Директория не найдена: {DATA_ROOT}")
    
    df_records = []
    patient_ids = []
    language_stats = {0: 0, 1: 0, 2: 0}
    
    for label, folder_name in [("PD", "Болезнь Паркинсона_Parkinson's disease (PD)"),
                                ("Control", "Контроль_Control (C)")]:
        folder = DATA_ROOT / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder}")
        
        wav_files = list(folder.glob("*.wav"))
        
        for wav_file in wav_files:
            try:
                match = re.match(r'(\d+)(?:\(\d+\))?_(PD\d+|C)_(Male|Female)(?:_(nolevodopa|off|on))?_(\w+)\.wav$',
                                wav_file.name, re.I)
                patient_id = int(match.group(1)) if match else hash(wav_file.name) % 10000
                
                language_id = detect_language_group_from_filename(wav_file.name)
                language_stats[language_id] += 1
                
                sr = librosa.get_samplerate(wav_file)
                duration = librosa.get_duration(path=wav_file)
                
                df_records.append({
                    "filepath": str(wav_file),
                    "filename": wav_file.name,
                    "label": label,
                    "sr": sr,
                    "duration": duration,
                    "patient_id": patient_id,
                    "language_id": language_id
                })
                patient_ids.append(patient_id)
            
            except Exception as e:
                continue
    
    df = pd.DataFrame(df_records)
    return df, language_stats


def preprocess_all_files_once(filepaths, labels, patient_ids, language_ids):
    """
    Предварительная обработка всех файлов: сегментация, извлечение спектрограмм и акустических признаков
    
    Аргументы:
        filepaths: list, пути к файлам
        labels: list, метки классов
        patient_ids: list, ID пациентов
        language_ids: list, ID языковых групп
    
    Возвращает:
        dict с предобработанными данными
    """
    segments_specs = []
    segments_acoustics = []
    segments_labels = []
    segments_file_indices = []
    segments_patient_ids = []
    segments_language_ids = []
    
    original_file_labels = np.array(labels)
    original_file_paths = np.array(filepaths)
    original_patient_ids = np.array(patient_ids)
    original_language_ids = np.array(language_ids)
    
    total_segments = 0
    skipped_short = 0
    start_time = time.time()
    
    for file_idx, (filepath, label, pid, lid) in enumerate(zip(filepaths, labels, patient_ids, language_ids)):
        segments = split_audio_into_segments(filepath)
        if len(segments) == 0:
            skipped_short += 1
            continue
        
        for segment in segments:
            spec = extract_spectrogram_for_segment(segment, target_frames=TARGET_FRAMES)
            segments_specs.append(spec)
            
            acoustic_feat = extract_acoustic_features_for_segment(segment)
            segments_acoustics.append(acoustic_feat)
            
            segments_labels.append(label)
            segments_file_indices.append(file_idx)
            segments_patient_ids.append(pid)
            segments_language_ids.append(lid)
        
        total_segments += len(segments)
    
    segments_specs = np.array(segments_specs, dtype=np.float32)
    segments_acoustics = np.array(segments_acoustics, dtype=np.float32)
    segments_labels = np.array(segments_labels, dtype=np.int64)
    segments_file_indices = np.array(segments_file_indices, dtype=np.int64)
    segments_patient_ids = np.array(segments_patient_ids, dtype=np.int64)
    segments_language_ids = np.array(segments_language_ids, dtype=np.int64)
    
    processing_time = time.time() - start_time
    
    return {
        'segments_specs': segments_specs,
        'segments_acoustics': segments_acoustics,
        'segments_labels': segments_labels,
        'segments_file_indices': segments_file_indices,
        'segments_patient_ids': segments_patient_ids,
        'segments_language_ids': segments_language_ids,
        'original_file_labels': original_file_labels,
        'original_file_paths': original_file_paths,
        'original_patient_ids': original_patient_ids,
        'original_language_ids': original_language_ids,
        'processing_time': processing_time,
        'total_segments': total_segments,
        'skipped_short': skipped_short
    }