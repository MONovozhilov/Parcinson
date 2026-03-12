import pandas as pd
import numpy as np
import re
import librosa
from pathlib import Path
from config import DATA_ROOT, LANGUAGE_KEYWORDS

def detect_language_group_from_filename(filename):
    filename_lower = filename.lower()
    for lang_id, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return lang_id
    return 0

def load_data_records():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Директория не найдена: {DATA_ROOT}")
    
    df_records = []
    patient_ids = []
    
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
                pass
    
    df = pd.DataFrame(df_records)
    return df, patient_ids