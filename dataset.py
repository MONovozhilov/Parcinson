import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import re
import time
from sklearn.preprocessing import StandardScaler
from best.config import *
from features import split_audio_into_segments, extract_spectrogram_for_segment, extract_acoustic_features_for_segment

def detect_language(filename):
    lower = filename.lower()
    for lid, keywords in LANGUAGE_KEYWORDS.items():
        if any(k in lower for k in keywords): return lid
    return 0

def build_dataframe(data_root):
    records, pids =[], set()
    root = Path(data_root)
    for label, folder in[("PD", "Болезнь Паркинсона_Parkinson's disease (PD)"), ("Control", "Контроль_Control (C)")]:
        for w in (root / folder).glob("*.wav"):
            match = re.match(r'(\d+)(?:\(\d+\))?_(PD\d+|C)_(Male|Female)', w.name, re.I)
            pid = int(match.group(1)) if match else hash(w.name) % 10000
            gid = 1 if match and match.group(3).lower() == 'male' else 0
            lid = detect_language(w.name)
            records.append({'filepath': str(w), 'label': label, 'patient_id': pid, 'language_id': lid, 'gender_id': gid})
            pids.add(pid)
    return pd.DataFrame(records)

def preprocess_files(filepaths, labels, patient_ids, language_ids, gender_ids):
    print(f"\n[{time.strftime('%H:%M:%S')}] Начало предобработки ({len(filepaths)} файлов)...")
    res = {k: [] for k in['specs', 'acoustics', 'labels', 'f_idx', 'p_idx', 'l_idx', 'g_idx']}
    for i, (fp, lbl, pid, lid, gid) in enumerate(zip(filepaths, labels, patient_ids, language_ids, gender_ids)):
        for seg in split_audio_into_segments(fp):
            res['specs'].append(extract_spectrogram_for_segment(seg))
            res['acoustics'].append(extract_acoustic_features_for_segment(seg))
            res['labels'].append(lbl)
            res['f_idx'].append(i)
            res['p_idx'].append(pid)
            res['l_idx'].append(lid) 
            res['g_idx'].append(gid)
        if (i+1) % 10 == 0: print(f"  Обработано {i+1}/{len(filepaths)}...", end='\r')
    print("\nПредобработка завершена.")
    return {k: np.array(v, dtype=np.float32 if k in ['specs', 'acoustics'] else np.int64) for k, v in res.items()}

class PreprocessedDataset(Dataset):
    def __init__(self, data_dict, mask=None, scaler=None):
        self.d = {k: v[mask] if mask is not None else v for k, v in data_dict.items() if k != 'original'}
        self.scaler = StandardScaler().fit(self.d['acoustics']) if scaler is None else scaler
        self.d['acoustics'] = self.scaler.transform(self.d['acoustics'])
    def __len__(self): return len(self.d['specs'])
    def __getitem__(self, i):
        return (self.d['specs'][i].copy(), self.d['acoustics'][i].copy(), self.d['labels'][i], 
                self.d['f_idx'][i], self.d['l_idx'][i], self.d['g_idx'][i])

def collate_fn(batch):
    return (torch.from_numpy(np.stack([b[0] for b in batch])).unsqueeze(1),
            torch.from_numpy(np.stack([b[1] for b in batch])),
            torch.tensor([b[2] for b in batch], dtype=torch.long),
            torch.tensor([b[3] for b in batch], dtype=torch.long),
            torch.tensor([b[4] for b in batch], dtype=torch.long),
            torch.tensor([b[5] for b in batch], dtype=torch.long))