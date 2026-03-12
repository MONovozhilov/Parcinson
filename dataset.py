import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config import DEVICE

class PreprocessedSegmentDataset(Dataset):
    def __init__(self, segments_specs, segments_acoustics, segments_labels,
                 segments_file_indices, segments_patient_ids, segments_language_ids,
                 segment_mask=None, scaler=None):
        if segment_mask is not None:
            self.segments_specs = segments_specs[segment_mask]
            self.segments_acoustics_raw = segments_acoustics[segment_mask]
            self.segments_labels = segments_labels[segment_mask]
            self.segments_file_indices = segments_file_indices[segment_mask]
            self.segments_patient_ids = segments_patient_ids[segment_mask]
            self.segments_language_ids = segments_language_ids[segment_mask]
        else:
            self.segments_specs = segments_specs
            self.segments_acoustics_raw = segments_acoustics
            self.segments_labels = segments_labels
            self.segments_file_indices = segments_file_indices
            self.segments_patient_ids = segments_patient_ids
            self.segments_language_ids = segments_language_ids
            
        if scaler is None:
            self.scaler = StandardScaler()
            self.segments_acoustics = self.scaler.fit_transform(self.segments_acoustics_raw)
        else:
            self.scaler = scaler
            self.segments_acoustics = self.scaler.transform(self.segments_acoustics_raw)

    def __len__(self):
        return len(self.segments_specs)

    def __getitem__(self, idx):
        return (
            self.segments_specs[idx].copy(),
            self.segments_acoustics[idx].copy(),
            self.segments_labels[idx],
            self.segments_file_indices[idx],
            self.segments_language_ids[idx]
        )

def collate_fn(batch):
    specs = torch.from_numpy(np.stack([item[0] for item in batch], axis=0)).unsqueeze(1)
    acoustics = torch.from_numpy(np.stack([item[1] for item in batch], axis=0))
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    file_indices = torch.tensor([item[3] for item in batch], dtype=torch.long)
    lang_ids = torch.tensor([item[4] for item in batch], dtype=torch.long)
    return specs, acoustics, labels, file_indices, lang_ids