import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import time
import random
import warnings
warnings.filterwarnings('ignore')

from config import (BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, 
                    EARLY_STOPPING_PATIENCE, IMPROVEMENT_THRESHOLD, NUM_WORKERS, 
                    PIN_MEMORY, DEVICE, N_SPLITS, USE_CUTMIX, CUTMIX_PROB)
from dataset import PreprocessedSegmentDataset, collate_fn
from model import HybridModel
from validation import validate_file_level
from utils import save_best_model, cutmix_spectrograms

def train_with_group_cv(preprocessed_data, n_splits=N_SPLITS, random_state=42):
    segments_specs = preprocessed_data['segments_specs']
    segments_acoustics = preprocessed_data['segments_acoustics']
    segments_labels = preprocessed_data['segments_labels']
    segments_file_indices = preprocessed_data['segments_file_indices']
    segments_patient_ids = preprocessed_data['segments_patient_ids']
    segments_language_ids = preprocessed_data['segments_language_ids']
    original_patient_ids = preprocessed_data['original_patient_ids']
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    unique_file_indices = np.unique(segments_file_indices)
    file_labels_for_cv = segments_labels[np.searchsorted(segments_file_indices, unique_file_indices)]
    patient_groups_for_cv = original_patient_ids[unique_file_indices]
    
    hyperparameters = {
        'segment_duration': 7.0,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'cutmix_enabled': USE_CUTMIX
    }
    
    fold_counter = 0
    for train_file_idx, val_file_idx in gkf.split(unique_file_indices, file_labels_for_cv,
                                                  groups=patient_groups_for_cv):
        fold_counter += 1
        
        train_files = unique_file_indices[train_file_idx]
        val_files = unique_file_indices[val_file_idx]
        train_patients = np.unique(original_patient_ids[train_file_idx])
        val_patients = np.unique(original_patient_ids[val_file_idx])
        
        train_segment_mask = np.isin(segments_file_indices, train_files)
        val_segment_mask = np.isin(segments_file_indices, val_files)
        
        train_dataset = PreprocessedSegmentDataset(
            segments_specs, segments_acoustics, segments_labels,
            segments_file_indices, segments_patient_ids, segments_language_ids,
            train_segment_mask
        )
        val_dataset = PreprocessedSegmentDataset(
            segments_specs, segments_acoustics, segments_labels,
            segments_file_indices, segments_patient_ids, segments_language_ids,
            val_segment_mask,
            scaler=train_dataset.scaler
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn
        )
        
        model = HybridModel(acoustic_feature_size=123, n_mels=128).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_val_file_acc = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc_seg': [], 'val_acc_file': [], 'val_f1_file': [], 'val_auc_file': []}
        
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            cutmix_applied = 0
            
            for spec, acoustic, labels_batch, _, _ in train_loader:
                spec = spec.to(DEVICE, non_blocking=PIN_MEMORY)
                acoustic = acoustic.to(DEVICE, non_blocking=PIN_MEMORY)
                labels_batch = labels_batch.to(DEVICE, non_blocking=PIN_MEMORY)
                
                optimizer.zero_grad()
                if USE_CUTMIX and random.random() < CUTMIX_PROB:
                    indices = torch.randperm(spec.size(0))
                    spec2 = spec[indices]
                    labels2 = labels_batch[indices]
                    spec_cutmix, mask, cut_ratio = cutmix_spectrograms(spec, spec2)
                    dominant_labels = labels_batch if cut_ratio < 0.5 else labels2
                    logits = model(spec_cutmix, acoustic)
                    loss = criterion(logits, dominant_labels)
                    _, predicted = logits.max(1)
                    train_correct += predicted.eq(dominant_labels).sum().item()
                    cutmix_applied += 1
                else:
                    logits = model(spec, acoustic)
                    loss = criterion(logits, labels_batch)
                    _, predicted = logits.max(1)
                    train_correct += predicted.eq(labels_batch).sum().item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * labels_batch.size(0)
                train_total += labels_batch.size(0)
            
            train_loss_avg = train_loss / train_total
            train_acc_seg = train_correct / train_total
            
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            
            val_file_metrics = validate_file_level(model, val_dataset, device=DEVICE, batch_size=BATCH_SIZE*2)
            val_acc_file = val_file_metrics['accuracy']
            val_f1_file = val_file_metrics['f1']
            val_auc_file = val_file_metrics['roc_auc']
            
            history['train_loss'].append(train_loss_avg)
            history['train_acc_seg'].append(train_acc_seg)
            history['val_acc_file'].append(val_acc_file)
            history['val_f1_file'].append(val_f1_file)
            history['val_auc_file'].append(val_auc_file)
            
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']
            
            if val_acc_file > best_val_file_acc + IMPROVEMENT_THRESHOLD:
                best_val_file_acc = val_acc_file
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_metrics_file = val_file_metrics
                best_scaler = train_dataset.scaler
                patience_counter = 0
            else:
                patience_counter += 1
            
            scheduler.step(val_acc_file)
            
            if (epoch + 1) % 100 == 0:
                print(f"Fold {fold_counter} Epoch {epoch+1}: Loss={train_loss_avg:.4f}, Val Acc={val_acc_file:.4f}, Time={epoch_time:.1f}s")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
        
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
        model_path = save_best_model(best_model_state, best_scaler, best_metrics_file, fold_counter, hyperparameters)
        
        fold_results.append({
            'fold_num': fold_counter,
            'best_val_acc_file': best_metrics_file['accuracy'],
            'best_val_f1_file': best_metrics_file['f1'],
            'best_val_auc_file': best_metrics_file['roc_auc'],
            'confusion_matrix_file': best_metrics_file['confusion_matrix'],
            'confusion_by_language': best_metrics_file['confusion_by_language'],
            'history': history,
            'model_path': model_path
        })
        
    return fold_results