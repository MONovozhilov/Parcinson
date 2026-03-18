import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import random
from pathlib import Path

from config import *
from dataset import PreprocessedDataset, collate_fn
from model import HybridModel
from metrics import validate_file_level, cutmix_spectrograms
from visualizations import plot_comprehensive_cms

def train_and_evaluate(data_dict, trial_params=None, save_artifacts=True):
    # Если параметров нет, берем дефолтные из config.py
    if trial_params is None: trial_params = {}
    
    lr = trial_params.get('learning_rate', LEARNING_RATE)
    wd = trial_params.get('weight_decay', WEIGHT_DECAY)
    bs = trial_params.get('batch_size', BATCH_SIZE)
    cm_prob = trial_params.get('cutmix_prob', CUTMIX_PROB)
    dp_rate = trial_params.get('dropout_rate', 0.3)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results =[]
    
    unique_files = np.unique(data_dict['f_idx'])
    labels = data_dict['labels'][np.searchsorted(data_dict['f_idx'], unique_files)]
    groups = data_dict['p_idx'][np.searchsorted(data_dict['f_idx'], unique_files)]
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(unique_files, labels, groups=groups), 1):
        if save_artifacts:
            print(f"\n{'='*40}\nНАЧАЛО ФОЛДА {fold}/{N_SPLITS}\n{'='*40}")
        
        train_mask = np.isin(data_dict['f_idx'], unique_files[train_idx])
        val_mask = np.isin(data_dict['f_idx'], unique_files[val_idx])
        
        train_ds = PreprocessedDataset(data_dict, train_mask)
        val_ds = PreprocessedDataset(data_dict, val_mask, scaler=train_ds.scaler)
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)
        
        model = HybridModel(dropout_rate=dp_rate).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_acc, patience = 0.0, 0
        best_metrics = None
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            for spec, aco, lbl, _, _, _ in train_loader:
                spec, aco, lbl = spec.to(DEVICE), aco.to(DEVICE), lbl.to(DEVICE)
                optimizer.zero_grad()
                
                if USE_CUTMIX and random.random() < cm_prob:
                    idx = torch.randperm(spec.size(0))
                    spec_cut, ratio = cutmix_spectrograms(spec, spec[idx], CUTMIX_BETA)
                    dom_lbl = lbl if ratio < 0.5 else lbl[idx]
                    loss = criterion(model(spec_cut, aco), dom_lbl)
                else:
                    loss = criterion(model(spec, aco), lbl)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            metrics = validate_file_level(model, val_ds, DEVICE)
            
            if metrics['accuracy'] > best_acc + IMPROVEMENT_THRESHOLD:
                best_acc, best_metrics = metrics['accuracy'], metrics
                patience = 0
                if save_artifacts and SAVE_RESULTS:
                    Path(RESULTS_DIR).mkdir(exist_ok=True)
                    torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model_fold_{fold}.pt")
            else: 
                patience += 1
            
            scheduler.step(metrics['accuracy'])
            
            if save_artifacts and ((epoch+1) % 5 == 0 or patience == 0): 
                mark = "🏆 НОВЫЙ МАКСИМУМ!" if patience == 0 else ""
                print(f"Epoch {epoch+1:3d} | Val Acc: {metrics['accuracy']:.4f} | Patience: {patience}/{EARLY_STOPPING_PATIENCE} {mark}")
                
            if patience >= EARLY_STOPPING_PATIENCE: 
                break
            
        best_metrics['fold'] = fold
        fold_results.append(best_metrics)
        
        if save_artifacts:
            plot_comprehensive_cms(best_metrics, f"Fold {fold}", f"cm_fold_{fold}.png")
        
    return fold_results