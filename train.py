import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import random
import os
from pathlib import Path

from best.config import *
from dataset import PreprocessedDataset, collate_fn
from model import HybridModel
from metrics import validate_file_level, cutmix_spectrograms
from visualizations import plot_comprehensive_cms, plot_training_history

def train_and_evaluate(data_dict, trial_params=None, save_artifacts=True):
    if trial_params is None: trial_params = {}
    
    lr = trial_params.get('learning_rate', LEARNING_RATE)
    wd = trial_params.get('weight_decay', WEIGHT_DECAY)
    bs = trial_params.get('batch_size', BATCH_SIZE)
    cm_prob = trial_params.get('cutmix_prob', CUTMIX_PROB)
    cm_beta = trial_params.get('cutmix_beta', CUTMIX_BETA)
    dp_rate = trial_params.get('dropout_rate', DROPOUT_RATE)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []
    
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
        best_epoch_idx = 0
        best_model_path = None
        history = {'train_acc':[], 'val_acc':[]} 
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            correct_train, total_train = 0, 0
            
            for spec, aco, lbl, _, _, _ in train_loader:
                spec, aco, lbl = spec.to(DEVICE), aco.to(DEVICE), lbl.to(DEVICE)
                optimizer.zero_grad()
                
                if USE_CUTMIX and random.random() < cm_prob:
                    idx = torch.randperm(spec.size(0))
                    spec_cut, ratio = cutmix_spectrograms(spec, spec[idx], cm_beta)
                    dom_lbl = lbl if ratio < 0.5 else lbl[idx]
                    logits = model(spec_cut, aco)
                    loss = criterion(logits, dom_lbl)
                    _, predicted = logits.max(1)
                    correct_train += predicted.eq(dom_lbl).sum().item()
                else:
                    logits = model(spec, aco)
                    loss = criterion(logits, lbl)
                    _, predicted = logits.max(1)
                    correct_train += predicted.eq(lbl).sum().item()
                
                total_train += lbl.size(0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            train_acc_epoch = correct_train / total_train
            metrics = validate_file_level(model, val_ds, DEVICE)
            val_acc_epoch = metrics['accuracy']
            
            history['train_acc'].append(train_acc_epoch)
            history['val_acc'].append(val_acc_epoch)
            
            if val_acc_epoch > best_acc + IMPROVEMENT_THRESHOLD:
                if save_artifacts and best_model_path and Path(best_model_path).exists():
                    Path(best_model_path).unlink()
                
                best_acc, best_metrics, best_epoch_idx, patience = val_acc_epoch, metrics, epoch, 0
                
                if save_artifacts and SAVE_RESULTS:
                    best_model_path = f"{RESULTS_DIR}/{best_acc:.2f}_best_model_fold_{fold}.pt"
                    torch.save(model.state_dict(), best_model_path)
            else: 
                patience += 1
            
            scheduler.step(val_acc_epoch)
            
            if save_artifacts and ((epoch+1) % 5 == 0 or patience == 0): 
                mark = "НОВЫЙ МАКСИМУМ" if patience == 0 else ""
                print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc_epoch:.4f} | Val Acc: {val_acc_epoch:.4f} | Patience: {patience}/{EARLY_STOPPING_PATIENCE} {mark}")
                
            if patience >= EARLY_STOPPING_PATIENCE: 
                break
            
        best_metrics['fold'] = fold
        fold_results.append(best_metrics)
        
        if save_artifacts:
            acc_str = f"{best_acc:.2f}"
            plot_comprehensive_cms(best_metrics, f"Fold {fold} (Acc: {acc_str})", f"{acc_str}_cm_fold_{fold}.png")
            plot_training_history(history, fold, f"{acc_str}_history_fold_{fold}.png", best_epoch_idx)
        
    return fold_results