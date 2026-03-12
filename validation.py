import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from config import DEVICE, NUM_LANGUAGES

def validate_file_level(model, dataset, device=DEVICE, batch_size=32):
    model.eval()
    all_specs = torch.from_numpy(dataset.segments_specs).unsqueeze(1).to(device, non_blocking=True)
    all_acoustics = torch.from_numpy(dataset.segments_acoustics).to(device, non_blocking=True)
    all_labels = torch.from_numpy(dataset.segments_labels).to(device, non_blocking=True)
    all_file_indices = torch.from_numpy(dataset.segments_file_indices).to(device, non_blocking=True)
    all_lang_ids = torch.from_numpy(dataset.segments_language_ids).to(device, non_blocking=True)
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(all_specs), batch_size):
            batch_specs = all_specs[i:i+batch_size]
            batch_acoustics = all_acoustics[i:i+batch_size]
            logits = model(batch_specs, batch_acoustics)
            all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0).cpu()
    all_labels = all_labels.cpu()
    all_file_indices = all_file_indices.cpu()
    all_lang_ids = all_lang_ids.cpu()
    
    unique_file_indices = torch.unique(all_file_indices)
    file_predictions = []
    file_true_labels = []
    file_probabilities = []
    file_language_ids = []
    
    for file_idx in unique_file_indices:
        mask = (all_file_indices == file_idx)
        file_logits = all_logits[mask]
        file_label = all_labels[mask][0].item()
        file_lang_id = all_lang_ids[mask][0].item()
        probs = torch.softmax(file_logits, dim=1)
        mean_probs = probs.mean(dim=0)
        pred_class = mean_probs.argmax().item()
        file_predictions.append(pred_class)
        file_true_labels.append(file_label)
        file_probabilities.append(mean_probs[1].item())
        file_language_ids.append(file_lang_id)
        
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    confusion_by_language = {}
    for lang_id in range(NUM_LANGUAGES):
        lang_mask = np.array(file_language_ids) == lang_id
        if lang_mask.sum() > 0:
            lang_true = np.array(file_true_labels)[lang_mask]
            lang_pred = np.array(file_predictions)[lang_mask]
            confusion_by_language[lang_id] = confusion_matrix(lang_true, lang_pred)
            
    return {
        'accuracy': accuracy_score(file_true_labels, file_predictions),
        'precision': precision_score(file_true_labels, file_predictions, zero_division=0),
        'recall': recall_score(file_true_labels, file_predictions, zero_division=0),
        'f1': f1_score(file_true_labels, file_predictions, zero_division=0),
        'roc_auc': roc_auc_score(file_true_labels, file_probabilities) if len(set(file_true_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(file_true_labels, file_predictions),
        'confusion_by_language': confusion_by_language,
        'predictions': file_predictions,
        'true_labels': file_true_labels,
        'probabilities': file_probabilities,
        'language_ids': file_language_ids
    }