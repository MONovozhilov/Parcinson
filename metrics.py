import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from config import NUM_LANGUAGES

def cutmix_spectrograms(spec, spec2, beta):
    bs, c, h, w = spec.shape
    cut_ratio = np.sqrt(1.0 - np.random.beta(beta, beta))
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
    spec_cut = spec.clone()
    spec_cut[:, :, y1:y2, x1:x2] = spec2[:, :, y1:y2, x1:x2]
    return spec_cut, ((x2 - x1) * (y2 - y1)) / (w * h)

def validate_file_level(model, dataset, device):
    model.eval()
    all_logits, all_trues, all_f_idx, all_langs, all_genders = [], [], [], [],[]
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=lambda b: b)
    
    with torch.no_grad():
        for i in range(0, len(dataset), 32):
            batch =[dataset[j] for j in range(i, min(i+32, len(dataset)))]
            s = torch.from_numpy(np.stack([b[0] for b in batch])).unsqueeze(1).to(device)
            a = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
            all_logits.append(model(s, a).cpu())
            all_trues.extend([b[2] for b in batch])
            all_f_idx.extend([b[3] for b in batch])
            all_langs.extend([b[4] for b in batch])
            all_genders.extend([b[5] for b in batch])
            
    logits = torch.cat(all_logits, dim=0)
    trues, f_idx = np.array(all_trues), np.array(all_f_idx)
    langs, genders = np.array(all_langs), np.array(all_genders)
    
    preds, file_trues, probs, file_langs, file_gends = [], [], [], [],[]
    for f in np.unique(f_idx):
        mask = (f_idx == f)
        mean_probs = torch.softmax(logits[mask], dim=1).mean(dim=0)
        preds.append(mean_probs.argmax().item())
        file_trues.append(trues[mask][0])
        probs.append(mean_probs[1].item())
        file_langs.append(langs[mask][0])   # 🆕
        file_gends.append(genders[mask][0])
        
    def get_cm(arr, ids): 
        return {i: confusion_matrix(np.array(file_trues)[arr==i], np.array(preds)[arr==i], labels=[0,1]) for i in ids if sum(arr==i)>0}

    return {
        'accuracy': accuracy_score(file_trues, preds),
        'precision': precision_score(file_trues, preds, zero_division=0),
        'recall': recall_score(file_trues, preds, zero_division=0),
        'f1': f1_score(file_trues, preds, zero_division=0),
        'roc_auc': roc_auc_score(file_trues, probs) if len(set(file_trues)) > 1 else 0.0,
        'cm_overall': confusion_matrix(file_trues, preds, labels=[0, 1]),
        'cm_lang': get_cm(np.array(file_langs), range(NUM_LANGUAGES)),
        'cm_gender': get_cm(np.array(file_gends), range(2))
    }