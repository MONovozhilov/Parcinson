import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shutil
from datetime import datetime
from config import NUM_LANGUAGES, LANG_NAMES

def cutmix_spectrograms(spec, spec2, beta=1.0):
    batch_size, channels, h, w = spec.shape
    cut_ratio = np.sqrt(1.0 - np.random.beta(beta, beta))
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    mask = torch.ones((batch_size, channels, h, w), device=spec.device)
    mask[:, :, y1:y2, x1:x2] = 0.0
    spec_cutmix = spec.clone()
    spec_cutmix[:, :, y1:y2, x1:x2] = spec2[:, :, y1:y2, x1:x2]
    return spec_cutmix, mask, cut_ratio

def save_best_model(model_state, scaler, metrics, fold_num, hyperparameters,
                    filepath_template="parkinson_best_fold_{fold_num}.pt"):
    save_path = filepath_template.format(fold_num=fold_num)
    checkpoint = {
        'model_state_dict': model_state,
        'scaler': scaler,
        'metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'fold_num': fold_num
        },
        'hyperparameters': hyperparameters,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'architecture': 'Hybrid CNN-LSTM with acoustic features (123-dim)'
    }
    torch.save(checkpoint, save_path)
    metadata_path = save_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': checkpoint['metrics'],
            'hyperparameters': checkpoint['hyperparameters'],
            'timestamp': checkpoint['timestamp']
        }, f, indent=2, ensure_ascii=False)
    return save_path

def plot_training_history(history, fold_num):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.title(f'Fold {fold_num}: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 4, 2)
    plt.plot(history['train_acc_seg'], label='Train Acc (seg)', marker='o', linewidth=2)
    plt.plot(history['val_acc_file'], label='Val Acc (file)', marker='s', linewidth=2)
    plt.title(f'Fold {fold_num}: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 4, 3)
    plt.plot(history['val_f1_file'], label='Val F1 (file)', marker='^', linewidth=2, color='green')
    plt.title(f'Fold {fold_num}: F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 4, 4)
    plt.plot(history['val_auc_file'], label='Val AUC (file)', marker='d', linewidth=2, color='purple')
    plt.title(f'Fold {fold_num}: ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'training_history_fold_{fold_num}.png', dpi=150, bbox_inches='tight')
    plt.close()

def summarize_cv_results(fold_results, n_splits, num_languages, lang_names):
    accs_file = [r['best_val_acc_file'] for r in fold_results]
    f1s_file = [r['best_val_f1_file'] for r in fold_results]
    aucs_file = [r['best_val_auc_file'] for r in fold_results]
    
    avg_cm_by_lang = {lang_id: [] for lang_id in range(num_languages)}
    for r in fold_results:
        if 'confusion_by_language' in r:
            for lang_id, cm in r['confusion_by_language'].items():
                avg_cm_by_lang[lang_id].append(cm)
    
    fig, axes = plt.subplots(1, num_languages + 1, figsize=(6*(num_languages+1), 5))
    avg_cm = np.zeros((2, 2))
    for r in fold_results:
        avg_cm += r['confusion_matrix_file']
    avg_cm = avg_cm / len(fold_results)
    
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', ax=axes[0],
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    axes[0].set_title('Общая (все языки)')
    axes[0].set_ylabel('Истинный класс')
    axes[0].set_xlabel('Предсказанный класс')
    
    for lang_id in range(num_languages):
        if avg_cm_by_lang[lang_id]:
            lang_cm = np.mean(avg_cm_by_lang[lang_id], axis=0)
            sns.heatmap(lang_cm, annot=True, fmt='.1f', cmap='Blues', ax=axes[lang_id+1],
                        xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
            axes[lang_id+1].set_title(f'{lang_names[lang_id]}')
            axes[lang_id+1].set_ylabel('Истинный класс')
            axes[lang_id+1].set_xlabel('Предсказанный класс')
        else:
            axes[lang_id+1].text(0.5, 0.5, f'Нет данных\nдля {lang_names[lang_id]}',
                                ha='center', va='center', fontsize=12)
            axes[lang_id+1].set_title(f'{lang_names[lang_id]}')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_by_language.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(accs_file, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(accs_file), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {np.mean(accs_file):.4f}')
    plt.title('Распределение точности')
    plt.xlabel('Accuracy')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.hist(f1s_file, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(f1s_file), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {np.mean(f1s_file):.4f}')
    plt.title('Распределение F1-score')
    plt.xlabel('F1 Score')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.hist(aucs_file, bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(aucs_file), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {np.mean(aucs_file):.4f}')
    plt.title('Распределение ROC AUC')
    plt.xlabel('ROC AUC')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cv_metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()