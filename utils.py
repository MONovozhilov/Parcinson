import torch
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from config import (
    SEGMENT_DURATION, SAMPLE_RATE, N_MELS, ACOUSTIC_FEATURE_SIZE,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, USE_CUTMIX, CUTMIX_PROB,
    NUM_EPOCHS, EARLY_STOPPING_PATIENCE, N_SPLITS, LANG_NAMES, NUM_LANGUAGES
)


def save_model_checkpoint(model_state, scaler, metrics, fold_num, hyperparameters, filepath_template):
    """
    Сохраняет чекпоинт модели
    
    Аргументы:
        model_state: dict, state_dict модели
        scaler: StandardScaler, скейлер признаков
        metrics: dict, метрики модели
        fold_num: int, номер фолда
        hyperparameters: dict, гиперпараметры
        filepath_template: str, шаблон имени файла
    
    Возвращает:
        str, путь к сохраненному файлу
    """
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


def load_model(checkpoint_path, device=torch.device('cpu')):
    """
    Загружает модель из чекпоинта
    
    Аргументы:
        checkpoint_path: str, путь к чекпоинту
        device: torch.device, устройство для загрузки
    
    Возвращает:
        tuple: (model, scaler, metrics)
    """
    from model import HybridModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = HybridModel(
        acoustic_feature_size=ACOUSTIC_FEATURE_SIZE,
        n_mels=N_MELS,
        target_length=437
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    metrics = checkpoint['metrics']
    
    return model, scaler, metrics


def plot_training_history(history, fold_num):
    """
    Визуализирует историю обучения
    
    Аргументы:
        history: dict, история обучения
        fold_num: int, номер фолда
    """
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.title(f'Fold {fold_num}: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(history['train_acc'], label='Train Acc (seg)', marker='o', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc (file)', marker='s', linewidth=2)
    plt.title(f'Fold {fold_num}: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(history['val_f1'], label='Val F1 (file)', marker='^', linewidth=2, color='green')
    plt.title(f'Fold {fold_num}: F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 4, 4)
    plt.plot(history['val_auc'], label='Val AUC (file)', marker='d', linewidth=2, color='purple')
    plt.title(f'Fold {fold_num}: ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_fold_{fold_num}.png', dpi=150, bbox_inches='tight')
    plt.close()


def summarize_cv_results(fold_results):
    """
    Анализирует и визуализирует результаты кросс-валидации
    
    Аргументы:
        fold_results: list of dicts, результаты по фолдам
    """
    accs = [r['best_val_acc'] for r in fold_results]
    f1s = [r['best_val_f1'] for r in fold_results]
    aucs = [r['best_val_auc'] for r in fold_results]
    
    print(f"\nИТОГОВЫЕ РЕЗУЛЬТАТЫ {N_SPLITS}-FOLD КРОСС-ВАЛИДАЦИИ")
    print(f"Среднее: Acc={np.mean(accs):.4f} ± {np.std(accs):.4f} | "
          f"F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
          f"AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    # Распределение метрик
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(accs, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(accs), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {np.mean(accs):.4f}')
    plt.title('Распределение точности')
    plt.xlabel('Accuracy')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(f1s, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(f1s), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {np.mean(f1s):.4f}')
    plt.title('Распределение F1-score')
    plt.xlabel('F1 Score')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(aucs, bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(aucs), color='red', linestyle='--', linewidth=2,
                label=f'Среднее: {np.mean(aucs):.4f}')
    plt.title('Распределение ROC AUC')
    plt.xlabel('ROC AUC')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cv_metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Усреднённая матрица ошибок
    avg_cm = np.zeros((2, 2))
    for r in fold_results:
        avg_cm += r['confusion_matrix']
    avg_cm = avg_cm / len(fold_results)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    plt.title('Усреднённая матрица ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig('confusion_matrix_avg.png', dpi=150, bbox_inches='tight')
    plt.close()


def select_global_best_model(fold_results, output_path="parkinson_global_best.pt"):
    """
    Выбирает и сохраняет глобально лучшую модель
    
    Аргументы:
        fold_results: list of dicts, результаты по фолдам
        output_path: str, путь для сохранения
    
    Возвращает:
        tuple: (best_fold_info, output_path)
    """
    best_fold = max(fold_results, key=lambda x: x['best_val_acc'])
    
    checkpoint = {
        'model_state_dict': best_fold['best_model_state'],
        'scaler': best_fold['best_scaler'],
        'metrics': best_fold['best_metrics'],
        'fold_num': best_fold['fold_num'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(checkpoint, output_path)
    
    return best_fold, output_path