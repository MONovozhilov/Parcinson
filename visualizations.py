import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import LANG_NAMES, GENDER_NAMES, SAVE_RESULTS, RESULTS_DIR

def check_save(filepath):
    if not SAVE_RESULTS: return False
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    return Path(RESULTS_DIR) / filepath

def plot_training_history(history, fold, filename, best_epoch):
    filepath = check_save(filename)
    if not filepath: return
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue', alpha=0.6, linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.title(f'График обучения - Фолд {fold}')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (Accuracy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_comprehensive_cms(metrics, title_prefix, filename):
    filepath = check_save(filename)
    if not filepath: return

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Матрицы ошибок: {title_prefix}', fontsize=16)

    def draw_cm(ax, cm, title):
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', ax=ax, xticklabels=['Control','PD'], yticklabels=['Control','PD'])
        else: ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        ax.set_title(title)
        ax.set_ylabel('Истинный')
        ax.set_xlabel('Предсказанный')

    draw_cm(axes[0, 1], metrics['cm_overall'], "Общая матрица (Overall)")
    axes[0,0].axis('off'); axes[0,2].axis('off')

    draw_cm(axes[1, 0], metrics['cm_gender'].get(0), GENDER_NAMES[0])
    draw_cm(axes[1, 2], metrics['cm_gender'].get(1), GENDER_NAMES[1])
    axes[1,1].axis('off')

    for i in range(3): 
        draw_cm(axes[2, i], metrics['cm_lang'].get(i), LANG_NAMES[i])

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_metric_distributions(fold_results, filename="metric_distributions.png"):
    filepath = check_save(filename)
    if not filepath: return
    accs =[r['accuracy'] for r in fold_results]
    aucs = [r['roc_auc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, name, col in zip(axes,[accs, aucs, f1s],['Accuracy', 'ROC AUC', 'F1 Score'],['skyblue', 'lightcoral', 'lightgreen']):
        sns.histplot(data, bins=5, kde=True, ax=ax, color=col)
        ax.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.3f}')
        ax.set_title(f'Распределение {name}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def generate_text_report(fold_results, filename="extended_report.txt"):
    filepath = check_save(filename)
    if not filepath: return

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\nОТЧЕТ О КРОСС-ВАЛИДАЦИИ\n" + "="*50 + "\n\n")
        
        accs =[r['accuracy'] for r in fold_results]
        precs = [r['precision'] for r in fold_results]
        recs = [r['recall'] for r in fold_results]
        f1s = [r['f1'] for r in fold_results]
        aucs = [r['roc_auc'] for r in fold_results]

        f.write(f"СРЕДНИЕ МЕТРИКИ (ПО ВСЕМ {len(fold_results)} ФОЛДАМ):\n")
        f.write(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}\n")
        f.write(f"Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}\n")
        f.write(f"F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")
        f.write(f"ROC AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n\n")
        
        for r in fold_results:
            f.write(f"--- FOLD {r['fold']} ---\n")
            f.write(f"Acc: {r['accuracy']:.4f} | Prec: {r['precision']:.4f} | Rec: {r['recall']:.4f} | F1: {r['f1']:.4f} | AUC: {r['roc_auc']:.4f}\n")
            f.write("Общая матрица:\n" + str(r['cm_overall'].tolist()) + "\n")
            f.write("По полу:\n")
            for g, cm in r['cm_gender'].items(): f.write(f"  {GENDER_NAMES[g]}:\n  {cm.tolist()}\n")
            f.write("По языку:\n")
            for l, cm in r['cm_lang'].items(): f.write(f"  {LANG_NAMES[l]}:\n  {cm.tolist()}\n")
            f.write("\n")