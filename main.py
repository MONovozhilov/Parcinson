import torch
import numpy as np
import shutil
from pathlib import Path
from config import DATA_ROOT, SAVE_RESULTS, RESULTS_DIR
from dataset import build_dataframe, preprocess_files
from train import train_and_evaluate
from visualizations import plot_metric_distributions, generate_text_report, plot_comprehensive_cms

def main():
    print(f"🚀 Запуск конвейера. ОДНОКРАТНОЕ ОБУЧЕНИЕ СТАРТУЕТ...")
    
    if SAVE_RESULTS:
        if Path(RESULTS_DIR).exists(): 
            shutil.rmtree(RESULTS_DIR)
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    df = build_dataframe(DATA_ROOT)
    data_dict = preprocess_files(
        df['filepath'].tolist(), (df['label'] == 'PD').astype(int).tolist(),
        df['patient_id'].tolist(), df['language_id'].tolist(), df['gender_id'].tolist() # 🆕
    )
    
    fold_results = train_and_evaluate(data_dict, save_artifacts=True)
    
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    print(f"\n🎯 ОБУЧЕНИЕ ЗАВЕРШЕНО: Средняя точность = {mean_acc:.4f}")
        
    mean_str = f"{mean_acc:.2f}" 
    plot_metric_distributions(fold_results, filename=f"{mean_str}_metric_distributions.png")
    generate_text_report(fold_results, filename=f"{mean_str}_extended_report.txt")
    
    avg_metrics = {'cm_overall': np.zeros((2,2)), 'cm_gender': {0: np.zeros((2,2)), 1: np.zeros((2,2))}, 'cm_lang': {0: np.zeros((2,2)), 1: np.zeros((2,2)), 2: np.zeros((2,2))}}
    for r in fold_results:
        avg_metrics['cm_overall'] += r['cm_overall']
        for g in [0,1]: 
            if g in r['cm_gender']: avg_metrics['cm_gender'][g] += r['cm_gender'][g]
        for l in[0,1,2]: 
            if l in r['cm_lang']: avg_metrics['cm_lang'][l] += r['cm_lang'][l] # 🆕
            
    for k in ['cm_overall']: avg_metrics[k] /= len(fold_results)
    for g in [0,1]: avg_metrics['cm_gender'][g] /= len(fold_results)
    for l in[0,1,2]: avg_metrics['cm_lang'][l] /= len(fold_results) # 🆕
        
    plot_comprehensive_cms(avg_metrics, f"УСРЕДНЕННОЕ ({len(fold_results)} ФОЛДОВ) Acc: {mean_str}", f"{mean_str}_cm_average_all.png")
    
    print(f"✅ Анализ успешно завершен! Результаты сохранены в папку: {RESULTS_DIR}")

if __name__ == "__main__":
    main()