from config import DATA_ROOT, SAVE_RESULTS, RESULTS_DIR, DEVICE
from dataset import build_dataframe, preprocess_files
from train import train_and_evaluate
from visualizations import plot_metric_distributions, generate_text_report, plot_comprehensive_cms
import numpy as np

def main():
    print(f"Сохранение результатов: {'ВКЛЮЧЕНО' if SAVE_RESULTS else 'ВЫКЛЮЧЕНО'}")
    print(f"device {DEVICE}")
    df = build_dataframe(DATA_ROOT)
    data_dict = preprocess_files(
        df['filepath'].tolist(), (df['label'] == 'PD').astype(int).tolist(),
        df['patient_id'].tolist(), df['language_id'].tolist(), df['gender_id'].tolist()
    )
    
    fold_results = train_and_evaluate(data_dict)
    
    plot_metric_distributions(fold_results)
    generate_text_report(fold_results)
    
    avg_metrics = {'cm_overall': np.zeros((2,2)), 'cm_gender': {0: np.zeros((2,2)), 1: np.zeros((2,2))}, 'cm_lang': {0: np.zeros((2,2)), 1: np.zeros((2,2)), 2: np.zeros((2,2))}}
    for r in fold_results:
        avg_metrics['cm_overall'] += r['cm_overall']
        
        for g in [0, 1]: 
            if g in r['cm_gender']: 
                avg_metrics['cm_gender'][g] += r['cm_gender'][g]
                
        for l in [0, 1, 2]: 
            if l in r['cm_lang']: 
                avg_metrics['cm_lang'][l] += r['cm_lang'][l]
            
    for k in ['cm_overall']: avg_metrics[k] /= len(fold_results)
    for g in [0,1]: avg_metrics['cm_gender'][g] /= len(fold_results)
    for l in [0,1,2]: avg_metrics['cm_lang'][l] /= len(fold_results)
        
    plot_comprehensive_cms(avg_metrics, "УСРЕДНЕННОЕ ПО ВСЕМ ФОЛДАМ", "cm_average_all.png")
    
    if SAVE_RESULTS: print(f"{RESULTS_DIR}")

if __name__ == "__main__":
    main()