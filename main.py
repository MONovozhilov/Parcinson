import torch
import numpy as np
import shutil
from config import DEVICE, N_SPLITS, NUM_LANGUAGES, LANG_NAMES
from data_loader import load_data_records
from features import preprocess_all_files_once
from training import train_with_group_cv
from utils import plot_training_history, summarize_cv_results

def select_and_save_global_best(fold_results, output_path="parkinson_global_best.pt"):
    best_fold = max(fold_results, key=lambda x: x['best_val_acc_file'])
    shutil.copy2(best_fold['model_path'], output_path)
    return best_fold, output_path

if __name__ == "__main__":
    df, patient_ids_list = load_data_records()
    
    filepaths = df["filepath"].tolist()
    labels = (df["label"] == "PD").astype(int).tolist()
    language_ids_list = df["language_id"].tolist()
    
    preprocessed_data = preprocess_all_files_once(
        filepaths, labels, patient_ids_list, language_ids_list
    )
    
    fold_results = train_with_group_cv(preprocessed_data, n_splits=N_SPLITS)
    
    for r in fold_results:
        plot_training_history(r['history'], r['fold_num'])
        
    summarize_cv_results(fold_results, N_SPLITS, NUM_LANGUAGES, LANG_NAMES)
    
    best_fold, global_best_path = select_and_save_global_best(fold_results, "parkinson_global_best.pt")
    
    print(f"Training complete. Best model: {global_best_path}")
    print(f"Best Accuracy: {best_fold['best_val_acc_file']:.4f}")