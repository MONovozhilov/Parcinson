import optuna
import numpy as np
from config import DATA_ROOT
from dataset import build_dataframe, preprocess_files
from train import train_and_evaluate

def objective(trial, data_dict):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size',[8, 16, 32]),
        'cutmix_prob': trial.suggest_float('cutmix_prob', 0.1, 1.0),
        'cutmix_beta': trial.suggest_float('cutmix_beta', 0.1, 1.0),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.9) 
    }

    fold_results = train_and_evaluate(data_dict, trial_params=params, save_artifacts=False)

    if fold_results is None:
        raise optuna.TrialPruned()

    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])

    return mean_accuracy


def run_tuning():
    df = build_dataframe(DATA_ROOT)
    data_dict = preprocess_files(
        df['filepath'].tolist(), (df['label'] == 'PD').astype(int).tolist(),
        df['patient_id'].tolist(), df['language_id'].tolist(), df['gender_id'].tolist()
    )

    study = optuna.create_study(direction='maximize', study_name="Parkinson_Audio_Optimization")

    study.optimize(lambda trial: objective(trial, data_dict), n_trials=50)

    print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")

    best_trial = study.best_trial
    print(f"Лучшая средняя точность: {best_trial.value * 100:.2f}%\n")
    print("гиперпараметры:")
    for key, value in best_trial.params.items():
        if type(value) == float:
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

if __name__ == "__main__":
    run_tuning()