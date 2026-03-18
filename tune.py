import optuna
import numpy as np
from config import DATA_ROOT
from dataset import build_dataframe, preprocess_files
from train import train_and_evaluate


def objective(trial, data_dict):
    # 1. Задаем диапазоны поиска гиперпараметров для Optuna
    params = {
        # Ищем Learning Rate в логарифмическом масштабе от 1e-5 до 1e-3
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),

        # Ищем Weight Decay (штраф за переобучение)
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),

        # Размер батча (перебираем 3 варианта)
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),

        # Вероятность применения CutMix
        'cutmix_prob': trial.suggest_float('cutmix_prob', 0.1, 1.0),
        'cutmix_beta': trial.suggest_float('cutmix_beta', 0.1, 1.0),

        # Уровень отсева нейронов (регуляризация внутри модели)
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 1.0)
    }

    # 2. Обучаем модель с этими параметрами
    # Отключаем save_artifacts, чтобы не выводить принты и не сохранять картинки для каждого Trial
    fold_results = train_and_evaluate(
        data_dict, trial_params=params, save_artifacts=False)

    # 3. Вычисляем главную метрику (Средняя точность по 5 фолдам)
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])

    # Optuna попытается максимизировать это значение
    return mean_accuracy


def run_tuning():
    print("🚀 Подготовка данных для Optuna...")
    df = build_dataframe(DATA_ROOT)
    data_dict = preprocess_files(
        df['filepath'].tolist(), (df['label'] == 'PD').astype(int).tolist(),
        df['patient_id'].tolist(), df['language_id'].tolist(), df['gender_id'].tolist()
    )

    print("\n🕵️‍♂️ Начинаем байесовскую оптимизацию...")
    # Создаем "Исследование" (Study), цель - максимизировать (maximize) метрику
    study = optuna.create_study(
        direction='maximize', study_name="Parkinson_Audio_Optimization")

    # Запускаем перебор. n_trials - количество попыток (рекомендуется 20-50)
    # Используем lambda, чтобы прокинуть data_dict в objective
    study.optimize(lambda trial: objective(trial, data_dict), n_trials=60)

    # === ВЫВОД РЕЗУЛЬТАТОВ ===
    print("\n" + "="*50)
    print("🏆 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print("="*50)

    best_trial = study.best_trial
    print(f"Лучшая средняя точность: {best_trial.value * 100:.2f}%\n")
    print("Идеальные гиперпараметры:")
    for key, value in best_trial.params.items():
        if type(value) == float:
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")


    print("\n💡 Скопируйте эти параметры в ваш config.py для дальнейшего использования!")


if __name__ == "__main__":
    run_tuning()
