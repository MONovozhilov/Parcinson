import torch

SAVE_RESULTS = True 
RESULTS_DIR = "run_results" 
DATA_ROOT = r"C:\Users\erith\Downloads\Чтение текста\Чтение текста"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEGMENT_DURATION = 7.0
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 1024
TARGET_FRAMES = int(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH)
ACOUSTIC_FEATURE_SIZE = 123

USE_CUTMIX = True
CUTMIX_PROB = 0.552846
CUTMIX_BETA = 0.784582

BATCH_SIZE = 8
LEARNING_RATE = 0.000338
WEIGHT_DECAY = 0.000402
DROPOUT_RATE = 0.367601

NUM_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 50
IMPROVEMENT_THRESHOLD = 0.001
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
N_SPLITS = 5

NUM_LANGUAGES = 3
LANGUAGE_KEYWORDS = {
    0:['rus', 'russian', 'рус', 'ру'],
    1: ['tat', 'tatar', 'тат', 'тт'],
    2: ['bil', 'bilingual', 'би', 'билингв']
}
LANG_NAMES = {0: 'Русский', 1: 'Татарский', 2: 'Билингв'}
GENDER_NAMES = {0: 'Женщины (Female)', 1: 'Мужчины (Male)'}