import torch

SAVE_RESULTS = True 
RESULTS_DIR = r"C:\Users\erith\OneDrive\Документы\GitHub\Parkinson\results"

DATA_ROOT = r"C:\Users\erith\Downloads\Чтение текста\Чтение текста"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEGMENT_DURATION = 7.0
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 1024
TARGET_FRAMES = int(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH)
ACOUSTIC_FEATURE_SIZE = 123

# ==================== ОБУЧЕНИЕ И CUTMIX ====================
USE_CUTMIX = True
CUTMIX_PROB = 0.8134600743847827
CUTMIX_BETA = 0.9881977651733127

BATCH_SIZE = 8
LEARNING_RATE = 0.00011387983829722286
WEIGHT_DECAY = 0.03212592830307472
DROPOUT_RATE = 0.16232456925581062

NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 60
IMPROVEMENT_THRESHOLD = 0.001
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

N_SPLITS = 5

NUM_LANGUAGES = 3
LANGUAGE_KEYWORDS = {
    0: ['rus', 'russian', 'рус', 'ру'],
    1:['tat', 'tatar', 'тат', 'тт'],
    2:['bil', 'bilingual', 'би', 'билингв']
}
LANG_NAMES = {0: 'Русский', 1: 'Татарский', 2: 'Билингв'}
GENDER_NAMES = {0: 'Женщины (Female)', 1: 'Мужчины (Male)'}