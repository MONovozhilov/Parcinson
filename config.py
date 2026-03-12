import torch
from pathlib import Path

SEGMENT_DURATION = 7.0
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 256
N_FFT = 1024
TARGET_FRAMES = int(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH)

USE_CUTMIX = True
CUTMIX_PROB = 0.7
CUTMIX_BETA = 1.0

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 100
IMPROVEMENT_THRESHOLD = 0.001
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_SPLITS = 5 

NUM_LANGUAGES = 3
LANGUAGE_KEYWORDS = {
    0: ['rus', 'russian', 'рус', 'ру'],
    1: ['tat', 'tatar', 'тат', 'тт'],
    2: ['bil', 'bilingual', 'би', 'билингв']
}
LANG_NAMES = {0: 'Русский', 1: 'Татарский', 2: 'Билингв'}

DATA_ROOT = Path("C:\\Users\\erith\\Downloads\\Чтение текста\\Чтение текста")

ACOUSTIC_FEATURE_SIZE = 123
CNN_CHANNELS = [32, 64, 128, 256]
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
MLP_SIZES = [512, 256, 128]
NUM_CLASSES = 2