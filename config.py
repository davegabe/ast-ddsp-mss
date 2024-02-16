# Dataset config
SR = 8000
DATASET_PATH = "slakh"
DATA_TYPE = ".wav"
INST_KEY = "inst_class" # Use "midi_program_name" or "inst_class" depending if you want to use the precise instrument name or the instrument class
STEMS = ["mixture", "bass"]  # ["mixture", "bass", "drums", "guitar", "piano"] (can use regular expressions)
SECONDS = 2
LENGTH = SR * SECONDS
CHANNELS = 1
MIN_DURATION = 12.0
MAX_DURATION = 640.0
AUG_SHIFT = True
N_TRACKS = -1
SILENCE_THRESHOLD = 0.003
VARIANCE_THRESHOLD = 0.0005
OVERLAP = 0.85

# Model config
MODEL_SIZE = "base384"
EXP_DIR = "exp"
BATCH_SIZE = 128
NUM_WORKERS = 6
N_EPOCHS = 800
LR = 3e-4
N_PRINT_STEPS = 100
SAVE_EPOCHS = 50
MAX_CHECKPOINTS = 10
VALIDATION_EPOCHS = 5
WARMUP = False
LR_SCHEDULER = True
LR_SCHEDULER_START = 25
LR_SCHEDULER_STEP = 25
LR_SCHEDULER_DECAY = 0.98
