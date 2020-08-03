import transformers

MERGE_NUMBER = 8
NUM_WORKERS = 4
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS_TOP_LAYER = 5
EPOCHS_ALL_LAYERS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.001
DROPOUT = 0.3
RANDOM_STATE = 42
TEST_RATIO = 0.1
PUNCTUATION_ENC = {"O": 0, ".": 1, "?": 2, ",": 3}
BASE_MODEL_PATH = "bert-base-uncased"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH, do_lower_case=True
)
WANDB_NAME = "punctuation_new_3replicas_top10"
WANDB_PROJECT = "Bert_punctuation"
TRAIN_FILE = "/home/jonas/speech-technologies/bert_punctuation/data/better_parrot_train"
VALID_FILE = "/home/jonas/speech-technologies/bert_punctuation/data/better_parrot_dev"
MODEL_PATH = "/home/jonas/speech-technologies/bert_punctuation/models/"
HOST = "127.0.0.1"
PORT = 9901
OUTPUT_CONFIDENCES = True
CONFIDENCE_THRESHOLD = 0.0
LIVE_ASR = False
SAVE_MODEL = True
MIN_MERGE = 1
MAX_MERGE = 15
NUM_REPLICATION = 15
TUNE_CONFIDENCES = False
LIVE_ASR_DEVICE = "cpu"
