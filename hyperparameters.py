import transformers

MERGE_NUMBER = 4
NUM_WORKERS = 8
MAX_LENGTH = 128
BATCH_SIZE = 128
EPOCHS_TOP_LAYERS = 2
EPOCHS_ALL_LAYERS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.001
DROPOUT = 0.3
RANDOM_STATE = 42
TEST_RATIO = 0.1
PUNCTUATION_ENC = {"O": 0, ".": 1, "?": 2, ",": 3}
BASE_MODEL_PATH = "bert-base-uncased"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH, do_lower_case=True
)
WANDB_NAME = "Bert_1M_sentences"
WANDB_PROJECT = "Bert_punctuation"
MODEL_PATH = "/home/jonas/speech-technologies/bert_punctuation/models/"
TEXT_FILE = "/home/jonas/speech-technologies/bert_punctuation/data/parrot_all"
