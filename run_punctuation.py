import torch
import torch.nn.functional as F
import engine_punctuation
import wandb
import re
import numpy as np
import time
import hyperparameters as hp
import os

from torch.cuda import amp
from torch import nn, optim
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from model_punctuation import BertPunc
from dataset_punctuation import BERTDataset
from utils_punctuation import predict_sentences
from tqdm import tqdm
from sklearn import model_selection
from datetime import datetime

save_path = f'{hp.MODEL_PATH}/{hp.WANDB_PROJECT}_{hp.WANDB_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.mkdir(save_path)

wandb.init(entity="Parrot", name=hp.WANDB_NAME, project=hp.WANDB_PROJECT)
wandb.watch_called = False

config = wandb.config
config.batch_size = hp.BATCH_SIZE
config.epochs_top_layer = hp.EPOCHS_TOP_LAYERS
config.epochs_all_layers = hp.EPOCHS_ALL_LAYERS
config.learning_rate = hp.LEARNING_RATE
config.weight_decay = hp.WEIGHT_DECAY
config.dropout = hp.DROPOUT
config.max_len = hp.MAX_LENGTH
config.model_path = f"{save_path}/model.bin"
config.punctuation_enc = hp.PUNCTUATION_ENC
config.text_file = hp.TEXT_FILE
config.num_workers = hp.NUM_WORKERS
config.merge_number = hp.MERGE_NUMBER


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [t.strip() for t in f.readlines()]


def encode_data(data, punctuation_enc, merge_number=4):
    """" 
    texts: [["hi", "my", "name", "is", "jonas"], ["hello".....]]
    labels: [[4 1 1 1 2], [....].....]]
    """
    X, Y = [], []
    sentences = ""
    targets = []
    print("=> Preparing data")
    for i, line in tqdm(enumerate(data), total=len(data)):
        # include dummy punctuation if it is missing (e.g. during inference)
        if len(line.split("\t")) == 1:
            line = line + "\t" + "."
        text, punc = line.split("\t")
        if punc == "!":
            continue
        text = " ".join(re.sub(r"[^A-Za-z0-9']", " ", text).split())

        if len(text) == 0:
            continue
        sentences = sentences + " " + text
        targets.extend([0] * (len(text.split()) - 1) + [punctuation_enc[punc]])
        if i % merge_number == 0 and i != 0:
            sentences = sentences.strip().split()
            assert len(sentences) == len(
                targets
            ), "Sentence length must match with labels length"
            X.append(sentences)
            Y.append(targets)
            sentences = ""
            targets = []

    return X, Y


def run():
    data = load_file(config.text_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler()
    best_loss = np.inf

    X, y = encode_data(data, config.punctuation_enc, config.merge_number)

    # do train valid split
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        X, y, random_state=hp.RANDOM_STATE, test_size=hp.TEST_RATIO
    )

    train_dataset = BERTDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    valid_dataset = BERTDataset(X_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    model = BertPunc().to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_train_steps_top = int(
        len(X_train) / config.batch_size * config.epochs_all_layers
    )
    num_train_steps_all = int(
        len(X_train) / config.batch_size * config.epochs_top_layer
    )

    model = nn.DataParallel(model)

    # train the linear layer first while keeping other parameters fixed
    for p in model.module.bert.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps_top
    )

    print("=> Training top layer")
    for epoch in range(config.epochs_top_layer):
        start = time.time()
        print(f"=> Training epoch {epoch+1}/{config.epochs_top_layer}")
        train_loss = engine_punctuation.train(
            train_loader,
            model,
            optimizer,
            device,
            scheduler,
            epoch,
            config.epochs_top_layer,
            scaler,
        )
        validation_loss, val_f1 = engine_punctuation.evaluate(
            valid_loader, model, device
        )

        f1_cols = "      ".join([key for key in list(config.punctuation_enc.keys())])
        f1_vals = " ".join(["{:.4f}".format(val) for val in val_f1])
        end = time.time()

        print(
            f"Epoch time:          {end-start:.4f}s\n"
            f"Train loss:          {train_loss:.4f}\n"
            f"Validation Loss:     {validation_loss:.4f}\n"
            f"punctuation keys:    {f1_cols}\n"
            f"F1 punctuation:      {f1_vals}\n"
        )

    for p in model.module.bert.parameters():
        p.requires_grad = True

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps_all
    )
    print("=> Training all layers")
    for epoch in range(config.epochs_all_layers):
        start = time.time()
        print(f"=> Training epoch {epoch+1}/{config.epochs_all_layers}")
        train_loss = engine_punctuation.train(
            train_loader,
            model,
            optimizer,
            device,
            scheduler,
            epoch,
            config.epochs_all_layers,
            scaler,
        )
        print("=> Evaluation")
        validation_loss, val_f1 = engine_punctuation.evaluate(
            valid_loader, model, device
        )

        f1_cols = "      ".join([key for key in list(config.punctuation_enc.keys())])
        f1_vals = " ".join(["{:.4f}".format(val) for val in val_f1])

        end = time.time()

        print(
            f"Epoch time:          {end-start:.4f}s\n"
            f"Train loss:          {train_loss:.4f}\n"
            f"Validation Loss:     {validation_loss:.4f}\n"
            f"punctuation keys:    {f1_cols}\n"
            f"F1 punctuation:      {f1_vals}\n"
        )

        if validation_loss < best_loss:
            torch.save(model.state_dict(), f"{config.model_path}")
            best_loss = validation_loss

        predict_sentences(
            config.model_path, len(config.punctuation_enc), config.dropout
        )


if __name__ == "__main__":
    run()
