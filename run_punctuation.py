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
from torch.utils.data import SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from model_punctuation import BertPunc
from dataset_punctuation import BERTDataset
from utils_punctuation import predict_sentences
from tqdm import tqdm
from sklearn import model_selection
from datetime import datetime
from random import randrange

save_path = f'{hp.MODEL_PATH}/{hp.WANDB_PROJECT}_{hp.WANDB_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.mkdir(save_path)
print(f"Making directory {save_path}")

wandb_name = f"learning_rate_{hp.LEARNING_RATE}_epochs_top_{hp.EPOCHS_TOP_LAYER}_epochs_all_{hp.EPOCHS_ALL_LAYERS}_min_merge_{hp.MIN_MERGE}_max_merge_{hp.MAX_MERGE}_replications_{hp.NUM_REPLICATION}"
wandb.init(entity="Parrot", name=wandb_name, project=hp.WANDB_PROJECT)
wandb.watch_called = False

config = wandb.config
config.batch_size = hp.BATCH_SIZE
config.epochs_top_layer = hp.EPOCHS_TOP_LAYER
config.epochs_all_layers = hp.EPOCHS_ALL_LAYERS
config.learning_rate = hp.LEARNING_RATE
config.weight_decay = hp.WEIGHT_DECAY
config.dropout = hp.DROPOUT
config.max_len = hp.MAX_LENGTH
config.model_path = save_path
config.punctuation_enc = hp.PUNCTUATION_ENC
config.train_file = hp.TRAIN_FILE
config.valid_file = hp.VALID_FILE
config.num_workers = hp.NUM_WORKERS
config.merge_number = hp.MERGE_NUMBER
config.max_len = hp.MAX_LENGTH
config.min_merge = hp.MIN_MERGE
config.max_merge = hp.MAX_MERGE
config.num_replication = hp.NUM_REPLICATION

def load_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [t.strip() for t in f.readlines()]


def encode_data(data, punctuation_enc, merge_number=4, train=True):
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
        if len(line.split(" ")) == 1:
            continue
        text = ' '.join(line.split()[:-1])
        punc = line.split()[-1]
        if punc == "!":
            continue
        text = " ".join(re.sub(r"[^A-Za-z'-]", " ", text).split())
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
            merge_number = randrange(hp.MIN_MERGE,hp.MAX_MERGE+1) if train else merge_number

    return X, Y


def run():
    data_train = load_file(config.train_file)
    data_valid = load_file(config.valid_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler()
    best_loss = np.inf

    # idea here is that every time we are taking randomized number of chunks so we run it multiple
    # times and each time we get different contexts
    X_train, y_train = [], []
    for _ in range(hp.NUM_REPLICATION):
        X, y = encode_data(data_train, config.punctuation_enc, config.merge_number, train=True)
        X_train = X_train + X
        y_train = y_train + y

    X_valid, y_valid = encode_data(data_valid, config.punctuation_enc, config.merge_number, train=False)

    train_dataset = BERTDataset(X_train, y_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True
    )

    valid_dataset = BERTDataset(X_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False
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
    # Steps for scheduler are different for top and all layers
    num_train_steps_top = int(
        len(X_train) / config.batch_size * config.epochs_top_layer
    )
    num_train_steps_all = int(
        len(X_train) / config.batch_size * config.epochs_all_layers
    )

    model = nn.DataParallel(model)

    # train the linear layer first while keeping other parameters fixed
    for p in model.module.bert.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=hp.WARMUP_STEPS, num_training_steps=num_train_steps_top
    )

    print("=> Training top layer")
    for epoch in range(config.epochs_top_layer):
        start = time.time()
        print(f"======== Epoch {epoch+1}/{config.epochs_top_layer} ========")
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

        wandb.log(
                    {
                   'epoch': epoch,
                   'train loss': train_loss,
                   'validation loss': validation_loss,
                   'F1 O': val_f1[0],
                   'F1 .': val_f1[1],
                   'F1 ?': val_f1[2],
                   'F1 ,': val_f1[3]
                   }
                )

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
        optimizer, num_warmup_steps=hp.WARMUP_STEPS, num_training_steps=num_train_steps_all
    )
    print("=> Training all layers")
    for epoch in range(config.epochs_all_layers):
        start = time.time()
        print(f"======== Epoch {epoch+1}/{config.epochs_all_layers} ========")
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

        wandb.log(
                    {
                   'train loss': train_loss,
                   'validation loss': validation_loss,
                   'F1 O': val_f1[0],
                   'F1 .': val_f1[1],
                   'F1 ?': val_f1[2],
                   'F1 ,': val_f1[3]
                   }
                )

        if validation_loss < best_loss and hp.SAVE_MODEL:
            model_name = f"{config.model_path}/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(model.state_dict(), model_name)
            best_loss = validation_loss

            # predict_sentences(
            #     model_name, len(config.punctuation_enc), config.dropout
            # )


if __name__ == "__main__":
    run()
