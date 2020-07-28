import torch
import transformers
import flask
import time
import torch.nn as nn
import hyperparameters as hp

from flask import Flask
from flask import request
from model_punctuation import BertPunc
from dataset_punctuation import BERTDataset
from utils_punctuation import postprocess_text

app = Flask(__name__)

MODEL = None
DEVICE = "cpu"
model_path = "/home/jonas/speech-technologies/bert_punctuation/models/Bert_punctuation_Bert_1M_sentences_20200728_075223/model.bin"


def sentence_prediction(sentence):
    sentence = sentence.split()
    capitalization = [1 if a[0].isupper() else 0 for a in sentence]
    tokenizer = hp.TOKENIZER

    tokenized_sentence = tokenizer.encode(sentence)
    loader = BERTDataset(sentence=[sentence], labels=[[0] * len(sentence)])

    ids = loader[0]["ids"].to(DEVICE).unsqueeze(0)
    mask = loader[0]["mask"].to(DEVICE).unsqueeze(0)
    token_type_ids = loader[0]["token_type_ids"].to(DEVICE).unsqueeze(0)

    punctuation = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    prediction = (
        punctuation.argmax(2)
        .cpu()
        .numpy()
        .reshape(-1)[: len(tokenized_sentence)]
        .tolist()
    )

    return postprocess_text(sentence, prediction, capitalization)

@app.route("/predict", methods=['POST'])
def predict():
    # sentence = request.args.get("sentence")
    data = request.get_json(force=True)
    sentence = data["text"]
    start_time = time.time()
    prediction = sentence_prediction(sentence)
    response = {}
    response["response"] = {
        "sentence": str(sentence),
        "prediction": prediction,
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BertPunc()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host="127.0.0.1", port=9902)
