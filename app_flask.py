import torch
import transformers
import flask
import time
import torch.nn as nn
import hyperparameters as hp
import torch.nn.functional as F
import ast

from flask import Flask
from flask import request
from model_punctuation import BertPunc
from dataset_punctuation import BERTDataset
from utils_punctuation import postprocess_text, postprocess_text_with_confidence, load_live_asr_input, live_asr_output

app = Flask(__name__)

MODEL = None
DEVICE = hp.LIVE_ASR_DEVICE
# model_path = "/home/jonas/speech-technologies/bert_punctuation/models/Bert_punctuation_punctuation_parrot_final_20200729_125950/model_20200729_143254.pt"
model_path = "/home/jonas/speech-technologies/bert_punctuation/models/Bert_punctuation_punctuation_new_3replicas_top10_20200802_145359/model_20200803_005754.pt"


def sentence_prediction(sentence):
    sentence = sentence.split()
    capitalization = [1 if a[0].isupper() else 0 for a in sentence]
    tokenizer = hp.TOKENIZER

    tokenized_sentence = tokenizer.encode(sentence)
    loader = BERTDataset(sentence=[sentence], labels=[[0] * len(sentence)])

    ids = loader[0]["ids"].to(DEVICE).unsqueeze(0)
    mask = loader[0]["mask"].to(DEVICE).unsqueeze(0)
    token_type_ids = loader[0]["token_type_ids"].to(DEVICE).unsqueeze(0)

    logits = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    prediction = (
        logits.argmax(2)
        .cpu()
        .numpy()
        .reshape(-1)[: len(tokenized_sentence)]
        .tolist()
    )

    if hp.OUTPUT_CONFIDENCES:

        logits = F.softmax(logits, dim=2)
        logits_confidence = [values[label].item() for values,label in zip(logits[0],prediction)]

        return postprocess_text_with_confidence(sentence, prediction, capitalization, logits_confidence)


    if hp.TUNE_CONFIDENCES:
        # This is for confidence hyperparameter tuning
        logits = F.softmax(logits, dim=2)
        logits_confidence = [values[label].item() for values,label in zip(logits[0],prediction)]
        logits_confidence = logits_confidence[1:-1]
        prediction = prediction[1:-1]
        for i, log in enumerate(logits_confidence):
            if log < hp.CONFIDENCE_THRESHOLD:
                prediction[i] = 0

        return prediction

    return postprocess_text(sentence, prediction, capitalization)


@app.route("/predict", methods=["POST"])
def predict():
    # sentence = request.args.get("sentence")
    data = request.get_json(force=True)
    sentence = data["text"]

    if hp.LIVE_ASR:
        original_data = ast.literal_eval(data["text"])
        # handle empty inputs
        if len(original_data["words"]) == 0:
            return flask.jsonify(original_data)

        sentence = load_live_asr_input(original_data)

    start_time = time.time()
    prediction = sentence_prediction(sentence)
    prediction = str(prediction)

    if hp.LIVE_ASR:
        response = live_asr_output(prediction, original_data)
        return flask.jsonify(response)

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
    app.run(host=hp.HOST, port=hp.PORT)
