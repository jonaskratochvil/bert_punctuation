import torch
import json
import transformers
import hyperparameters as hp

from dataset_punctuation import BERTDataset
from model_punctuation import BertPunc
from torch import nn
from tqdm import tqdm

def load_json(json_file):
    with open(json_file, 'r') as f:
        text = json.load(f)
    return text

def load_live_asr_input(json_file):
    text = load_json(json_file)
    final_sentence = ""
    for w in text["words"]:
        final_sentence = final_sentence + " " + w["word"]

    return final_sentence

def live_asr_output(prediction, original_data):
    original_data = load_json(original_data)
    prediction = prediction.split()
    for i in range(len(original_data["words"])):
        original_data["words"][i]["word"] = prediction[i]

    return original_data

def postprocess_text_with_confidence(sentence, labels, capitalization, confidences):
    final_sentence = ""
    confidences = confidences[1:-1]
    labels = labels[1:-1]
    upper = True
    for i, word in enumerate(sentence):
        if upper or capitalization[i]:
            word = word.capitalize()
            upper = False

        if confidences[i] < hp.CONFIDENCE_THRESHOLD:
            labels[i] = 0

        if labels[i] == 0:
            final_sentence = final_sentence + " " + word
        elif labels[i] == 1:
            final_sentence = final_sentence + " " + word + "."
            upper = True
        elif labels[i] == 2:
            final_sentence = final_sentence + " " + word + "?"
            upper = True
        else:
            final_sentence = final_sentence + " " + word + ","

    final_sentence = final_sentence[:-1] + "." if final_sentence[-1] == "," else final_sentence

    return final_sentence.strip()


def postprocess_text(sentence, labels, capitalization):
    final_sentence = ""
    labels = labels[1:-1]
    upper = True
    for i, word in enumerate(sentence):
        if upper or capitalization[i]:
            word = word.capitalize()
            upper = False
        if labels[i] == 0:
            final_sentence = final_sentence + " " + word
        elif labels[i] == 1:
            final_sentence = final_sentence + " " + word + "."
            upper = True
        elif labels[i] == 2:
            final_sentence = final_sentence + " " + word + "?"
            upper = True
        else:
            final_sentence = final_sentence + " " + word + ","

    return final_sentence.strip()


def predict_sentences(current_model_path, num_classes, dropout):
    gold = "hi, how are you today? i am fine thank you. what is your name?"
    sentence = "hi how are you today i am fine thank you what is your name"
    sentence = sentence.split()
    capitalization = [1 if a[0].isupper() else 0 for a in sentence]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = hp.TOKENIZER
    tokenized_sentence = tokenizer.encode(sentence)
    revert_sentence = tokenizer.convert_ids_to_tokens(tokenized_sentence)

    test_loader = BERTDataset(sentence=[sentence], labels=[[0] * len(sentence)])

    model = BertPunc()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(current_model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        data = test_loader[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        logits, _ = model(**data)
        output = postprocess_text(
            sentence,
            logits.argmax(2)
            .cpu()
            .numpy()
            .reshape(-1)[: len(tokenized_sentence)]
            .tolist(),
            capitalization,
        )
        print(f"Gold: {gold}")
        print(f"Prediction: {output}")

def predict_validation_set():
    from run_punctuation import load_file, encode_data
    current_model_path = "/home/jonas/speech-technologies/bert_punctuation/models/Bert_punctuation_punctuation_parrot_final_20200729_125950/model_20200729_143254.pt"
    device = "cuda"
    final_loss = 0
    val_f1s = []

    data_valid = load_file(hp.VALID_FILE)
    X_valid, y_valid = encode_data(data_valid, hp.PUNCTUATION_ENC, hp.MERGE_NUMBER, train=False)

    valid_dataset = BERTDataset(X_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=hp.BATCH_SIZE, num_workers=hp.NUM_WORKERS, shuffle=False
    )

    model = BertPunc().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(current_model_path))
    model.eval()

    with torch.no_grad():
        for data in tqdm(valid_dataset, total=len(valid_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            output, loss = model(**data)
            final_loss += loss.item()

            mask_cpu = data["mask"].cpu().data.numpy()
            indices = np.where(mask_cpu == 1)

            # Take only predictions where mask is not zero
            y_pred = output.argmax(dim=2).cpu().data.numpy()
            pred_values = y_pred[indices].flatten()

            y_true = data["target_label"].cpu().data.numpy()
            true_values = y_true[indices].flatten()

            val_f1s.append(
                metrics.f1_score(
                    true_values, pred_values, average=None, labels=label_vals
                )
            )
    print(np.array(val_f1s).mean(axis=0))
    print(final_loss / len(data_loader))



if __name__ == "__main__":
    predict_validation_set()

# if __name__ == "__main__":
#     predict_sentences("/home/jonas/punctuation_models/punctuation/BERT/models/model.bin", 4, 2, 0.3)
