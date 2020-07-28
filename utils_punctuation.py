import torch
import transformers
import hyperparameters as hp

from dataset_punctuation import BERTDataset
from model_punctuation import BertPunc
from torch import nn


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
    return final_sentence


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
        output, _ = model(**data)
        output = postprocess_text(
            sentence,
            output.argmax(2)
            .cpu()
            .numpy()
            .reshape(-1)[: len(tokenized_sentence)]
            .tolist(), capitalization
        )
        print(f"Gold: {gold}")
        print(f"Prediction: {output}")


# if __name__ == "__main__":
#     predict_sentences("/home/jonas/punctuation_models/punctuation/BERT/models/model.bin", 4, 2, 0.3)
