import transformers
import torch
import hyperparameters as hp


class BERTDataset:
    def __init__(self, sentence, labels):
        self.sentence = sentence
        self.labels = labels
        self.tokenizer = hp.TOKENIZER
        self.max_len = hp.MAX_LENGTH

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        # See: https://huggingface.co/transformers/model_doc/bert.html
        # The token used for padding, for example when batching sequences of different lengths
        # pad token: [PAD] : 0
        # The classifier token which is used when doing sequence classification
        # It is the first token of the sequence when built with special tokens
        # cls token: [CLS] : 101
        # The separator token, which is used when building a sequence from multiple sequences
        # It is also used as the last token of a sequence built with special tokens
        # sep token: [SEP] : 102

        sentence = self.sentence[item]
        labels = self.labels[item]

        ids = []
        target_labels = []

        for i, s in enumerate(sentence):
            inputs = self.tokenizer.encode(s, add_special_tokens=False)
            # jonas: jo ##n ##as
            input_len = len(inputs)
            ids.extend(inputs)
            target_labels.extend([labels[i]] * input_len)

        ids = ids[: self.max_len - 2]
        target_labels = target_labels[: self.max_len - 2]

        ids = [101] + ids + [102]  # add CLS and SEP symbols at the beginning and end
        target_labels = [0] + target_labels + [0]  # pad labels to match extended ids

        # we do not want BERT to attent to the padded indexes. For the BertTokenizer, 1 indicate a value that should be attended to
        # while 0 indicate a padded value.
        mask = [1] * len(ids)

        # token type ids in BERT is a binary mask identifying where one sentence ends and other begins
        # usually [1,1,1,1,0,0,0,0,0,0,0] but we have just one sequence so [0,0,0,0,0,0,0,0,0,0,0,0,0]
        token_type_ids = [0] * len(ids)

        padding_len = self.max_len - len(ids)
        # Here always use 0, good resource: https://www.kaggle.com/debanga/huggingface-tokenizers-cheat-sheet
        mask = mask + ([0] * padding_len)  # pad mask
        token_type_ids = token_type_ids + ([0] * padding_len)

        # here ue PAD symbol
        ids = ids + ([0] * padding_len)  # pad inputs
        target_labels = target_labels + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_label": torch.tensor(target_labels, dtype=torch.long),
        }
