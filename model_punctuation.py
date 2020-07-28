import torch
import transformers
import torch.nn as nn
import hyperparameters as hp

from torch.cuda import amp


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1  # do not compute where attention mask is 0
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss, target.view(-1), torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class BertPunc(nn.Module):
    def __init__(self):
        super(BertPunc, self).__init__()
        self.num_classes = len(hp.PUNCTUATION_ENC)
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        # self.bert = transformers.BertModel.from_pretrained("bert-large-uncased")
        self.dropout = nn.Dropout(hp.DROPOUT)
        # pretrained BERT has linear layer 768
        self.out_labels = nn.Linear(768, len(hp.PUNCTUATION_ENC))

    @amp.autocast()
    def forward(self, ids, mask, token_type_ids, target_label=None):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.dropout(o1)
        labels = self.out_labels(bo)

        if target_label is not None:
            loss = loss_fn(labels, target_label, mask, self.num_classes)

            return labels, loss

        return labels
