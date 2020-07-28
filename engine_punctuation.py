import torch
import torch.nn as nn
import numpy as np
import hyperparameters as hp

from sklearn import metrics
from tqdm import tqdm
from torch.cuda import amp


def train(data_loader, model, optimizer, device, scheduler, epoch, num_epochs, scaler):
    model.train()
    loop = tqdm(data_loader, total=len(data_loader), leave=False)
    final_loss = 0

    for data in loop:
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()
        with amp.autocast():
            _, loss = model(**data)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        final_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

    return final_loss / len(data_loader)


def evaluate(data_loader, model, device):
    model.eval()
    val_f1s = []
    label_vals = list(hp.PUNCTUATION_ENC.values())
    final_loss = 0

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            output, loss = model(**data)
            final_loss += loss.item()

            mask_cpu = data["mask"].cpu().data.numpy()  # .reshape(-1)
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

    return final_loss / len(data_loader), np.array(val_f1s).mean(axis=0)
