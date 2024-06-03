import sys
import argparse
import os
import time

import numpy as np
import pandas as pd

import timm
assert timm.__version__ == "0.3.2"  # version check
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from util import models_vit
from util.dataset import CustomDataset, train_transform, test_transform



paser = argparse.ArgumentParser()
paser.add_argument('--batch_size', type=int, default=4)
paser.add_argument('--test_csv', type=str, default='./csv/test_pair.csv')
paser.add_argument('--pretrained_weight', type=str, default='./pretrained/pretrained.pth')
args = paser.parse_args()


test_loader = DataLoader(CustomDataset(test_df, test_transform), batch_size=args.batch_size, shuffle=False)   


model = models_vit.CustomModel(ckpt= args.pretrained_weight, global_pool=True)
model = model.to(device)


model.eval()

test_losses = []
test_preds = []
test_labels = []
tqdm_loader = tqdm(test_loader)

with torch.no_grad():
    for img_1, img_2, label, _ in tqdm_loader:

        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        label = label.to(device)

        output = model(img_1, img_2)
        loss = criterion(output, label)
        test_losses.append(loss.item())
        test_preds.append(output.softmax(1).cpu().numpy())
        test_labels.append(label.cpu().numpy())


test_acc = accuracy_score(np.concatenate(test_labels), np.concatenate(test_preds).argmax(1))
test_auc = roc_auc_score(np.concatenate(test_labels), np.concatenate(test_preds)[:,1])
test_report=classification_report(np.concatenate(test_labels), np.concatenate(test_preds).argmax(1))



test_loss = np.mean(test_losses)


print(f' best_model :  test_loss {test_loss} test_acc {test_acc} test_auc {test_auc}')

print(test_report)
