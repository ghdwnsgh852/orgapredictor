import sys

from torch.utils.tensorboard import SummaryWriter



import os
import random
import time

from util import models_vit
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import time
assert timm.__version__ == "0.3.2"  # version check

batch_size = 4

name = "finetune"





ckpt_path = f"./ckpt/{name}"
os.makedirs(ckpt_path, exist_ok=True)

logs_path = f"./logs/{name}"
os.makedirs(logs_path, exist_ok=True)
print('path done')
train_writer = SummaryWriter(f"{logs_path}/train")
val_writer = SummaryWriter(f"{logs_path}/val")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


train_df = pd.read_csv("./csv/train_pair.csv")
val_df = pd.read_csv("./csv/valid_pair.csv")
test_df = pd.read_csv("./csv/test_pair.csv")

print('csv done')



class CustomDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        # self.df=df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row_1 = self.df.iloc[idx]
        patientid_1 = row_1["Patient"]
        passage_1 = row_1["Passage"]
        day_1 = row_1["day1"]
        img_path_1 = f'{row_1["img1"]}'

        day_2 = row_1["day2"]
        day=row_1["day"]
        img_path_2 = f'{row_1["img2"]}'
        img_1 = Image.open(img_path_1).convert("RGB")
        img_2 = Image.open(img_path_2).convert("RGB")

        label = row_1["label"]
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)


        return img_1, img_2, label, day



train_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)


train_loader = DataLoader(
    CustomDataset(train_df, train_transform), batch_size=batch_size, shuffle=True
)

val_loader = DataLoader(
    CustomDataset(val_df, test_transform), batch_size=batch_size, shuffle=False
)

test_loader = DataLoader(
    CustomDataset(test_df, test_transform), batch_size=batch_size, shuffle=False
)

print('loader done')

model = models_vit.CustomModel(ckpt= '/home/ra9027/breast_organoid/code_20240426/checkpoint-80.pth', global_pool=True)

# for _, param in model.named_parameters():
#     param.requires_grad = False

# for _, param in model.fc.named_parameters():
#     param.requires_grad = True

epochs = 100
lr = 1e-4

model = model.to(device)

model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


best_loss = 10000
best_epoch = 0
best_acc = 0
best_auc = 0

print('start training')

for epoch in range(1, epochs + 1):
    train_losses = []
    start = time.time()
    model.train()
    for i, (img_1, img_2, label, _) in enumerate(train_loader):
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        label = label.to(device)
        # day = day.to(device)
        optimizer.zero_grad()
        output = model(img_1, img_2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()

    val_losses = []

    val_preds = []
    val_labels = []

    with torch.no_grad():
        for i, (img_1, img_2, label, _) in enumerate(val_loader):
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            label = label.to(device)

            output = model(img_1, img_2)
            loss = criterion(output, label)
            val_losses.append(loss.item())
            val_preds.append(output.softmax(1).cpu().numpy())
            val_labels.append(label.cpu().numpy())
    end = time.time()
    val_acc = accuracy_score(
        np.concatenate(val_labels), np.concatenate(val_preds).argmax(1)
    )
    val_auc = roc_auc_score(np.concatenate(val_labels),
                            np.concatenate(val_preds)[:, 1])

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    torch.save(model.state_dict(), f"{ckpt_path}/checkpoint-{epoch}.pth")
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        best_acc = val_acc
        best_auc = val_auc
        torch.save(model.state_dict(),
                   f"{ckpt_path}/checkpoint-best-{epoch}.pth")

    print(
        f"epoch {epoch} train_loss {train_loss} val_loss {val_loss} val_acc {val_acc} val_auc {val_auc} time {end-start}"
    )
    sys.stdout.flush()
    train_writer.add_scalar("loss", train_loss, epoch)
    val_writer.add_scalar("loss", val_loss, epoch)
    val_writer.add_scalar("acc", val_acc, epoch)
    val_writer.add_scalar("auc", val_auc, epoch)


train_writer.close()
val_writer.close()






model.load_state_dict(torch.load(f"{ckpt_path}/checkpoint-best-{best_epoch}.pth"))


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


print(f' last_model :  test_loss {test_loss} test_acc {test_acc} test_auc {test_auc}')

print(test_report)





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
