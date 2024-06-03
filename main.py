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
paser.add_argument('--name', type=str, default='finetune')
paser.add_argument('--mode',type=str, default='finetune')
paser.add_argument('--epochs', type=int, default=100)
paser.add_argument('--lr', type=float, default=1e-4)
paser.add_argument('--device', type=str, default='cuda')
paser.add_argument('--train_csv', type=str, default='./csv/train_pair.csv')
paser.add_argument('--val_csv', type=str, default='./csv/valid_pair.csv')
paser.add_argument('--test_csv', type=str, default='./csv/test_pair.csv')
paser.add_argument('--name', type=str, default=None)
paser.add_argument('--pretrained_weight', type=str, default='./pretrained/pretrained.pth')


args = paser.parse_args()




ckpt_path = f"./ckpt/{args.name}"
os.makedirs(ckpt_path, exist_ok=True)

logs_path = f"./logs/{args.name}"
os.makedirs(logs_path, exist_ok=True)
print('path done')
train_writer = SummaryWriter(f"{logs_path}/train")
val_writer = SummaryWriter(f"{logs_path}/val")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


train_df = pd.read_csv(args.train_csv)
val_df = pd.read_csv(args.val_csv)
test_df = pd.read_csv(args.test_csv)

print('csv done')







train_loader = DataLoader(
    CustomDataset(train_df, train_transform), batch_size=args.batch_size, shuffle=True
)

val_loader = DataLoader(
    CustomDataset(val_df, test_transform), batch_size=args.batch_size, shuffle=False
)

test_loader = DataLoader(
    CustomDataset(test_df, test_transform), batch_size=args.batch_size, shuffle=False
)

print('loader done')

model = models_vit.CustomModel(ckpt= args.pretrained_weight, global_pool=True)



model = model.to(device)

if args.mode == 'linear':
    for _, param in model.named_parameters():
        param.requires_grad = False

    for _, param in model.fc.named_parameters():
        param.requires_grad = True


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()


best_loss = 10000
best_epoch = 0
best_acc = 0
best_auc = 0

print('start training')

for epoch in range(1, args.epochs + 1):
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
