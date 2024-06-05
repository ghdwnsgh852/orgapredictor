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
paser.add_argument('--model_path', type=str, default='./pretrained/pretrained.pth')
args = paser.parse_args()


test_loader = DataLoader(CustomDataset(test_df, test_transform), batch_size=args.batch_size, shuffle=False)   


model = models_vit.CustomModel(ckpt= args.pretrained_weight, global_pool=True)
model = model.to(device)


model.eval()



def test(model, test_loader):

    dir_name= f"./test_result/{model.split('/')[-2]}/{model.split('/')[-1].split('.')[0]}/{test_loader}"
    os.makedirs(dir_name, exist_ok=True)
    f= open(f'{dir_name}/result.txt','w')
    csvdf=csvs[test_loader]
    aa = torch.load(model)
    # for key, value in aa.copy().items():
    #     aa[key.replace('module.','')]=value
    #     del aa[key]
    custom_model = models_vit.CustomModel( global_pool=True)
    custom_model.load_state_dict(aa)
    model = custom_model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()

    test_losses = []

    test_preds = []
    test_labels = []
    tqdm_loader = tqdm(loaders[test_loader])
    img_path1=[]
    img_path2=[]
    days=[]

    with torch.no_grad():
        for img_1, img_2, label, day ,img_path_1,img_path_2 in tqdm_loader:

            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            label = label.to(device)

            output = model(img_1, img_2)
            loss = criterion(output, label)
            test_losses.append(loss.item())
            test_preds.append(output.softmax(1).cpu().numpy())
            test_labels.append(label.cpu().numpy())
            img_path1.append(img_path_1)
            img_path2.append(img_path_2)
            days.append(day)


    test_acc = accuracy_score(np.concatenate(test_labels),
                              np.concatenate(test_preds).argmax(1))
    test_auc = roc_auc_score(np.concatenate(test_labels),
                             np.concatenate(test_preds)[:, 1])
    test_report = classification_report(np.concatenate(test_labels),
                                        np.concatenate(test_preds).argmax(1))



    test_loss = np.mean(test_losses)

    img_paths1=np.concatenate(img_path1,axis=0)
    img_paths2=np.concatenate(img_path2,axis=0)

    days=np.concatenate(days,axis=0)

    f.write(f' test_loss {test_loss} test_acc {test_acc} test_auc {test_auc} \n')


    f.write(test_report)

    fprs, tprs, thresholds = roc_curve(np.concatenate(test_labels),
                                       np.concatenate(test_preds)[:, 1])
    plt.figure(figsize=(10, 10))

    plt.plot([0, 1], [0, 1], label='STR')
    plt.plot(fprs, tprs, label="ROC")

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.savefig(f'{dir_name}/roc_curve.png',format='png',dpi=200)
    plt.close()

    cm = confusion_matrix(np.concatenate(test_labels),
                          np.concatenate(test_preds).argmax(1))

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.grid(False)    
    plt.savefig(f'{dir_name}/confusion_matrix.png',format='png',dpi=200)
    plt.close()
    y_prob2 = np.concatenate(test_preds)[:, 1]

    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(np.concatenate(test_labels), y_prob2)
    # get the best threshold
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = thresholds[ix]
    f.write('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f  \n' %
          (best_thresh, tpr[ix], 1 - fpr[ix], J[ix]))

    y_prob_pred = (y_prob2 >= best_thresh).astype(bool)
    f.write(classification_report(np.concatenate(test_labels), y_prob_pred))

    cm = confusion_matrix(np.concatenate(test_labels), y_prob_pred)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.grid(False)
    plt.ylabel('True')
    plt.savefig(f'{dir_name}/Adjustmented_threshold_confusion_matrix.png',format='png',dpi=200)
    plt.close()
    gc.collect()
    torch.cuda.empty_cache()
    f.close()

    csvdf['prob']=np.concatenate(test_preds)[:, 1]
    csvdf['pred']=np.concatenate(test_preds).argmax(1)
    csvdf['adj_pred']=y_prob_pred


    # df=pd.DataFrame({'img_path1':img_paths1,'img_path2':img_paths2,'days':days,'label':np.concatenate(test_labels),'prob':np.concatenate(test_preds)[:, 1],'pred':np.concatenate(test_preds).argmax(1),'adj_pred':y_prob_pred})
    csvdf.to_csv(f'{dir_name}/result.csv',index=False)


