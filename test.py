
import argparse
import os

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
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from util import models_vit
from util.util import attentionmap
from util.dataset import CustomDataset, test_transform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from numpy import argmax
import gc
paser = argparse.ArgumentParser()
paser.add_argument('--batch_size', type=int, default=32)
paser.add_argument('--test_csv', type=str, default='./csv/test_pair.csv')
paser.add_argument('--model_path', type=str, default='./ckpt/pretrained.pth')
paser.add_argument('--output_dir', type=str, default='./output')
args = paser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

test_df=pd.read_csv(args.test_csv)
test_loader = DataLoader(CustomDataset(test_df, test_transform), batch_size=args.batch_size, shuffle=False)   
print('loader done')
os.makedirs(args.output_dir, exist_ok=True)

dir_name= args.output_dir
attentionmap_dir = os.path.join(dir_name, 'attentionmap')
os.makedirs(dir_name, exist_ok=True)
os.makedirs(f'{dir_name}/attentionmap', exist_ok=True)
f= open(f'{dir_name}/result.txt','w')

weight = torch.load(args.model_path)
for key, value in weight.copy().items():
    weight[key.replace('module.','')]=value
    del weight[key]


custom_model = models_vit.AttentionModel( global_pool=True)
custom_model.load_state_dict(weight)
model = custom_model.to(device)
model.register_hook()


model.eval()
tqdm_loader = tqdm(test_loader)


test_preds = []


img_path1=[]
img_path2=[]

img_path1s = []
img_path2s = []
days = []
attentionmap1s = []
attentionmap2s = []
masks1 = []
masks2 = []
day_1s = []
day_2s = []
test_labels = []
patientids = []
passages=[]
test_probs=[]

with torch.no_grad():
    for img_1, img_2, label, day, img_path_1, img_path_2, patientid_1, passage_1, day_1, day_2 in tqdm_loader:

        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        label = label.to(device)
        model.feature=[]
        gc.collect()
        torch.cuda.empty_cache()
        output=model.attention_map(img_1 , img_2 )

        test_preds.append(output.softmax(1).cpu().numpy())
        test_labels.append(label.cpu().numpy())
        img_path1s.append(img_path_1)
        img_path2s.append(img_path_2)
        patientids.append(patientid_1)
        days.append(day)
        passages.append(passage_1)
        day_1s.append(day_1)
        day_2s.append(day_2)

        i=0

        for  img_path_1, img_path_2 in zip( img_path_1, img_path_2):
            att_mat1 = model.feature[0][0][i].unsqueeze(0).detach().cpu()
            att_mat2 = model.feature[1][0][i].unsqueeze(0).detach().cpu()
            i+=1

            origanal_img_1 = cv2.cvtColor(cv2.imread(img_path_1), cv2.COLOR_BGR2RGB)
            origanal_img_2 = cv2.cvtColor(cv2.imread(img_path_2), cv2.COLOR_BGR2RGB)
            attentionmap1, mask1 = attentionmap(att_mat1, origanal_img_1)
            attentionmap2, mask2 = attentionmap(att_mat2, origanal_img_2)
            mask1 = cv2.normalize(mask1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask2 = cv2.normalize(mask2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            mask1 = cv2.applyColorMap(mask1, cv2.COLORMAP_JET)
            mask2 = cv2.applyColorMap(mask2, cv2.COLORMAP_JET)

            cv2.imwrite(f'{attentionmap_dir}/{img_path_1.split("/")[-1].split(".")[0]}_attentionmap1.jpg', attentionmap1)
            attentionmap1s.append(
                f'{img_path_1.split("/")[-1].split(".")[0]}_attentionmap1.jpg'
            )
            cv2.imwrite(f'{attentionmap_dir}/{img_path_1.split("/")[-1].split(".")[0]}_mask1.jpg', mask1)
            masks1.append(
                f'{img_path_1.split("/")[-1].split(".")[0]}_mask1.jpg')
            cv2.imwrite(f'{attentionmap_dir}/{img_path_2.split("/")[-1].split(".")[0]}_attentionmap2.jpg', attentionmap2)
            attentionmap2s.append(
                f'{img_path_2.split("/")[-1].split(".")[0]}_attentionmap2.jpg'
            )
            cv2.imwrite(f'{attentionmap_dir}/{img_path_2.split("/")[-1].split(".")[0]}_mask2.jpg', mask2)
            masks2.append(
                f'{img_path_2.split("/")[-1].split(".")[0]}_mask2.jpg')

img_path1s = np.concatenate(img_path1s, axis=0)
img_path2s = np.concatenate(img_path2s, axis=0)
day_1s = np.concatenate(day_1s, axis=0)
day_2s = np.concatenate(day_2s, axis=0)

days = np.concatenate(days, axis=0)

df = pd.DataFrame({
    'img_path1': img_path1s,
    'img_path2': img_path2s,
    'patient':np.concatenate( patientids),
    'day_1s': day_1s,
    'day_2s': day_2s,
    'days': days,
    'passage':np.concatenate( passages),
    
    'attentionmap1': attentionmap1s,
    'masks1': masks1,
    'attentionmap2': attentionmap2s,
    'masks2': masks2,
    'label': np.concatenate(test_labels),
    'prob':np.concatenate(test_preds)[:, 1]
})
df.to_csv(f'{dir_name}/attentionmap.csv',
            index=False)



test_acc = accuracy_score(np.concatenate(test_labels),
                            np.concatenate(test_preds).argmax(1))
test_auc = roc_auc_score(np.concatenate(test_labels),
                            np.concatenate(test_preds)[:, 1])
test_report = classification_report(np.concatenate(test_labels),
                                    np.concatenate(test_preds).argmax(1))






f.write(f'test_acc {test_acc} test_auc {test_auc} \n')


f.write(test_report)
f.close()
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







