# OrgaPredictor
<img alt="image" src="https://github.com/ghdwnsgh852/orgapredictor/assets/29062695/3e75b61d-2955-440c-a271-bcd8d8766852"><br>

<img alt="image" src="https://github.com/ghdwnsgh852/orgapredictor/assets/29062695/a44993d2-809f-468c-a237-20e9d6d59276">



## Introduction

The complexity and heterogeneity of cancer pathogenesis necessitate the development of personalized in-vitro models. Patient-derived organoids (PDOs) offer a promising approach, maintaining patient-specific mutations and enabling individualized therapeutic strategies. However, the establishment success rates of organoids vary significantly across cancer types, posing a challenge especially for large-scale application. 

This study introduces an advanced AI-based model utilizing deep learning to predict the success of organoid establishment at an early stage using microscopic images. Employing a dual-image input approach combined with Masked Autoencoder (MAE) pretraining, Vision Transformer (ViT) models demonstrated high predictive accuracy, surpassing human performance in several metrics. 

The ViT-Dual-MAE-Finetune and Linearprobe models were particularly effective, achieving the highest Area Under the Curve (AUC) scores of 0.88 with total test set. The ViT-Dual-MAE-Linearprobe model consistently improved across organoid passages, reaching a peak AUC of 0.98 in Passage 2. Furthermore, the models performed well with HER2 and TNBC subtypes. Attention map analysis revealed that successful predictions often focused on the edges of individual organoids, suggesting that these features may be critical indicators of growth. 

These findings highlight the potential of AI models in enhancing the efficiency and scalability of organoid-based research, paving the way for significant advancements in precision medicine and personalized cancer treatment.


# Using OrgaPreditor

### Fine-tuning OrgaPreditor
To fine-tune OrgaPredictor with your own dataset:

```
python3 main.py --lr 1e-5 --batch_size 2 --num_epoch 100 --data_dir YOUR_DATASET_PATH --ckpt_dir DOWNL
OADED_MODEL_PATH --result_dir DESIRABLE_RESULT_PATH --mode "train" --cuda_devices 0 --train_continue "on"
```

## Contact
Junho Hong(ra9027@yonsei.ac.kr) Taeyong Kweon(kmskty3@yuhs.ac)

