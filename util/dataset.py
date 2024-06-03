from torch.utils.data import DataLoader, Dataset

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
        day = row_1["day"]
        img_path_2 = f'{row_1["img2"]}'
        img_1 = Image.open(img_path_1).convert("RGB")
        img_2 = Image.open(img_path_2).convert("RGB")
        label = row_1["label"]

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, label, day
