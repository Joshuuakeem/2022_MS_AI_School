import copy
import os.path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from customdataset import my_dataset
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### 0. aug setting -> train val test
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09,
                           rotate_limit=25, p=1),
        A.Resize(width=224, height=224),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.7),
        A.RandomShadow(p=1),
        A.RandomFog(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    ##### 1. Loding classification Dataset
    train_dataset = my_dataset("./dataset/train/", transform=train_transform)
    val_dataset = my_dataset("./dataset/valid/", transform=val_transform)
    test_dataset = my_dataset("./dataset/test/", transform=test_transform)

    ###### def visulize_augmentations()
    def visulize_augmentations(dataset, idx=0, samples=40, cols=5):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform
                                       if not isinstance(
                t, (A.Normalize, ToTensorV2)
            )])
        rows = samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i in range(samples):
            image, _ = dataset[10]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    ##### 2. Data Loader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)
    ##### 3. model loader
    # import torch
    # HUB_URL = "SharanSMenon/swin-transformer-hub:main"
    # MODEL_NAME = "swin_tiny_patch4_window7_224"
    # model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True) # load from torch hub

    import torch.nn as nn
    net = models.swin_t(weights='IMAGENET1K_V1')
    net.head = nn.Linear(in_features=768, out_features=450)
    net.to(device)

    #### 4 epoch, optim loss
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    epochs = 100

    best_val_acc = 0.0

    train_steps = len(train_loader)
    valid_steps = len(val_loader)
    save_path = "best.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                 columns=['Epoch', 'Accuracy'])

    if os.path.exists(save_path) :
        best_val_acc = max(pd.read_csv('./modelAccuracy.csv')['Accuracy'].tolist())
        net.load_state_dict(torch.load(save_path))

    for epoch in range(epochs) :
        runing_loss = 0
        val_acc = 0
        train_acc = 0

        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='red')
        for step, data in enumerate(train_bar) :
            images , labels = data
            images , labels = images.to(device) , labels.to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            runing_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1} / {epoch}], loss{loss.data:.3f}"

        net.eval()
        with torch.no_grad() :
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for data in valid_bar :
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, 'Epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'Accuracy'] = round(val_accuracy, 3)
        print(f"epoch [{epoch+1}/{epochs}]"
              f" train loss : {(runing_loss / train_steps):.3f} "
              f"train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}"
        )

        if val_accuracy > best_val_acc :
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        if epoch == epochs - 1 :
            dfForAccuracy.to_csv("./modelAccuracy.csv" , index=False)

if __name__ == '__main__':
    main()