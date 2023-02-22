import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from customdata import customDataset
import copy
import os
import sys
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
import pandas as pd
from tqdm import tqdm
import rexnetv1



def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### 0. aug setting -> train val test
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size= 224),
        A.Resize(224, 224),
        A.RandomCrop(width= 180, height= 180),
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.6),
        A.ShiftScaleRotate(shift_limit= 0.05, scale_limit= 0.06,
                                    rotate_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
        A.RandomBrightnessContrast(p= 0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size= 224),
        A.Resize(width= 224, height= 224),
        A.CenterCrop(width= 180, height= 180),
        A.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    ##### 1. Loding classification Dataset
    train_dataset = customDataset("./dataset/train/", transform=train_transform)
    val_dataset = customDataset("./dataset/val/", transform=val_transform)
    test_dataset = customDataset("./dataset/test/", transform=test_transform)


    ##### 2. Data Loader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    net = rexnetv1.ReXNetV1()
    net.load_state_dict(torch.load("./rexnetv1_1.0.pth"))
    net.output[1] = nn.Conv2d(1280, 100, kernel_size=1, stride=1)
    net.to(device)

    #### 4 epoch, optim loss
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    epochs = 20

    best_val_acc = 0.0
    train_step = len(train_loader)
    valid_step = len(val_loader)
    save_path = "best.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                 columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    if os.path.exists(save_path) :
        best_val_acc = max(pd.read_csv('./modelAccuracy.csv')['Accuracy'].tolist())
        net.load_state_dict(torch.load(save_path))

    for epoch in range(epochs):
        running_loss = 0
        val_acc = 0
        train_acc = 0
        val_running_loss = 0

        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='red')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.float().to(device), labels.to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, 1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch [{epoch+1}/{epochs}], loss >> {loss.data:.3f}"
            
        net.eval()
        
        with torch.no_grad():
            valid_bar = tqdm(val_loader, file=sys.stdout, colour="green")
            for data in valid_bar:
                images, labels = data
                images, labels = images.float().to(device), labels.to(device)
                val_outputs = net(images)
                val_loss = loss_function(val_outputs, labels)
                val_running_loss += val_loss.item()
                val_acc += (torch.argmax(val_outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        # 그래프 그리기 위하여 csv로 저장
        dfForAccuracy.loc[epoch, 'epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'train_loss'] = round((running_loss / train_step), 3)
        dfForAccuracy.loc[epoch, 'val_loss'] = round((val_running_loss / valid_step), 3)
        dfForAccuracy.loc[epoch, 'train_acc'] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, 'val_acc'] = round(val_accuracy, 3)

        print(f"epoch [{epoch+1}/{epochs}]"
              f" train loss : {(running_loss / train_step):.3f} val_loss : {(val_running_loss / valid_step):.3f} "
              f"train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")

        # best.pt 저장
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        # csv 저장
        if epoch == epochs - 1:
            dfForAccuracy.to_csv("./modelAccuracy.csv", index = False)

if __name__ == "__main__":
    main()