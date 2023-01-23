import os
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from customdataset import customDataset
# from timm.loss import LabelSmoothingCrossEntropy
from tqdm import main, tqdm
import pandas as pd
import warnings
import sys

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation
train_trsnform = A.Compose([
    A.Resize(height=224, width=224),
    A.ShiftScaleRotate(),
    A.RGBShift(),
    A.VerticalFlip(),
    A.HorizontalFlip(),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(),
    ToTensorV2()
])

# dataset
train_dataset = customDataset("./data/train/", transform=train_trsnform)
val_dataset = customDataset("./data/val/", transform=val_transform)
# test_dataset = CustomDataset("./data/test/", transform=val_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# model call
net = models.efficientnet_b0(pretrained=True)
net.fc = nn.Linear(in_features=512, out_features=10)
net.to(device)

# hyperparameters
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001) 
epochs = 20

##### Variable declare
best_val_acc = 0.0
train_step = len(train_loader)
valid_step = len(val_loader)
save_path = "best.pt"
dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                             columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

for epoch in range(epochs):
    running_loss = 0
    val_acc = 0
    train_acc = 0
    val_running_loss = 0

##### train code
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

##### valid code
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

    dfForAccuracy.loc[epoch, 'epoch'] = epoch + 1
    dfForAccuracy.loc[epoch, 'train_loss'] = round((running_loss / train_step), 3)
    dfForAccuracy.loc[epoch, 'val_loss'] = round((val_running_loss / valid_step), 3)
    dfForAccuracy.loc[epoch, 'train_acc'] = round(train_accuracy, 3)
    dfForAccuracy.loc[epoch, 'val_acc'] = round(val_accuracy, 3)

    print(f"epoch [{epoch+1}/{epochs}]"
          f" train loss : {(running_loss / train_step):.3f} val_loss : {(val_running_loss / valid_step):.3f} "
          f"train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")

##### best.pt
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(net.state_dict(), save_path)

##### csv
    if epoch == epochs - 1:
        dfForAccuracy.to_csv("./modelAccuracy.csv", index = False)

if __name__ == "__main__":
    main()
