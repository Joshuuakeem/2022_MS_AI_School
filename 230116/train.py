import torch
import torch.optim as optim
import torch.nn as nn

from utils import train
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
from customdataset import my_dataset

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.ToTensor()
])

# dataset dataloader
train_data = my_dataset("./dataset/train/", transform=train_transform)
val_data = my_dataset("./dataset/val/", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=True)

# model call
net = models.__dict__["vgg19"](pretrained=True)
# print(net)
net.classifier[6] = nn.Linear(in_features=4096, out_features=4)
net.to(device)
# print(net)


#hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
epoch_number = 100

if __name__ == "__main__" :
    train(epoch_number, net,train_loader, val_loader,criterion,
          optimizer, scheduler, device)
