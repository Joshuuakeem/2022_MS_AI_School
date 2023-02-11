import torch.nn as nn
from collections import defaultdict
import torch
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import copy
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import models
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class catvsdogDataset(Dataset):
    def __init__(self, image_file_path, transform=None):
        self.image_file_paths = glob.glob(
            os.path.join(image_file_path, "*", "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_file_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_temp = image_path.split('\\')
        label_temp = label_temp[3]
        if "cat" == label_temp:
            label = 0
        elif "dog" == label_temp:
            label = 1
        print(image_path, label)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self.image_file_paths)


train_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

train_dataset = catvsdogDataset("./dataset/train/", transform=train_transform)
val_dataset = catvsdogDataset("./dataset/val/", transform=val_transform)

def visualize_augmentation(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def calculate_accuracy(output, target):
    output = target.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metrics_name} : {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

params = {
    "model": "resnet18",
    "device": device,
    "lr": 0.001,
    "batch_size": 64,
    "num_workers": 4,
    "epoch": 10,
}

model = models.__dict__[params["model"]](pretrained=True)
model.fc = nn.Linear(512, 2)
model = model.to(params["device"])

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"],
                          shuffle=True, num_workers=params["num_workers"])

val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False,
                        num_workers=params["num_workers"])

def save_model(model, save_dir, file_name='last.pt'):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print("multi GPU activate")
        torch.save(model.module.state_dict(), output_path)
    else:
        print("single GPU activate")
        torch.save(model.state_dict(), output_path)

def train(train_loader, model, criterion, optimizer, epoch, params, save_dir):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    if __name__ == "__main__":
        for i, (image, target) in enumerate(stream):
            images = image.to(params["device"])
            targets = target.to(params["device"])

            output = model(images)
            loss = criterion(output, targets)
            accuracy = calculate_accuracy(output, targets)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stream.set_description(
                "Epoch : {epoch}. Train. {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor)
            )

    save_model(model, save_dir)

def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(stream):
            images = image.to(params["device"])
            targets = target.to(params["device"])

            output = model(images)
            loss = criterion(output, targets)
            accuracy = calculate_accuracy(output, targets)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)

            stream.set_description(
                "Epoch : {epoch}. Train. {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor)
            )

save_dir = "./weights"

if __name__ == "__main__":
    for epoch in range(1, params["epoch"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params, save_dir)
        validate(val_loader, model, criterion, epoch, params)