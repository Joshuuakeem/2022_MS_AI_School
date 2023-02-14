# train loop
# val loop
# 모델 save
# 평가 함수
import torch
import os
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from metric_monitor_temp import MetricMonitor


def calculate_acc(output, target):
    # 평가 함수
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((output == target).sum(dim=0), output.size(0)).item()


def save_model(model, save_dir, file_name="last.pt"):
    # save model
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print("멀티 GPU 저장 !! ")
        torch.save(model.module.state_dict(), output_path)
    else:
        print("싱글 GPU 저장 !! ")
        torch.save(model.state_dict(), output_path)

# train loop
def train(train_loader, model, criterion, optimizer, epoch, device)
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for batch_idx, (image, target) in enumerate(train_loader) :
        images = image.to(device)
        target  = target.to(device)
        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_acc(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# val loop
def validate(val_loader, model, criterion, optimizer, epoch, device)
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    for batch_idx, (image, target) in enumerate(val_loader) :
        images = image.to(device)
        target  = target.to(device)
        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_acc(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.rest()
    def rest(self):
        self.metrics = defaultdict(lambda : {'val':0, "count":0, "avg":0})

    def update(self, metrics_name, val):
        metrics = self.metrics[metrics_name]

        metrics["val"] += val
        metrics["count"] += 1
        metrics["avg"] = metrics["val"] / metrics["count"]
    
    def __str__(self):
        return " | ".join(
            [
                "{metric_name} : {avg.{float_precision}f}".format(
                    metric_name=metric_name, avg=metrics["avg"],
                    float_precision=self.float_precision
                )
                for(metric_name, metrics) in self.metrics.item()
            ]
        )
