from PIL import Image
from matplotlib import pyplot as plt
import cv2

import numpy as np
import time

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

# 기존 torchvision Data pipeline
# 1. dataset class -> image loader -> transform


class CatDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        # 원래 라면 image label
        # Read an image with PIL
        image = Image.open(file_path).convert("RGB")

        # transform time check
        start_tiem = time.time()
        if self.transform:
            image = self.transform(image)
        end_tiem = (time.time() - start_tiem)

        return image, end_tiem

    def __len__(self):
        return len(self.file_paths)


# data aug transforms
# # train
# train_transform = transforms.Compose([
#     transforms.Pad(padding=10),
#     transforms.Resize((256, 256)),
#     transforms.CenterCrop(size=(30)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ColorJitter(brightness=0.2, contrast=0.3),
#     transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
#     transforms.RandomPerspective(distortion_scale=0.7, p=0.5),
#     transforms.ToTensor()
# ])

# # val
# val_transform = transforms.Compose([
#     transforms.Pad(padding=10),
#     transforms.Resize((256, 256)),
#     transforms.CenterCrop(size=(30)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ColorJitter(brightness=0.2, contrast=0.3),
#     transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
#     transforms.ToTensor()

# ])


torchvision_transform = transforms.Compose([
    # transforms.Pad(padding=10),
    # transforms.Resize((256, 256)),
    # transforms.CenterCrop(size=(30)),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.ColorJitter(brightness=0.2, contrast=0.3),
    # transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
    # transforms.RandomPerspective(distortion_scale=0.7, p=0.5),
    # transforms.RandomRotation(degrees=(0, 100)),
    # transforms.RandomAffine(
    #     degrees=(30, 60), translate=(0.1, 0.3), scale=(0.5, 0.7)),
    transforms.ElasticTransform(alpha=255.0),
    transforms.ToTensor()
])

cat_dataset = CatDataset(file_paths=["cat01.jpeg"],
                         transform=torchvision_transform)

# from matplotlib import pyplot as plt
total_time = 0
for i in range(100):
    image, end_ime = cat_dataset[0]
    total_time += end_ime

print("torchvision tiem/image >> ", total_time*10)

plt.figure(figsize=(10, 10))
plt.imshow(transforms.ToPILImage()(image).convert("RGB"))
plt.show()
