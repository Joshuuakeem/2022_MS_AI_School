import copy

import albumentations as A
from albumentations.pytorch import ToTensorV2
from customdata import my_dataset
import matplotlib.pyplot as plt

def aug_function(mode_flog) :
    if mode_flog == "train" :
        train_transform = A.Compose([
            A.SmallestMaxSize(max_size=400),
            A.Resize(width=256,height=256),
            A.RandomCrop(height=224,width=224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.06,
                               rotate_limit=20, p=1),
            A.RGBShift(r_shift_limit=17, g_shift_limit=17, b_shift_limit=17,
                       p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomShadow(p=1),
            A.RandomFog(p=1),
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return train_transform
    elif mode_flog == "val":
        val_transform = A.Compose([
            A.SmallestMaxSize(max_size=400),
            A.Resize(width=256, height=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        return val_transform


def visualize_augmentations(dataset, idx=0, cols=5) :
    dataset = copy.deepcopy(dataset)
    samples = 5
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(
        t, (A.Normalize, ToTensorV2)
    )])
    rows = samples // cols
    figure , ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))

    for i in range(samples) :
        image , _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     train_aug = aug_function(mode_flog="train")
#     train_dataset = my_dataset("./dataset/train/", transform=train_aug)
#     visualize_augmentations(train_dataset)