import cv2
import torch
import os
import glob
from torch.utils.data import Dataset

class my_dataset(Dataset) :
    def __init__(self, path, transform=None):
        self.path_all = glob.glob(os.path.join(path, "*", "*.jpg"))
        self.transform = transform
        self.label_dict ={"african-wildcat" : 0,
                          "blackfoot-cat" : 1,
                          "chinese-mountain-cat" : 2,
                          "domestic-cat" : 3,
                          "european-wildcat" : 4,
                          "jungle-cat" : 5,
                          "sand-cat" :6,
                          }

    def __getitem__(self, item):
        # 1. image path [] -> img_path
        img_path = self.path_all[item]

        # 2. get label
        folder_name = img_path.split("\\")[1]
        label = self.label_dict[folder_name]

        # 3. get image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Augment an image
        if self.transform is not None :
            image = self.transform(image=image)["image"]

        # 5. return in image, label
        return image, label

    def __len__(self):
        return len(self.path_all)

# if __name__ == "__main__":
#     test = my_dataset("./dataset/train/", transform=None)
#     for i in test :
#         print(i)