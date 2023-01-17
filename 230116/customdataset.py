import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class my_dataset(Dataset) :
    def __init__(self, path, transform=None):
        # path -> ./dataset/train/
        self.all_path = glob.glob(os.path.join(path, "*", "*.png"))
        self.transform = transform
        self.label_dict = {"dark":0, "green":1, "light":2, "medium":3}

    def __getitem__(self, item):
        file_data = self.all_path[item]
        # image open
        image = Image.open(file_data).convert('RGB')
        # label
        temp = file_data.split("\\")[1]
        label = self.label_dict[temp]
        # aug
        if self.transform is not None :
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.all_path)

