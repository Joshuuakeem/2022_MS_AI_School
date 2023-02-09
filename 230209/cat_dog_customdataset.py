"""
dataset 
    - train
        - cat 
            - cat1.jpg....
        - dog
    - val  
        - cat 
        - dog
    - test
        - cat 
        - dog
"""
from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image

label_dict = {"cat": 0, "dog": 1}


class cat_dog_mycustomdataset(Dataset):
    def __init__(self, data_path):
        # csv folder 읽기, 변환 할당, 데이터 필터링 등 과 같은 초기 논리가 발생
        self.all_data_path = glob.glob(os.path.join(data_path, '*', '*.jpg'))

    def __getitem__(self, index):
        image_path = self.all_data_path[index]
        img = Image.open(image_path).convert('RGB')
        label_temp = image_path.split('/')
        # ['.', 'dataset', 'train', 'cat', 'cat.2718.jpg']
        label = label_dict[label_temp[2].split('\\')[1]]
        return img, label

    def __len__(self):
        # 전체 데이터 길이 반환
        return len(self.all_data_path)


test = cat_dog_mycustomdataset("./dataset/train/")

for i in test:
    print(i)
    pass
