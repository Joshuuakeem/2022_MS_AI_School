from sklearn.model_selection import train_test_split
import os
import random
import shutil

def create_train_val_split_folder(path):
    all_categories = os.listdir(path)
    os.makedirs("./data/train/", exist_ok=True)
    os.makedirs("./data/val/", exist_ok=True)

    for category in sorted(all_categories):
        os.makedirs(f"./data/train/{category}", exist_ok=True)
        all_image = os.listdir(f"./dataset/{category}/")
        for image in random.sample(all_image, int(0.9 * len(all_image))):
            # shutil.move(기존 경로, 새 경로)
            shutil.move(f"./dataset/{category}/{image}",
                        f"./data/train/{category}")

    for category in sorted(all_categories) :
        os.makedirs(f"./data/val/{category}", exist_ok=True)
        all_image = os.listdir(f"./dataset/{category}/")
        for image in all_image : # 이미 90%는 옮겨 감
            shutil.move(f"./dataset/{category}/{image}",
                        f"./data/val/{category}/")

if __name__ == "__main__":
    create_train_val_split_folder("./dataset/")