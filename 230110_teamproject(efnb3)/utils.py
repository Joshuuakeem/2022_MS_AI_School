import pandas as pd
import os
import glob

def image_rename(csv_path, image_path, save_dir, mode):
    df = pd.read_csv(csv_path)
    file_names = df["filename"]
    new_file_names = df["Classes"]
    
    for i, (file_name, new_file_name) in enumerate(zip(file_names, new_file_names)):
        
        os.makedirs(os.path.join(save_dir, mode, new_file_name), exist_ok=True)
        
        new_path = os.path.join(save_dir, mode, new_file_name)
        old_name = os.path.join(image_path, file_name)
        os.rename(old_name, os.path.join(new_path, new_file_name+f"_{i+1}.png"))


image_rename("./val.csv", "./images/", "./dataset", "val")
image_rename("./test.csv", "./images/", "./dataset", "test")
image_rename("./train.csv", "./images/", "./dataset", "train")