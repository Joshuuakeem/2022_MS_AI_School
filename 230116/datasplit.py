"""
label -> dark, green, light, medium
org_data
image
    - labels image.png

test
    - labels image.png

dataset
    train
        - dark
        - green
        - light
        - medium
    val
        - dark
        -green
        -light
        -medium
"""
import os
import glob
import shutil

image_folder_path = "./image"
image_path = glob.glob(os.path.join(image_folder_path, "*.png"))
# print(image_path)

for path in image_path:
    # print(path)
    ##### 파일이름만 추출
    file_name = os.path.basename(path)
    # print(file_name)
    # file move
    if "dark" in file_name :
        # print(file_name)
        # 이동할 폴더 생성
        os.makedirs("./dataset/train/dark", exist_ok=True)
        shutil.move(path, f"./dataset/train/dark/{file_name}")
    elif "green" in file_name :
        # print(file_name)
        os.makedirs("./dataset/train/green", exist_ok=True)
        shutil.move(path, f"./dataset/train/green/{file_name}")
    elif "light" in file_name :
        #print(file_name)
        os.makedirs("./dataset/train/light", exist_ok=True)
        shutil.move(path, f"./dataset/train/light/{file_name}")
    elif "medium" in file_name :
        #print(file_name)
        os.makedirs("./dataset/train/medium", exist_ok=True)
        shutil.move(path, f"./dataset/train/light/{file_name}")



image_folder_path = "./test_image"
image_path = glob.glob(os.path.join(image_folder_path, "*.png"))
# print(image_path)

for path in image_path:
    # print(path)
    ##### 파일이름만 추출
    file_name = os.path.basename(path)
    # print(file_name)
    # file move
    if "dark" in file_name :
        # print(file_name)
        # 이동할 폴더 생성
        os.makedirs("./dataset/val/dark", exist_ok=True)
        shutil.move(path, f"./dataset/val/dark/{file_name}")
    elif "green" in file_name :
        # print(file_name)
        os.makedirs("./dataset/val/green", exist_ok=True)
        shutil.move(path, f"./dataset/val/green/{file_name}")
    elif "light" in file_name :
        #print(file_name)
        os.makedirs("./dataset/val/light", exist_ok=True)
        shutil.move(path, f"./dataset/val/light/{file_name}")
    elif "medium" in file_name :
        #print(file_name)
        os.makedirs("./dataset/val/medium", exist_ok=True)
        shutil.move(path, f"./dataset/val/light/{file_name}")


