import os
import glob
import shutil

age20s = [i for i in range(20, 30)]
age30s = [i for i in range(30, 40)]
age40s = [i for i in range(40, 50)]
age50s = [i for i in range(50, 60)]
age60s = [i for i in range(60, 111)]

path = "./crop_part1"
all_image_path = glob.glob(os.path.join(path, "*.jpg"))

for image_path in sorted(all_image_path):
    file_name = image_path.split("\\")[-1]
    get_info = file_name.split("_")
    path = image_path
    ages = int(get_info[0])
    sexs = get_info[1]
    
    if (ages in age20s) and (sexs == "0"):
        new_path = "./dataset/20m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages in age20s) and (sexs == "1"):
        new_path = "./dataset/20w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)

    if (ages in age30s) and (sexs == "0"):
        new_path = "./dataset/30m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages in age30s) and (sexs == "1"):
        new_path = "./dataset/30w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)

    if (ages in age40s) and (sexs == "0"):
        new_path = "./dataset/40m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages in age40s) and (sexs == "1"):
        new_path = "./dataset/40w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)

    if (ages in age50s) and (sexs == "0"):
        new_path = "./dataset/50m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages in age50s) and (sexs == "1"):
        new_path = "./dataset/50w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)

    if (ages in age60s) and (sexs == "0"):
        new_path = "./dataset/60m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages in age60s) and (sexs == "1"):
        new_path = "./dataset/60w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
