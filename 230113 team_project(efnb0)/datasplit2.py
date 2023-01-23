import os
import glob
import shutil

all_path = glob.glob(os.path.join("./AFAD-Full", "*", "*", "*.jpg"))
for path in all_path:
    temp = path.split("\\")
    ages = temp[1]
    sexs = temp[2]
    file_name = temp[-1]
    if (ages.startswith("2")) and (sexs == "111"):
        new_path = "./dataset/20m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages.startswith("2")) and (sexs == "112"):
        new_path = "./dataset/20w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
        
    if (ages.startswith("3")) and (sexs == "111"):
        new_path = "./dataset/30m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages.startswith("3")) and (sexs == "112"):
        new_path = "./dataset/30w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
        
    if (ages.startswith("4")) and (sexs == "111"):
        new_path = "./dataset/40m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages.startswith("4")) and (sexs == "112"):
        new_path = "./dataset/40w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
        
    if (ages.startswith("5")) and (sexs == "111"):
        new_path = "./dataset/50m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages.startswith("5")) and (sexs == "112"):
        new_path = "./dataset/50w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
        
    if (ages.startswith("6")) and (sexs == "111"):
        new_path = "./dataset/60m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages.startswith("6")) and (sexs == "112"):
        new_path = "./dataset/60w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
        
    if (ages.startswith("7")) and (sexs == "111"):
        new_path = "./dataset/60m/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
    if (ages.startswith("7")) and (sexs == "112"):
        new_path = "./dataset/60w/"
        os.makedirs(new_path, exist_ok=True)
        os.rename(path, new_path+file_name)
        

