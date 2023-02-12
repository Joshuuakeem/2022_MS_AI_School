import os
import glob
import argparse
# image_pretreatment.py
# 오렌지 : Orange
# 자몽 : grapefruit
# 레드향 : Kanpei
# 한라봉 : Dekopon


def image_file_check(opt):
    # image-folder-path
    # image_folder_path
    image_path = opt.image_folder_path
    # 각 폴더별 데이터 양 체크 #
    """
    image 
        - 자몽
            - xxx.jpg
        - 레드향
        - 한라봉 
    image_path -> ./image/orange/*.jpg
    """
    all_data = glob.glob(os.path.join(image_path, "*", "*.jpg"))
    print("전체 데이터 갯수 : ", len(all_data))
    # 오렌지
    ornage_data = glob.glob(os.path.join(image_path, "orange", "*.jpg"))
    print("오렌지 데이터 갯수 >> ", len(ornage_data))
    # 자몽
    grapefruit_data = glob.glob(os.path.join(
        image_path, "grapefruit", "*.jpg"))
    print("자몽 데이터 갯수 >> ", len(grapefruit_data))
    # 레드향
    kanpei_data = glob.glob(os.path.join(image_path, "kanpei", "*.jpg"))
    print("레드향 데이터 갯수 >> ", len(kanpei_data))
    # 한라봉
    dekopon_data = glob.glob(os.path.join(image_path, "dekopon", "*.jpg"))
    print("한라봉 데이터 갯수 >> ", len(dekopon_data))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder-path", type=str, default="./image")
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    image_file_check(opt)
