import os
import glob
import cv2
from xml.etree.ElementTree import parse

# xml 파일 찾을 수 있는 함수 제작


def find_xml_file(xml_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(xml_folder_path):
        for filename in files:
            # image.xml -> .xml
            ext = os.path.splitext(filename)[-1]
            if ext == ".xml":
                root = os.path.join(path, filename)
                # ./cavt_annotations/annotations.xml
                all_root.append(root)
            else:
                print("no xml file..")
                break
    return all_root


xml_dirs = find_xml_file("./cvat_annotations/")
# ['./cvat_annotations/annotations.xml']

for xml_dir in xml_dirs:
    tree = parse(xml_dir)
    root = tree.getroot()
    img_metas = root.findall("image")
    for img_meta in img_metas:
        # xml 에 기록된 이미지 이름
        image_name = img_meta.attrib['name']

        # image size info
        img_width = int(img_meta.attrib['width'])
        img_height = int(img_meta.attrib['height'])

        # box meta info
        box_metas = img_meta.findall("box")

        for box_meta in box_metas:
            box_label = box_meta.attrib['label']
            box = [
                int(float(box_meta.attrib['xtl'])),
                int(float(box_meta.attrib['ytl'])),
                int(float(box_meta.attrib['xbr'])),
                int(float(box_meta.attrib['ybr']))
            ]
            print(box[0], box[1], box[2], box[3])
