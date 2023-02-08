import os
import glob
import cv2
from xml.etree.ElementTree import parse

# xml 파일 찾을 수 있는 함수 제작


def find_xml_file(xml_folder_path):
    all_root = []
    print(",,", xml_folder_path)
    for (path, dir, files) in os.walk(xml_folder_path):
        print("...", path, dir, files)
        for filename in files:
            # image.xml -> .xml
            ext = os.path.splitext(filename)[-1]
            print(filename)
            if ext == ".xml":
                root = os.path.join(path, filename)
                # ./cavt_annotations/annotations.xml
                all_root.append(root)
            else:
                print("no xml file..")
                break
    return all_root


xml_dirs = find_xml_file("./cvat_annotations/")
print(xml_dirs)
# ['./cvat_annotations/annotations.xml']
