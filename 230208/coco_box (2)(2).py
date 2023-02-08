import xml.etree.ElementTree as ET
import os
import json
import cv2
import pandas as pd

# json path
json_path = "./annotations/instances_default_box.json"

# json 파일 읽기
with open(json_path, "r") as f:
    coco_info = json.load(f)

# print(coco_info)

assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 정보 수집
categories = dict()
for category in coco_info['categories']:
    # print(category)
    categories[category["id"]] = category["name"]

# print("categories info >> ", categories)

# annotation 정보 수집
ann_info = dict()
for annotation in coco_info['annotations']:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    # print(
    #     f"image_id : {image_id}, category_id : {category_id} , bbox : {bbox}")

    if image_id not in ann_info:
        ann_info[image_id] = {
            "boxes": [bbox], "categories": [category_id]
        }
    else:
        ann_info[image_id]["boxes"].append(bbox)
        ann_info[image_id]["categories"].append(categories[category_id])

# print("ann_info >> ", ann_info)
# import xml.etree.ElementTree as ET

tree = ET.ElementTree()
root = ET.Element("annotations")

# CSV 파일 생성
columns = ['file_name', 'x1', 'y1', 'w', 'h']
df = pd.DataFrame(columns = columns)
data = dict()
for i in columns:
    data[i] = []

for i, image_info in enumerate(coco_info['images']):
    # print(image_info)

    # xml file save folder
    os.makedirs("./xml_folder/", exist_ok=True)
    xml_save_path = "./xml_folder/test.xml"

    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']
    # print(filename, width, height, img_id)
    xml_frame = ET.SubElement(
        root, "image", id=str(i), name=filename, width="%d" % width,
        height="%d" % height)

    # 이미지 가져오기 위한 처리
    file_path = os.path.join("./images", filename)
    img = cv2.imread(file_path)
    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue
    # box category
    label_list = {1: "kiwi", 2: "apply"}

    for bbox, category in zip(annotation['boxes'], annotation['categories']):
        x1, y1, w, h = bbox
        print(filename, x1, y1, w, h)
        ET.SubElement(xml_frame, "box", label="Kiwi", occluded="0",
                      source="manual", x1=str(x1), y1=str(y1), w=str(w), h=str(h), z_order="0")
        rec_img = cv2.rectangle(img, (int(x1), int(y1)),
                                (int(x1+w), int(y1+h)), (225, 0, 255), 2)
        data ['file_name'].append(filename)
        data ['x1'].append(x1)
        data ['y1'].append(y1)
        data ['w'].append(w)
        data ['h'].append(h)

for i in columns:
    df[i] = data[i]

df.to_csv('./cocobox.csv')



    #cv2.imshow("test", rec_img)
    #cv2.waitKey(0)
    #tree._setroot(root)
    #tree.write(xml_save_path, encoding='utf-8')
    #print("xml ok ")
