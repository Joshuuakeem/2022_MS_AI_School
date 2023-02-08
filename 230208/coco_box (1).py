import os
import json
import cv2

# json path
json_path = "./annotations/instances_default.json"

# json 파일 읽기

with open(json_path, "r") as f:
    coco_info = json.load(f)

# print(coco_info)

assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 정보 수집
categories = dict()
for category in coco_info['categories']:
    print(category)
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

for image_info in coco_info['images']:
    # print(image_info)
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']
    print(filename, width, height, img_id)
