import os
import json
import cv2

json_path = "./annotations/instances_default.json"

with open(json_path, "r") as f:
    coco_info = json.load(f)

# 파일 읽기 실패
assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 수집
categories = dict()
for category in coco_info['categories']:
    categories[category["id"]] = category["name"]

print(categories)

# annotation 정보
ann_info = dict()
for annotation in coco_info['annotations']:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    print(image_id, category_id, bbox, segmentation)
