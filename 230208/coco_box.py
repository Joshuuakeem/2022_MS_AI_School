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

print("categories info >> ", categories)
