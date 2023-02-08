import os
import json
import cv2
import pandas as pd

json_path = "./annotations/instances_default.json"

with open(json_path, "r") as f:
    coco_info = json.load(f)

# 파일 읽기 실패
assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 수집
categories = dict()
for category in coco_info['categories']:
    categories[category["id"]] = category["name"]

# annotation 정보
ann_info = dict()
for annotation in coco_info['annotations']:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    if image_id not in ann_info:
        ann_info[image_id] = {
            "boxes": [bbox], "segmentation": [segmentation],
            "categories": [category_id]
        }
    else:
        ann_info[image_id]["boxes"].append(bbox)
        ann_info[image_id]["segmentation"].append(segmentation)
        ann_info[image_id]["categories"].append(categories[category_id])

file_list = []
box_x_list = []
box_y_list = []
box_w_list = []
box_h_list = []



for image_info in coco_info["images"]:
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']

    file_path = os.path.join("./images", filename)
    img = cv2.imread(file_path)

    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue

    for bbox, segmentation, category in zip(annotation['boxes'],
                                            annotation['segmentation'], annotation['categories']):
        x1, y1, w, h = bbox
        import numpy as np
        for seg in segmentation:
            # print(seg)
            poly = np.array(seg, np.int32).reshape((int(len(seg)/2), 2))
            print(poly)
            poly_img = cv2.polylines(img, [poly], True, (255, 0, 0), 2)

     # CSV 파일 생성

    for bbox, category in zip(annotation['boxes'], annotation['categories']):

        x1, y1, w, h = bbox
        file_list.append(filename)
        box_x_list.append(x1)
        box_y_list.append(y1)
        box_w_list.append(w)
        box_h_list.append(h)
    
    df = pd.DataFrame()
    df['file_name'] = file_list
    df['box_x'] = box_x_list
    df['box_y'] = box_y_list
    df['box_w'] = box_w_list
    df['box_h'] = box_h_list
    
    df.to_csv('./cocoseg.csv')
    #cv2.imwrite(f"./{filename}", rec_img)
    #cv2.imshow("test", poly_img)
    #cv2.waitKey(0)
