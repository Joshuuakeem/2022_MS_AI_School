import torch
import os
import glob
import cv2
import xml.etree.ElementTree as ET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model call
model = torch.hub.load('ultralytics/yolov5', 'custom', path="./runs/train/exp_0125/weights/best.pt")

# inference Settings
model.conf = 0.5 # NMS confidence threshold
model.iou = 0.45 # NMS IoU threshold
model.to(device)

# image loader
image_dir = "./dataset/test/images/"
image_path = glob.glob(os.path.join(image_dir, "*.jpg"))
label_dict = {0: "big bus", 1: "big truck", 2: "bus-l-", 3: "bus-s-", 4: "car",
              5: "mid truck", 6: "small bus", 7: "small truck", 8: "truck-l-",
              9: "truck-m-", 10: "truck-s-", 11: "truck-xl-"}

tree = ET.ElementTree()
root = ET.Element("annotations")
"""
<annotations>
</annotations>
"""
seen_count = 0

for img_path in image_path :
    # Image
    img = cv2.imread(img_path)

    # Inference
    results = model(img, size=640)

    # Results
    bbox = results.xyxy[0]

    # image name
    image_name = os.path.basename(img_path)
    
    # image w h
    h, w, c = img.shape

    # xml fix code
    xml_frame = ET.SubElement(root, "image", id="%d"%seen_count, name=image_name,
                              width="%d"%w, height="%d"%h)
    """
    <annotations>
        <image>
        </image>
    </annotations>
    """
    for box in bbox :
    
        # box
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()
        xtl = str(round(x1, 3))
        ytl = str(round(y1, 3))
        xbr = str(round(x2, 3))
        ybr = str(round(y2, 3))

        # class
        class_number = box[5].item()
        class_number_int = int(class_number)
        labels = label_dict[class_number_int]

        # sc number
        sc = box[4].item()

        # bbox xml
        ET.SubElement(xml_frame, "box", label=labels, occluded="0", source="manual",
                      xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, z_order="0")

    seen_count +=1

tree._setroot(root)
tree.write("test.xml", encoding="utf-8")        

