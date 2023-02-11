import random
import cv2

import albumentations as A

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # white


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    # visualize a single bounding box on the image
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(
        x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)

    cv2.putText(img, text=class_name, org=(x_min, y_min+30),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color,
                thickness=thickness)

    cv2.imshow("test", img)
    cv2.waitKey(0)


image = cv2.imread("./01.jpg")

# dog -> [468.94, 92.01, 171.06, 248.45] 2
# cat -> [3.96, 183.38, 200.88, 214.03] 1
bbox = [3.96, 183.38, 200.88, 214.03]
class_name = 'cat'

visualize_bbox(image, bbox, class_name, color=BOX_COLOR, thickness=2)