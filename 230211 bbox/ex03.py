import cv2
import random
import albumentations as A

# Define a function to visualize an image


def visualize(image):
    cv2.imshow("visualization", image)
    cv2.waitKey(0)


# Load the image from the disk
image = cv2.imread("./weather.jpg")

# visualize the original image
# visualize(image)

# RandomRain
transform = A.Compose([
    # A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1)
    A.RandomSnow(brightness_coeff=2.5,
                 snow_point_lower=0.1, snow_point_upper=0.3, p=1)
])
transformed = transform(image=image)
visualize(transformed["image"])
