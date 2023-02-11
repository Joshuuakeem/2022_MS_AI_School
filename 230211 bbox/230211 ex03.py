import cv2
import random
import albumentations as A

# Define a function to save an image

def save_image(image, filename):
    cv2.imwrite(filename, image)


# Load the image from the disk
image = cv2.imread("./weather.jpg")

# RandomRain
transform = A.Compose([
    # A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1)
    # A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.1, snow_point_upper=0.4, p=1)
    # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.3, p=1)
    # A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1)
    A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1)
])
transformed = transform(image=image)

save_image(transformed["image"], "visualization.jpg")