import cv2

import os
import random

folder_in = "/home/sonlt373/Desktop/SoNg/FF_project/data/20240520_data"

folder_out = "/home/sonlt373/Desktop/SoNg/FF_project/data/20240520_data_clean"

for path in os.listdir(folder_in):
    img_path = os.path.join(folder_in, path)
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    print(image.shape)

    crop_image = image[0:height, int(random.uniform(0, 0.1)* width) : int(random.uniform(0.8, 0.92)* width)]

    if random.uniform(0, 1) > 0.5:
        crop_image = cv2.rotate(crop_image, cv2.ROTATE_90_CLOCKWISE)

    if random.uniform(0, 1) > 0.5:
        crop_image = cv2.rotate(crop_image, cv2.ROTATE_180)

    cv2.imwrite(os.path.join(folder_out, path), crop_image)
