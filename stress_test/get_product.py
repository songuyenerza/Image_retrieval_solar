from ultralytics import YOLO
import cv2
import numpy as np
import random
import os
from tqdm import tqdm

def crop_mask(image, mask):
    res = cv2.bitwise_and(image,image,mask = mask)
    kernel = np.ones((25, 25), np.uint8) 

    mask = cv2.dilate(mask, kernel, iterations=5) 
        # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming you want to crop to the largest contour if there are multiple
        # This finds the bounding box of the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop the result image to this bounding box
        # This step removes the black area by cropping to the ROI
        cropped_res = res[y:y+h, x:x+w]

        return cropped_res
    else:
        return None


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

# Define an inference source
folder_input = '/home/sonlt373/Desktop/SoNg/FF_project/data/20240520_data_clean_base'


folder_save = "/home/sonlt373/Desktop/SoNg/FF_project/data/20240520_Data_product"
model = YOLO('/home/sonlt373/Desktop/SoNg/FF_project/dev/FF_product_retrieval/stress_test/weights/yolov8s_640_product_v0.pt')  

for path in tqdm(os.listdir(folder_input)[:]):
    img_path = os.path.join(folder_input, path)


    image = cv2.imread(img_path)


    # Run inference on an image
    prediction = model(image, 
                    device='cpu', 
                    retina_masks=True, 
                    imgsz=640, 
                    conf=0.8, 
                    iou=0.8)

    # Run inference on an image
    try:

        masks = prediction[0].masks.data

        masks = masks.cpu()
        annotations = np.array(masks)
        msak_sum = annotations.shape[0]
        height = annotations.shape[1]
        weight = annotations.shape[2]

    except:
        annotations = []


    # color = (0, 255, 0)
    # image_with_masks = image
    # image_with_masks = image_with_masks.astype(np.float32)

    for i, mask_i in enumerate(annotations):
        # image = overlay(image, mask_i, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), alpha=0.3)
        mask_i = mask_i.astype(np.uint8)

        cropped_mask = crop_mask(image, mask_i)
        try:
            cv2.imwrite(os.path.join(folder_save, f"{path[:-4]}_mask_{i}.jpg"), cropped_mask)
        except:
            pass
