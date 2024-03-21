import cv2
import numpy as np
import os

def enhance_clahe(img, clip_limit=1.0, grid_size=(8, 8)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
    enhanced = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    return enhanced

def stack_images(scale, img_array):
    num_rows = len(img_array)
    num_cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    img_height = img_array[0][0].shape[0]
    img_width = img_array[0][0].shape[1]
    if rows_available:
        for i in range(0, num_rows):
            for j in range(0, num_cols):
                if img_array[i][j].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[i][j] = cv2.resize(img_array[i][j], (0, 0), None, scale, scale)
                else:
                    img_array[i][j] = cv2.resize(img_array[i][j], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[i][j].shape) == 2:
                    img_array[i][j] = cv2.cvtColor(img_array[i][j], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((img_height, img_width, 3), np.uint8)
        horizontal = [image_blank] * num_rows
        for i in range(0, num_rows):
            horizontal[i] = np.hstack(img_array[i])
        vertical = np.vstack(horizontal)
    else:
        for i in range(0, num_rows):
            if img_array[i].shape[:2] == img_array[0].shape[:2]:
                img_array[i] = cv2.resize(img_array[i], (0, 0), None, scale, scale)
            else:
                img_array[i] = cv2.resize(img_array[i], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[i].shape) == 2:
                img_array[i] = cv2.cvtColor(img_array[i], cv2.COLOR_GRAY2BGR)
        horizontal = np.hstack(img_array)
        vertical = horizontal
    return vertical

def nothing(x):
    pass

def create_trackbars(img_path, low_hsv=(10, 39, 64), high_hsv=(86, 255, 255)):
    cv2.namedWindow("trackbars")
    cv2.resizeWindow("trackbars", 640, 240)
    cv2.createTrackbar("Low Hue", "trackbars", low_hsv[0], 180, nothing)
    cv2.createTrackbar("High Hue", "trackbars", high_hsv[0], 180, nothing)
    cv2.createTrackbar("Low Sat", "trackbars", low_hsv[1], 255, nothing)
    cv2.createTrackbar("High Sat", "trackbars", high_hsv[1], 255, nothing)
    cv2.createTrackbar("Low Val", "trackbars", low_hsv[2], 255, nothing)
    cv2.createTrackbar("High Val", "trackbars", high_hsv[2], 255, nothing)

    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, img_hsv


def apply_mask(img, img_hsv):
    while True:
        low_hue = cv2.getTrackbarPos("Low Hue", "trackbars")
        high_hue = cv2.getTrackbarPos("High Hue", "trackbars")
        low_sat = cv2.getTrackbarPos("Low Sat", "trackbars")
        high_sat = cv2.getTrackbarPos("High Sat", "trackbars")
        low_val = cv2.getTrackbarPos("Low Val", "trackbars")
        high_val = cv2.getTrackbarPos("High Val", "trackbars")
        lower_bound = np.array([low_hue, low_sat, low_val])
        upper_bound = np.array([high_hue, high_sat, high_val])
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(img, img, mask=mask)
        stack = stack_images(0.6, ([img, img_hsv], [mask, result]))
        cv2.imshow("hsv explorer", stack)

        output_folder = "results"
        os.makedirs(output_folder, exist_ok=True)
        stacked_image_path = os.path.join(output_folder, "stacked-image.png")
        cv2.imwrite(stacked_image_path, stack)
        segmented_image_path = os.path.join(output_folder, "segmented_image.png")
        cv2.imwrite(segmented_image_path, result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
