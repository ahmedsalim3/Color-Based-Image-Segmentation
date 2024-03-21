import cv2
from hsv_explorer import *

test_img = cv2.imread("static/audi.jpeg",1)
enhanced_img = enhance_clahe(test_img)
img, img_hsv = create_trackbars(enhanced_img,(42,48,0), (79, 255, 255))
apply_mask(img, img_hsv)
