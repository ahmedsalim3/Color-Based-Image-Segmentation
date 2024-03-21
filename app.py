from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

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

def apply_mask_once(img, img_hsv, low_hsv, high_hsv):
    lower_bound = np.array([low_hsv[0], low_hsv[1], low_hsv[2]])
    upper_bound = np.array([high_hsv[0], high_hsv[1], high_hsv[2]])
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(img, img, mask=mask)
    return mask, result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            enhanced_img = enhance_clahe(img)
            img_hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
            low_hsv = (10, 39, 64)
            high_hsv = (86, 255, 255)

            mask, result = apply_mask_once(enhanced_img, img_hsv, low_hsv, high_hsv)
            stacked_images = stack_images(0.6, [[img, img_hsv], [mask, result]])

            _, stacked_image_png = cv2.imencode('.png', stacked_images)
            stacked_image_base64 = base64.b64encode(stacked_image_png)
            stacked_image_b64_string = "data:image/png;base64," + stacked_image_base64.decode()

            return render_template('index.html', stacked_image=stacked_image_b64_string)
    return render_template('index.html')

@app.route('/update_hsv', methods=['POST'])
def update_hsv():
    try:
        data = request.json
        low_hue = int(data['low_hue'])
        high_hue = int(data['high_hue'])
        low_sat = int(data['low_sat'])
        high_sat = int(data['high_sat'])
        low_val = int(data['low_val'])
        high_val = int(data['high_val'])
        img_base64 = data['file']
        img_nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
        img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
        enhanced_img = enhance_clahe(img)
        img_hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
        mask, result = apply_mask_once(enhanced_img, img_hsv, (low_hue, low_sat, low_val), (high_hue, high_sat, high_val))
        stacked_images = stack_images(0.6, [[img, img_hsv], [mask, result]])
        
        _, stacked_image_png = cv2.imencode('.png', stacked_images)
        stacked_image_base64 = base64.b64encode(stacked_image_png)
        _, segmented_image_png = cv2.imencode('.png', result)
        segmented_image_base64 = base64.b64encode(segmented_image_png)
        stacked_image_b64_string = "data:image/png;base64," + stacked_image_base64.decode()
        segmented_image_b64_string = "data:image/png;base64," + segmented_image_base64.decode()

        return jsonify({'stacked_image': stacked_image_b64_string, 'segmented_image': segmented_image_b64_string})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run()