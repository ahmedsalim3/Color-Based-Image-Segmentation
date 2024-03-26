# Color Based Image Segmentation
![img1](https://github.com/amedsalim/Color-Based-Image-Segmentation/assets/126220185/3bcf5e0e-ef13-4050-9e1f-5b5764e1f4e4)
<h6 align="center">View (<a href="https://hsv-segmentation-8665d7d73b94.herokuapp.com/">live demonstration</a>)</h6>

![img2](https://github.com/amedsalim/Color-Based-Image-Segmentation/assets/126220185/ab56161e-b130-4e64-9b35-846abb371181)

This repository contains a Python tool that utilizes OpenCV for real-time color-based image segmentation and filtering. It enhances the contrast of an image using CLAHE before performing color thresholding.

It provides a simple graphical interface with trackbars for adjusting the color range in terms of hue, saturation, and value (HSV). The program loads an image, applies the selected color range as a mask, and displays the original image, the HSV representation, the mask, and the filtered image in a stacked layout. Users can fine-tune the color filtering settings with the trackbars.

## Features
- **Contrast Enhancement:** Utilizes CLAHE with a default clip limit of 1 to enhance image contrast and improve visibility without compromising specific regions' brightness or darkness.
- **Real-time Color Filtering:** Adjust color filtering settings using trackbars.
- **Visualization:** Displays the original image, HSV representation, color mask, and filtered result.
- **Interactive Image Editing:** Supports computer vision and image processing tasks with an intuitive interface.
- **Deployment:** Application deployment using Flask framework with an HTML frontend for user-friendly image processing tasks.

## Getting Started

To use the tool:

1. Clone the repository to your local machine.
2. Ensure you have PyCharm installed.
3. Install the required dependencies using the `requirements.txt` file.
4. To run the tool, open [test.py](https://github.com/amedsalim/Color-Based-Image-Segmentation/blob/main/test.py), change the image path, and run the script.
5. To deploy the application using Flask, run [app.py](https://github.com/amedsalim/Color-Based-Image-Segmentation/blob/main/app.py) and navigate to `http://127.0.0.1:5000`.
