import cv2
import math
import imutils

import numpy as np
from skimage import transform
from skimage._shared.utils import convert_to_float
import Config as cfg

from skimage.transform import radon, warp, AffineTransform
from matplotlib import pyplot as plt

def is_character(image, bounding_box):
    image_h, image_w = image.shape[:2]

    area = bounding_box[4]
    relative_area = area / (image_w * image_h)

    if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
        print(f"CC area: {area}")
        print(f"CC relative area: {relative_area}")

    if not cfg.CHARACTER_AREA_MIN<=relative_area<=cfg.CHARACTER_AREA_MAX:
        if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
            if cfg.CHARACTER_AREA_MIN > relative_area:
                print(f"CC area too small: {cfg.CHARACTER_AREA_MIN} > {relative_area}")
            else:
                print(f"CC area too big: {cfg.CHARACTER_AREA_MAX} < {relative_area}")

        return False

    w, h = bounding_box[2:4]

    aspect_ration = h / w

    if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
        print(f"CC aspect ratio: {aspect_ration}")

    if not cfg.CHARACTER_ASPECT_RATIO_MIN<=aspect_ration<=cfg.CHARACTER_ASPECT_RATIO_MAX:
        if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
            if cfg.CHARACTER_ASPECT_RATIO_MIN > aspect_ration:
                print(f"CC aspect ratio too small: {cfg.CHARACTER_ASPECT_RATIO_MIN} > {aspect_ration}")
            else:
                print(f"CC aspect ratio too big: {cfg.CHARACTER_ASPECT_RATIO_MAX} < {aspect_ration}")

        return False

    w_ratio = w / image_w
    h_ratio = h / image_h

    if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
        print(f"CC width: {w} CC height: {h}")
        print(f"CC relative width: {w_ratio} CC relative height: {h_ratio}")
    
    if not cfg.CHARACTER_WIDTH_MIN<=w_ratio<=cfg.CHARACTER_WIDTH_MAX:

        if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
            if cfg.CHARACTER_WIDTH_MIN > w_ratio:
                print(f"CC is too narrow: {cfg.CHARACTER_WIDTH_MIN} > {w_ratio}")
            else:
                print(f"CC is too wide: {cfg.CHARACTER_WIDTH_MAX} < {w_ratio}")

        return False
    
    if not cfg.CHARACTER_HEIGHT_MIN<=h_ratio<=cfg.CHARACTER_HEIGHT_MAX:

        if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
            if cfg.CHARACTER_HEIGHT_MIN > h_ratio:
                print(f"CC is too low: {cfg.CHARACTER_HEIGHT_MIN} > {h_ratio}")
            else:
                print(f"CC is too tall: {cfg.CHARACTER_HEIGHT_MAX} < {h_ratio}")

        return False
    
    if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
        print("Character")

    return True

def radon_vertical(image, theta=None, preserve_range=False):
    image = convert_to_float(image, preserve_range)

    rows, cols = image.shape[:2]
    
    max_angle = cfg.MAX_VERTICAL_TILT
    padding = int(cols * math.tan(math.radians(max_angle)))
    padded_image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)

    radon_image = np.zeros((cols + padding * 2, max_angle * 2 + 1), dtype=image.dtype)

    theta = np.arange(-max_angle, max_angle + 1)
    for i, angle in enumerate(np.deg2rad(theta)):

        matrix = np.array(
            [
                [1, math.tan(angle), 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        
        warped = transform.warp(padded_image, matrix)

        radon_image[:, i] = warped.sum(0)

    return radon_image

def sobel_edge_detector(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return magnitude

def rotate_horizontaly(image, angle, cropping= True):
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    center_x, center_y = image.shape[0] / 2, image.shape[1] / 2
    rotation_matrix = np.array(
            [
                [cos_a, sin_a, -center_x * (cos_a + sin_a - 1)],
                [-sin_a, cos_a, -center_y * (cos_a - sin_a - 1)],
                [0, 0, 1],
            ])
    
    rotated = (transform.warp(image, rotation_matrix)* 255).astype(np.uint8)

    if cropping is True:
        h, w = rotated.shape[:2]
        crop = int((w * math.sin(abs(angle)) * math.cos(abs(angle))) / 2) 
        rotated = rotated[crop: h-crop, :]

    return rotated

def rotate_verticaly(image, angle):
    tan_a = math.tan(angle)

    center_x = image.shape[0] / 2
    shear_matrix = np.array(
            [
                [1, tan_a, -center_x * tan_a],
                [0, 1, 0],
                [0, 0, 1],
            ])
    
    rotated = ((transform.warp(image, shear_matrix))* 255).astype(np.uint8)
    return rotated

def RSM(sinogram):
    rsm = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    return np.argmax(rsm)

def tilt_correction(image):
    magnitude = sobel_edge_detector(imutils.resize(image, width= min(200, image.shape[1])))

    sinogram_horizontal = radon(magnitude, theta=np.arange(90 - cfg.MAX_HORIZONTAL_TILT, 90 + cfg.MAX_HORIZONTAL_TILT))

    rotation_horizontal = cfg.MAX_HORIZONTAL_TILT - RSM(sinogram_horizontal)
    horizontal_rad = -math.radians(rotation_horizontal)

    rotated_horizontaly_magnitude = rotate_horizontaly(magnitude, horizontal_rad)

    sinogram_vertical = radon_vertical(rotated_horizontaly_magnitude)

    rotation_vertical = cfg.MAX_VERTICAL_TILT - RSM(sinogram_vertical)
    vertical_rad = -math.radians(rotation_vertical)
    
    rotated_horizontaly_input = rotate_horizontaly(image, horizontal_rad)
    sheared_verticaly_input = rotate_verticaly(rotated_horizontaly_input, vertical_rad)

    if cfg.DEBUG_PLATE_ROTATION is True:
        print(f"Horizontal tilt {rotation_horizontal}, Vertical tilt {rotation_vertical}")

        sheared_verticaly_magnitude = rotate_verticaly(rotated_horizontaly_magnitude, vertical_rad)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 3, 1)
        plt.imshow(magnitude, cmap='gray')
        plt.title('Edges')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(sinogram_horizontal, cmap='gray')
        plt.title('Sinogram horizontal')
        plt.xlabel(r'Projection Angle ($\alpha$, degrees)')
        plt.ylabel('x')
        plt.axis('on')

        plt.subplot(2, 3, 3)
        plt.imshow(rotated_horizontaly_magnitude, cmap='gray')
        plt.title('Rotated horizontal')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(sinogram_vertical, cmap='gray')
        plt.title('Sinogram vertical')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(sheared_verticaly_magnitude, cmap='gray')
        plt.title('Rotated vertical')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(sheared_verticaly_input, cmap='gray')
        plt.title('Output')
        plt.axis('off')

        plt.show()

    return sheared_verticaly_input

def get_characters(image):
    cv2.imwrite("Original_plate_image.png", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    width = gray.shape[1]
    blur = cv2.bilateralFilter(gray , int(width*0.02), int(width*0.01), int(width*0.2))

    rotated = tilt_correction(blur)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(rotated)

    binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    binary = (binary* 255).astype(np.uint8)
 
    label_count, labels, bounding_boxes, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    character_labels = []

    for i in range(1, label_count):
        if is_character(labels, bounding_boxes[i]):
            character_labels.append(i)
        
        if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
            label_image = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            character_image = ((labels == i).astype(np.uint8) * 255)
            label_image[character_image == 255] = color
            cv2.imshow("CCA", label_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    character_labels = sorted(character_labels, key= lambda i: bounding_boxes[i][0])

    character_images = []
    for i in character_labels:
        bounding_box = bounding_boxes[i]
        x, y, w, h = bounding_box[:4]
            
        character_image = ((labels == i).astype(np.uint8) * 255)[y : y+h, x : x+w]
        character_images.append(character_image)   

    if cfg.DEBUG_CHARACTER_SEGMENTATION is True:
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(3, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Gray')
        plt.axis('off')

        plt.subplot(3, 3, 3)
        plt.imshow(blur, cmap='gray')
        plt.title('Blur')
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.imshow(rotated, cmap='gray')
        plt.title('Tilt corrected')
        plt.axis('off')

        plt.subplot(3, 3, 5)
        plt.imshow(equalized, cmap='gray')
        plt.title('Histogram equalized')
        plt.axis('off')

        plt.subplot(3, 3, 6)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary')
        plt.axis('off')

        colored_components = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        for i in range(1, label_count):
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            component_mask = (labels == i).astype(np.uint8) * 255
            colored_components[component_mask == 255] = color

        plt.subplot(3, 3, 7)
        plt.imshow(colored_components, cmap='gray')
        plt.title('Connected components')
        plt.axis('off')

        characters_colored = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        for i in character_labels:
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            component_mask = (labels == i).astype(np.uint8) * 255
            characters_colored[component_mask == 255] = color

        plt.subplot(3, 3, 8)
        plt.imshow(characters_colored, cmap='gray')
        plt.title('Detected characters')
        plt.axis('off')

        plt.show()

    return character_images

    
     