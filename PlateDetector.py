import os
import cv2
import json
import time 
import imutils
import pytesseract

import Config as cfg

from matplotlib import pyplot as plt
from ultralytics import YOLO

from CharacterSegmentor import get_characters 
from CharacterClassification import recognize_template_matching, syntax_analisys_poland, recognize_tesseract
from Utils import load_templates, box_label

characters_templates = {}

if cfg.USE_TESSERACT is True:
    pytesseract.pytesseract.tesseract_cmd = cfg.TESSRACT_PATH
else:
    characters_templates = load_templates(cfg.CHARACTERS_LIST, cfg.CHARACTER_TEMPLATES_PATH)

model = YOLO(cfg.MODEL_WEIGHTS_PATH)

all_files = os.listdir(cfg.INPUT_PATH)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

license_plates_detected = 0
license_plates_segmented = 0
license_plates_recognized = 0

characters_segmented = 0

detection_time_avarage = -100
segmetation_time_avarage = -100
recognition_time_avarage = -100

for image in image_files:
    image_path = os.path.join(cfg.INPUT_PATH, image)
    orig_image = cv2.imread(image_path)

    detection_start_time = time.time()
    result = model(orig_image, conf=cfg.PLATE_DETECTION_CONFIDENCE, save=False, verbose=True)[0]
    detection_end_time = time.time()
    detection_time = (detection_end_time - detection_start_time) * 1000


    if detection_time_avarage == -100:
            detection_time_avarage = detection_time
    else:
            detection_time_avarage = (detection_time_avarage + detection_time) / 2

    result_image = result.orig_img.copy()

    if cfg.DEBUG_PLATE_DETECTION is True:
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
        plt.title('Detected plates')
        plt.axis('off')
 
        plt.show()

    boxes = sorted(result.boxes, key= lambda roi: roi.xyxy[0].cpu().numpy().astype(int)[0])

    plates = []

    for roi in boxes:
        x1, y1, x2, y2 = roi.xyxy[0].cpu().numpy().astype(int)

        if cfg.MIN_PLATE_WIDTH > (x2 - x1) or cfg.MIN_PLATE_HEIGHT > (y2 - y1):
            continue

        license_plates_detected += 1

        roi_image = orig_image[y1:y2, x1:x2]

        segmentation_start_time = time.time()

        character_images = get_characters(roi_image)

        characters_segmented += len(character_images)

        if not cfg.MIN_CHAR_COUNT <= len(character_images) <= cfg.MAX_CHAR_COUNT:
            continue

        segmentation_end_time = time.time()
        segmentation_time = (segmentation_end_time - segmentation_start_time) * 1000

        if segmetation_time_avarage == -100:
            segmetation_time_avarage = segmentation_time
        else:
            segmetation_time_avarage = (segmetation_time_avarage + segmentation_time) / 2

        license_plates_segmented += 1

        plate_number = ""
        plate_confidence = -2500

        recognition_start_time = time.time()
        for char_image in character_images:
            char, confidence = "", -100
            
            if cfg.USE_TESSERACT is True:
                char, confidence = recognize_tesseract(imutils.resize(char_image, height=cfg.OCR_CHARACTER_HEIGHT))
            else:
                char, confidence = recognize_template_matching(cv2.resize(char_image, (cfg.OCR_CHARACTER_WIDTH, cfg.OCR_CHARACTER_HEIGHT)), characters_templates)

            plate_number += char

            if plate_confidence == -2500:
                plate_confidence = confidence
            else:
                plate_confidence = (plate_confidence + confidence) / 2

        if plate_confidence < cfg.OCR_CONFIDENCE:
            continue
        
        if cfg.USE_SYNTAX_ANALISYS is True:
            plate_number = syntax_analisys_poland(plate_number)

        recognition_end_time = time.time()
        recognition_time = (recognition_end_time - recognition_start_time) * 1000

        if recognition_time_avarage == -100:
            recognition_time_avarage = recognition_time
        else:
            recognition_time_avarage = (recognition_time_avarage + recognition_time) / 2

        license_plates_recognized += 1

        print(plate_number)
        print(plate_confidence)

        plate_data = {}
        plate_data["detected_number"] = plate_number
        plate_data["confidence"] = float(plate_confidence)
        plate_data["detection_time"] = detection_time
        plate_data["segmentation_time"] = segmentation_time
        plate_data["recognition_time"] = recognition_time
        plates.append(plate_data)

        box_label(result_image, roi.xyxy[0].cpu().numpy().astype(int), plate_number + " " + str(round(100 * plate_confidence)) + "% ")


    file_name = os.path.splitext(os.path.basename(image))[0]
    os.makedirs(cfg.SAVE_DICT_PATH, exist_ok=True)
    cv2.imwrite(cfg.SAVE_DICT_PATH + file_name + ".png", result_image)
    with open(cfg.SAVE_DICT_PATH + file_name + ".json", "w", encoding="utf-8") as file:
        json.dump(plates, file, indent=4)

print(f"\nDetected {license_plates_detected} license plates")
print(f"Segmentated {license_plates_segmented} license plates")
print(f"Recognized {license_plates_recognized} license plates")

print(f"Segmented characters {characters_segmented}")

print(f"Avarage detection time {detection_time_avarage}")
print(f"Avarage segmentation time {segmetation_time_avarage}")
print(f"Avarage recognition time {recognition_time_avarage}")