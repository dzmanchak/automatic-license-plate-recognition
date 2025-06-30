import cv2
import os
import pytesseract

import numpy as np
import pandas as pd
import Config as cfg

from matplotlib import pyplot as plt

def recognize_template_matching(image, templates):
    image = cv2.bitwise_not(image)

    best_char = ""
    best_conf = -100

    for char, template in templates.items():
        conf = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED))[1]

        if cfg.DEBUG_PLATE_CHARACTER_RECOGNITION is True:
            print("Char: " + str(char) + " Conf: " + str(conf))
        if conf > best_conf:
            
            best_char = char
            best_conf = conf

    if cfg.DEBUG_PLATE_CHARACTER_RECOGNITION is True:
        print("Best char confidence: " + str(best_conf))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Char image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(templates[best_char], cmap='gray')
        plt.title('Best char matches')
        plt.axis('off')

        plt.show()

    return (best_char, best_conf)

def recognize_tesseract(image):
    image = cv2.bitwise_not(image)
    image = np.pad(image, 4, mode='constant', constant_values=255)

    data = pytesseract.image_to_data(image, config=f'-c tessedit_char_whitelist={cfg.CHARACTERS_LIST} --oem 0 --psm 10', lang='eng', output_type='data.frame')

    data = data.iloc[-1]
    character = str(data.iloc[-1])[0]
    confidence = data.iloc[-2]
    
    if cfg.DEBUG_PLATE_CHARACTER_RECOGNITION is True:
        print(f"Char: {character} Confidence: {confidence}")
        cv2.imshow("Character", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return character, confidence / 100  

def syntax_analisys_poland(plate_number):
    plate_number = list(plate_number)
    city_index_best_matches = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "4": "A",
        "3": "E",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "P",
    }
    
    car_index_best_matches = {
        "D": "0", 
        "O": "0", 
        "I": "1",
        "Z": "2",
        "B": "8"
    }

    for i, char in enumerate(plate_number):
        if 1 >= i >= 0 and char in city_index_best_matches:
            plate_number[i] = city_index_best_matches[char]
        elif i >= 3 and char in car_index_best_matches:
            plate_number[i] = car_index_best_matches[char]

    plate_number = "".join(plate_number)
    return plate_number