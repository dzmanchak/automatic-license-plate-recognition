import os
import cv2

import Config as cfg

def load_templates(character_list, template_path):
    templates = {}

    for char in character_list:
        for file in os.listdir(template_path):
            filename, extension = os.path.splitext(file)

            if filename == char and extension in (".png", ".jpg"):
                template = cv2.imread(os.path.join(template_path, file), cv2.IMREAD_GRAYSCALE)
                template = cv2.resize(template, (cfg.OCR_CHARACTER_WIDTH, cfg.OCR_CHARACTER_HEIGHT))
                templates[char] = template

    return templates

def box_label(image, box, label='', color=(0, 255, 0), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)