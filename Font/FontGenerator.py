import os
from PIL import ImageFont, Image, ImageDraw
import cv2
import numpy as np

font_path = r"C:\Users\dimon\AKPR\Font\arklatrs-webfont.ttf"
output_folder = r"C:\Users\dimon\AKPR\Font\characters"

font_size = 200
character_list = "0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ"

crop = True

font = ImageFont.truetype(font_path, font_size)

for character in character_list:
    left, bottom, right, top = font.getbbox(character)
    width = right - left
    height = top - bottom

    if width <= 0 or height <= 0:
        continue

    print(f"Character: {character}")

    image = Image.new('L', (width, height), color=255)

    drawing = ImageDraw.Draw(image)
    drawing.text((0, 0),
            character,
            fill=(0),
            font=font)
    
    output_path = os.path.join(output_folder, f"{character}.png")

    if crop is True:
        img = np.array(image)

        label_count, labels, bounding_boxes, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

        x, y, w, h = bounding_boxes[0][:4]
        img = img[y:y+h, x:x+w]

        image = Image.fromarray(img)


    image.save(output_path)

character_set = "0123456789\nABCDEFGHIJ\nKLMNOPRSTU\nVWXYZ"
font = ImageFont.truetype(font_path, 150)

image = Image.new('L', (1500, 1500), color=255)
drawing = ImageDraw.Draw(image)
drawing.text((0, 0),
        character_set,
        fill=(0),
        font=font,
        align="center",)

bounding_boxes = cv2.connectedComponentsWithStats(cv2.bitwise_not(np.array(image)), connectivity=8)[2][1:]

min_x, min_y, max_x, max_y = 1500, 1500, 0, 0

for bounding_box in bounding_boxes:
    x, y, w, h = bounding_box[:4]

    min_x = min(min_x, x)
    min_y = min(min_y, y)
    max_x = max(max_x, x + w)
    max_y = max(max_y, y + h)


image = image.crop((min_x, min_y, max_x, max_y))

image_char_set = Image.new("L", ((max_x - min_x) + 2, (max_y - min_y) + 2), 255)
image_char_set.paste(image, (1, 1))

image_char_set.save(os.path.join(output_folder, "CharacterSet.png"))

