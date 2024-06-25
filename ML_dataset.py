# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 07:24:29 2024

@author: Bala Eesan
"""

from PIL import ImageFont, ImageDraw, Image
from faker import Faker
from random import randint
import keras_ocr
import cv2
import math
import numpy as np
import os
import random


pipeline = keras_ocr.pipeline.Pipeline()

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

# Creating list for text removal:
remove_list = ["MD", "ABDUL", "AHAD", "AZIZ", "03", "JUL", "1989", "AT", "04", "OCT", "2015", "2025", "DKOSOSIZICO"]
fake = Faker()

# Selecting random male/female portrait
def select_portrait(dataset_path):
  face_images = os.listdir(dataset_path)
  random_face_image = random.choice(face_images)
  portrait_path = os.path.join(dataset_path, random_face_image)
  portrait = cv2.imread(portrait_path)
  return portrait

# Removing text to be replaced with keras_ocr and cv2
def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")

    for box in prediction_groups[0]:
      if box[0].upper() in remove_list:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255,
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('filtered_image.jpg',img_rgb)

    return img_rgb

img_path = '/content/wMOp41qN70.jpg'
img_rgb = inpaint_text(img_path, pipeline)

# Replacing text with Faker
def fake_text(img_rgb):
  pil_im = Image.fromarray(img_rgb)
  draw = ImageDraw.Draw(pil_im)

  font = ImageFont.truetype("Roboto-Black.ttf", 28)
  fake_list = fake.profile()
  fake_lname = fake.last_name().upper()
  fake_date = str(fake.past_date('-45y'))
  fake_date2 = str(fake.future_date())
  fake_date3 = str(fake.future_date('+20y'))

  options = ['male', 'female']
  choices = random.choice(options)
  if choices == 'male':
    draw.text((350, 155), fake.first_name_male().upper()+' '+fake_lname, font=font, fill = (0,0,0))
    dataset_path = '/content/Male_Portraits'
  else:
    draw.text((350, 155), fake.first_name_female().upper()+' '+fake_lname, font=font, fill = (0,0,0))
    dataset_path = '/content/Female_Portraits'

  draw.text((350, 225), fake_date, font=font, fill = (0,0,0))
  draw.text((350, 300), fake_list['blood_group'], font=font, fill = (0,0,0))
  draw.text((350, 370), fake.first_name_male().upper()+' '+fake_lname, font=font, fill = (0,0,0))
  draw.text((350, 430), fake_date2, font=font, fill = (0,0,0))
  draw.text((625, 430), fake_date3, font=font, fill = (0,0,0))
  draw.text((350, 505), fake.bothify('DK#######C#####'), font=font, fill = (0,0,0))

  cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
  img_rgb = cv2.cvtColor(cv2_im_processed, cv2.COLOR_BGR2RGB)
  cv2.imwrite('appended_image.jpg',img_rgb)

  return dataset_path

dataset_path = fake_text(img_rgb)

# Replacing image using overlay
def place_portrait(dl_license, portrait, position):
    # Adjust size of the portrait to be placed
    portrait_resized = cv2.resize(portrait, (275, 350))

    # Set the position in which portrait should be centered
    x, y = position

    # Overlay the portrait on the image
    dl_license[y:y + portrait_resized.shape[0], x:x + portrait_resized.shape[1]] = portrait_resized

    return dl_license

# Putting all codes together and adding different background images
for i in range(10): # Replace 10 with any number to generate those many ID cards
  img_path = '/content/wMOp41qN70.jpg'
  img_rgb = inpaint_text(img_path, pipeline)
  dataset_path = fake_text(img_rgb)

  background_path = '/content/Backgrounds'
  backgroud = os.listdir(background_path)
  random_background = random.choice(backgroud)

  dl_license = cv2.imread("appended_image.jpg")
  portrait = select_portrait(dataset_path)
  position = (25,125)

  output = place_portrait(dl_license, portrait, position)

  cv2_im_processed = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
  img_rgb = cv2.cvtColor(cv2_im_processed, cv2.COLOR_BGR2RGB)
  cv2.imwrite('overlay_image.jpg',img_rgb)

  img1 = Image.open("/content/Backgrounds/" + random_background)
  img2 = Image.open("overlay_image.jpg")

  newsize = (950, 600)
  img1 = img1.resize(newsize)

  newsize = (600, 450)
  img2 = img2.resize(newsize)

  x = randint(0,100)
  y = randint(0,100)

  img1.paste(img2, (x,y))

  file_name = 'overlay_' + str(i+1) +'.png'

  img1.save(file_name)