# coding: utf-8
import os
import sys
import typing

import click
import crayons
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import pathlib

directory = "cs-653/code/data"
folders = [
    "kale",
    "chayote",
    "chilli pepper",
    "brocoli",
    "mint",
    "mushroom",
    "cucumber",
    "beets",
    "potatoes",
    "carrots",
    "beans-small",
    "bellpepper",
    "cabbage",
    "eggplant",
    "beans",
    "artichoke",
    "chives",
    "celery",
    "garlic",
]

image_paths = []
cv_images = []

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Remove any .gif etc and only allow above jpeg, jpg, png and bmp
for folder in folders:
    for entry in tqdm(os.scandir(f"{directory}/{folder}")):
        img_path = pathlib.Path(entry)
        # print(img_path.suffix.lower(), img_path.suffix.lower() in ALLOWED_EXTENSIONS)
        if img_path.suffix.lower() in ALLOWED_EXTENSIONS:
            image_paths.append(entry)

# Resize all images to 100,100 with 3 color channels RBF
for d in tqdm(image_paths):
    outfile = f"min-{d.name}"
    try:
        im = Image.open(d.path)
        im.thumbnail((100, 100), Image.ANTIALIAS)
        pil_image = im.convert("RGB")
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        cv_images.append((open_cv_image, d.name, d.path))
    except IOError as e:
        print(e)

width_max = 100
height_max = 100
print(f"max heigh {height_max} , max width {width_max} {len(cv_images)}")

for img in cv_images:
    h, w, _ = img[0].shape
    width_max = min(width_max, w)
    height_max = min(height_max, h)

print(f"max heigh {height_max} , max width {width_max} {len(cv_images)}")
images_padded = []

progress_bar = tqdm(total=len(cv_images), desc="processing images", leave=True)
i = 0

# now we will create an empty image with 100,100 width/height and paste our image
# in the center of it
# this way all the images have same 100 x 100 size, with white padding and background
for img in cv_images:
    i = i + 1
    # read image
    old_image_height, old_image_width, channels = img[0].shape

    # create new image of desired size and color white for background padding
    new_image_width = 100
    new_image_height = 100
    color = (255, 255, 255)
    result = np.full(
        (new_image_height, new_image_width, channels), color, dtype=np.uint8
    )

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[
        y_center : y_center + old_image_height, x_center : x_center + old_image_width
    ] = img[0]

    # save result
    directory = img[2].split("/")
    directory = pathlib.Path("/".join(directory[:-1]))
    outfile = f"{directory}/padded-min-{img[1]}"
    print(f"saving {outfile}")
    try:
        cv2.imwrite(f"{outfile}", result)
    except IOError as e:
        print(e)
    progress_bar.update(1)
