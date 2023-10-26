# Author: Ou Wenxuan
# Email: ouwenxuan@senseauto.com
# This script apply semantic mask to the alpha channl of the original image

import os
import glob

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# semantic labels
LABELS = {
    "unlabeled": 255,
    "ego_vehicle": 255,
    "rectification_border": 255,
    "out_of_roi": 255,
    "static": 255,
    "dynamic": 255,
    "ground": 255,
    "road": 0,
    "sidewalk": 1,
    "parking": 255,
    "rail track": 255,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "guard rail": 255,
    "bridge": 255,
    "tunnel": 255,
    "pole": 5,
    "pole_group": 255,
    "traffic_light": 6,
    "traffic_sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "caravan": 255,
    "trailer": 255,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
    "license_plate": -1,
}


dataset_dir = "input_data/20230404-112436_7"
output_dir = os.path.join(dataset_dir, "masked_images")

# object to mask out
masked_obj = "sky"

# create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_filenames = os.path.join(dataset_dir, "images/*.jpg")
mask_filenames = os.path.join(dataset_dir, "masks/*.npz")
image_filenames = glob.glob(image_filenames)
mask_filenames = glob.glob(mask_filenames)

image_filenames.sort()
mask_filenames.sort()

assert len(image_filenames) == len(mask_filenames), "Image number and mask number are not matched!!!"


for i in range(len(image_filenames)):
    print("progress: ", i, " of ", len(image_filenames))
    image = np.asarray(Image.open(image_filenames[i]))
    mask = np.load(mask_filenames[i])
    mask = mask["arr_0"]

    mask = mask != LABELS[masked_obj]  # mask out selected object
    # mask = mask * 255     # TODO: for debug only

    mask = mask[:, :, np.newaxis]

    masked_image = np.concatenate((image, mask), axis=2)
    masked_image = masked_image.astype(np.uint8)
    # print("output shape: ", masked_image.shape, " output dtype: ", masked_image.dtype)

    im = Image.fromarray(masked_image)
    output_filename = os.path.basename(image_filenames[i]).split(".")[0]
    output_filename = os.path.join(output_dir, output_filename+".png") # save as png
    im.save(output_filename)