import os
import csv
from shutil import move

IDX_DICT = {'cbb': 465,
            'cbsd': 1442,
            'cgm': 772,
            'cmd': 2657,
            'healthy': 315}

LBL_DESC_DICT = {0: 'cbb',
                 1: 'cbsd',
                 2: 'cgm',
                 3: 'cmd',
                 4: 'healthy'}

LABELS = '/home/noteme/PycharmProjects/comp/data/cassava-leaf-disease-classification/train.csv'
NEW_IMAGES_FOLDER = '/home/noteme/PycharmProjects/comp/data/cassava-leaf-disease-classification/train_images'
DEST_FOLDER = '/home/noteme/PycharmProjects/comp/data/cassava-disease/train'

# read train.csv file and load it as dict
with open(LABELS, 'r') as fp:
    reader = csv.reader(fp)
    next(reader, None)  # skip header
    labels_dict = {name: LBL_DESC_DICT[int(label)] for name, label in reader}

# rename and add (move) new images to the older images using previous format (class-wise folders)
image_names = os.listdir(NEW_IMAGES_FOLDER)
for image_name in image_names:

    label = labels_dict[image_name]
    # update IDX_DICT idx
    IDX_DICT[label] += 1
    old_abs_path = os.path.join(NEW_IMAGES_FOLDER, image_name)
    new_path = os.path.join(DEST_FOLDER, label, f"train-{label}-{IDX_DICT[label]}.jpg")
    move(old_abs_path, new_path)
    pass