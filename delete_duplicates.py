import datetime
start_time = datetime.datetime.now()

import pandas as pd
import numpy as np
import os
import torch
import imagehash
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from p_tqdm import p_map
from PIL import Image
from shutil import copyfile

TRAIN_DIR = '/home/noteme/PycharmProjects/comp/data/cassava-disease/train'
TRAIN_CSV_PATH = '/home/noteme/PycharmProjects/comp/data/cassava-disease/train.csv'

label_dic = {'cbb': 0,
             'cbsd': 1,
             'cgm': 2,
             'cmd': 3,
             'healty': 4}

labels = '/home/noteme/PycharmProjects/comp/data/cassava-disease/train.csv'

train = pd.read_csv(TRAIN_CSV_PATH)

funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]

image_pathes = []
hashes = []

folders = glob.glob(f"{TRAIN_DIR}/*")
for folder in folders:
    pathes = glob.glob(f"{folder}/*")
    for path in pathes:
        image_pathes.append(path)


def get_hashes(path):
    image = Image.open(path)
    return np.array([f(image).hash for f in funcs]).reshape(256)

hashes.extend(p_map(get_hashes, image_pathes))

hashes_all = np.array(hashes)
hashes_all = torch.Tensor(hashes_all.astype(int))
sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).numpy()/256 for i in range(hashes_all.shape[0])])
indices1 = np.where(sims > 0.9)
indices2 = np.where(indices1[0] != indices1[1])
image_ids1 = [image_pathes[i] for i in indices1[0][indices2]]
image_ids2 = [image_pathes[i] for i in indices1[1][indices2]]
dups = {tuple(sorted([image_id1,image_id2])):True for image_id1, image_id2 in zip(image_ids1, image_ids2)}
print('found %d duplicates' % len(dups))
duplicate_image_ids = sorted(list(dups))
print(duplicate_image_ids)
duplicate_image_ids = sorted(list(dups))

dup_folder = '/home/noteme/PycharmProjects/comp/data/cassava-disease/duplicates'
for tup in duplicate_image_ids:
    x, y = tup
    new_x = os.path.join(dup_folder, os.path.basename(x))
    new_y = os.path.join(dup_folder, os.path.basename(y))
    copyfile(x, new_x)
    copyfile(y, new_y)
