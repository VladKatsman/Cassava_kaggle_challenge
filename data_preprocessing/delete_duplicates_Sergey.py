''' 1) P_TQDM P_MAP does not work for Sergey unfortunately, had to use TQDM instead '''
''' 2) FUNC compute_hashes has been refactored from p_map to list comprehension with TQDM  '''
''' 3) FUNC compute_similarity has been refactored (hashes.shape[0]) >>> len(hashes)) '''
''' 4) FUNC remove_intra_class_duplicates has been refactored from p_map to list comprehension with TQDM '''

import datetime
import numpy as np
import os
import torch
import imagehash
import glob
from p_tqdm import p_map
from tqdm import tqdm
from PIL import Image


FUNCS = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]


def get_hashes(path):
    image = Image.open(path)
    return np.array([f(image).hash for f in FUNCS]).reshape(256)

def compute_hashes(pathes):
    hashes = [get_hashes(path) for path in tqdm(pathes)]
    hashes_all = np.array(hashes)
    hashes_all = torch.Tensor(hashes_all.astype(int))
    return hashes_all

def compute_similarity(hashes):
    sims = np.array([(hashes[i] == hashes).sum(dim=1).numpy() / 256 for i in range(len(hashes))])
    np.fill_diagonal(sims, 0)  # remove comparison of images with itself
    return sims

def find_duplicates(similarity, image_pathes):
    duplicates_idx = []
    # find indices of duplicates, not touching the first element
    for row_idx in range(len(similarity)):
        if row_idx in duplicates_idx:       # check if the image_idx is in the dup_list. Skip the row if it is True
            continue
        dup_num = np.argwhere(similarity[row_idx] > 0.9)[:,0]   # return a scalar of the dup_num only associated with the image.
        duplicates_idx.extend(dup_num)

    duplicates_pathes = [image_pathes[idx] for idx in set(duplicates_idx)] # set() added in order to have unique_val in dup_idx list
    return duplicates_pathes

org_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2020-11-29 TRAIN ORIGINAL\cbsd'
pathes = glob.glob(f"{org_folder}/*")
image = Image.open(pathes[498])
image.show()

# hashes = compute_hashes(pathes)
# sims = compute_similarity(hashes)
# print(np.argwhere(sims[498] > 0.7)[:,0])

# def find_pathes(folder):
#     image_pathes = []
#     pathes = glob.glob(f"{folder}/*")
#     for path in pathes:
#         image_pathes.append(path)
#     return image_pathes
#
#
# def remove_intra_class_duplicates(folder):
#
#     # find intra-class image pathes
#     image_pathes = find_pathes(folder)
#
#     # compute hashes
#     hashes = compute_hashes(image_pathes)
#
#     # compute similiarities
#     sims = compute_similarity(hashes)
#
#     # find duplicates
#     duplicates = find_duplicates(similarity=sims, image_pathes=image_pathes)
#
#     # remove duplicates
#     [os.remove(duplicate) for duplicate in tqdm(duplicates)]
#     # p_map(os.remove, duplicates)
#
# if __name__ == '__main__':
#     start_time = datetime.datetime.now()
#     TRAIN_DIR = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2020-11-29 TRAIN ORIGINAL\cbsd'
#     folders = glob.glob(f"{TRAIN_DIR}/*")
#
#     # remove duplicates within each class
#     for folder in folders:
#         remove_intra_class_duplicates(folder)       # 19 removed
#
#     print(f"all intra-class duplicates removed in "\
#           f"{round((datetime.datetime.now() - start_time).total_seconds()/60, 1)} mins")
