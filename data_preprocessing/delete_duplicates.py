import datetime
import numpy as np
import os
import torch
import imagehash
import glob
from p_tqdm import p_map
from PIL import Image
from tqdm import tqdm

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
    hashes = []
    hashes.extend(p_map(get_hashes, pathes))
    hashes_all = np.array(hashes)
    hashes_all = torch.Tensor(hashes_all.astype(int))
    return hashes_all


def compute_similarity(hashes):
    sims = np.array([(hashes[i] == hashes).sum(dim=1).numpy() / 256 for i in tqdm(range(hashes.shape[0]))])
    np.fill_diagonal(sims, 0)  # remove comparison of images with itself
    return sims


def find_duplicates(similarity, image_pathes, external_folders=False):
    duplicates_idx = []

    # filter all duplicates
    for row_id in tqdm(range(len(similarity))):
        if not external_folders:
            if row_id in duplicates_idx:
                continue
        dup_indices = np.argwhere(similarity[row_id] > 0.9)[:,0]
        for idx in dup_indices:
            duplicates_idx.append(idx)

    # remove duplicates
    duplicates_idx = np.unique(duplicates_idx)
    duplicates = [image_pathes[idx] for idx in duplicates_idx]
    return duplicates


def find_pathes(folder):
    image_pathes = []
    pathes = glob.glob(f"{folder}/*")
    for path in pathes:
        image_pathes.append(path)
    return image_pathes


def remove_intra_class_duplicates(folder):

    # find intra-class image pathes
    image_pathes = find_pathes(folder)

    # compute hashes
    hashes = compute_hashes(image_pathes)

    # compute similiarities
    sims = compute_similarity(hashes)

    # find duplicates
    duplicates = find_duplicates(similarity=sims, image_pathes=image_pathes)

    # remove duplicates
    p_map(os.remove, duplicates)


def remove_cross_class_duplicates(folders):
    image_pathes = []
    for folder in folders:
        image_pathes.extend(find_pathes(folder))

    # compute hashes
    hashes = compute_hashes(image_pathes)

    # compute similiarities
    sims = compute_similarity(hashes)

    # find duplicates
    duplicates = find_duplicates(similarity=sims, image_pathes=image_pathes, external_folders=True)

    # remove duplicates
    p_map(os.remove, duplicates)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    TRAIN_DIR = '/home/noteme/PycharmProjects/comp/data/cassava-disease/train'
    extra_images_folder = '/home/noteme/PycharmProjects/comp/data/cassava-disease/extraimages/extraimages'
    folders = glob.glob(f"{TRAIN_DIR}/*")

    # # remove duplicates within extra images
    # remove_intra_class_duplicates(extra_images_folder)
    #
    # # remove duplicates within each class
    # for folder in folders:
    #     remove_intra_class_duplicates(folder)   # removed 1256 images

    # remove duplicates between classes
    remove_cross_class_duplicates(folders)      # removed 174 images

    print(f"all intra-class duplicates removed in "\
          f"{round((datetime.datetime.now() - start_time).total_seconds()/60, 1)} mins")
