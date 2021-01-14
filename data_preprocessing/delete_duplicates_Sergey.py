import pandas as pd
import numpy as np
import os

import datetime
import torch
import imagehash
import glob
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
    return np.array([f(image).hash for f in (FUNCS)]).reshape(256)


def compute_hashes(pathes):
    hashes = [get_hashes(path) for path in tqdm(pathes)]
    hashes_all = np.array(hashes)
    hashes_all = torch.Tensor(hashes_all.astype(int))

    return hashes_all


def compute_similarity(hashes,image_pathes):
    dups = []
    dups_p = []
    # sims = np.array([(hashes[i] == hashes).sum(dim=1).numpy() / 256 for i in tqdm(range(hashes.shape[0]))])

    for i in tqdm(range(hashes.shape[0])):
        sim = (hashes[i] == hashes).sum(dim=1).numpy() / 256
        sim[i] = 0
        idxx = i
        duplicates, duplicate_pairs = find_duplicates([sim], image_pathes, idxx, external_folders=True)
        dups.extend(duplicates)
        dups_p.extend(duplicate_pairs)
        print()
    # np.fill_diagonal(sims, 0)  # remove comparison of images with itself
    return dups, dups_p


def find_duplicates(similarity, image_pathes, idxx, external_folders=False):
    duplicates_idx = []
    duplicate_pairs = []
    # filter all duplicates
    for row_id in (range(len(similarity))):
        if not external_folders:
            if row_id in duplicates_idx:
                continue
        dup_indices = np.argwhere(similarity[row_id] > 0.9)[:, 0]
        if any(np.argwhere(similarity[row_id] > 0.9)):
            duplicate_pairs.append((os.path.basename(image_pathes[idxx]),[os.path.basename(image_pathes[idx]) for idx in dup_indices]))
        for idx in dup_indices:
            duplicates_idx.append(idx)

    # remove duplicates
    duplicates_idx = np.unique(duplicates_idx)
    duplicates = [os.path.basename(image_pathes[idx]) for idx in duplicates_idx]
    return duplicates,duplicate_pairs


def find_pathes(folder):
    image_pathes = []
    pathes = glob.glob(f"{folder}/*")
    for path in pathes:
        image_pathes.append(path)
    return image_pathes


def remove_extra_class_duplicates(folder):
    # find intra-class image pathes
    image_pathes = find_pathes(folder)

    # compute hashes
    hashes = compute_hashes(image_pathes)

    # compute similiarities
    sims = compute_similarity(hashes)

    # find duplicates
    duplicates, duplicate_pairs = find_duplicates(similarity=sims, image_pathes=image_pathes)

    dups_to_remove = pd.DataFrame(duplicates,columns=['duplicates'])
    dups_pairs = pd.DataFrame(duplicate_pairs, columns=['duplicates', 'duplicate_pairs'])

    return dups_to_remove, dups_pairs

def img_pathes(base_names, root1, root2):
    ''' (list,str,str) >>> list
        returns abs_pathes of img in 2020 and 2019 folders
    '''
    img_list = []
    for file in base_names:
        if file[0].isdigit():
            img_list.append(os.path.join(root1, file))
        else:
            img_list.append(os.path.join(root2, file.split('-')[1], file))

    return img_list

def img_pathes2(base_names, root1, root2,root3):
    ''' (list,str,str) >>> list
        returns abs_pathes of img in 2020 and 2019 folders
    '''
    img_list = []
    for file in base_names:
        if file[0].isdigit():
            img_list.append(os.path.join(root1, file))
        elif 'extra' in file:
            img_list.append(os.path.join(root3, file))
        else:
            img_list.append(os.path.join(root2, file.split('-')[1], file))

    return img_list


def remove_intra_class_duplicates(image_pathes):

    # compute hashes
    hashes = compute_hashes(image_pathes)

    # compute similiarities
    sims = compute_similarity(hashes)

    # find duplicates
    duplicates, duplicate_pairs = find_duplicates(similarity=sims, image_pathes=image_pathes)

    dups_to_remove = pd.DataFrame(duplicates,columns=['duplicates'])
    dups_pairs = pd.DataFrame(duplicate_pairs, columns=['duplicates', 'duplicate_pairs'])

    return dups_to_remove, dups_pairs

def remove_cross_class_duplicates(image_pathes):

    # compute hashes
    hashes = compute_hashes(image_pathes)
    # hashes = np.genfromtxt(r'D:\CASSAVA COMPET\hashes_all.csv',delimiter=',')
    # hashes = np.array(hashes)
    # hashes = torch.Tensor(hashes.astype(int))

    # compute similiarities
    duplicates,duplicate_pairs = compute_similarity(hashes,image_pathes)

    # find duplicates
    # duplicates,duplicate_pairs = find_duplicates(similarity=sims, image_pathes=image_pathes, external_folders=True)

    dups_to_remove = pd.DataFrame(duplicates, columns=['duplicates'])
    dups_pairs = pd.DataFrame(duplicate_pairs, columns=['duplicates', 'duplicate_pairs'])

    return dups_to_remove, dups_pairs

    # remove duplicates
    # p_map(os.remove, duplicates)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    PATH_2019 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2019 COMP\train'
    PATH_2020 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2020-11-27\train_images'
    csv_2020 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2020-11-27\train.csv'
    extra_images_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\extraimages\extraimages'


    # remove duplicates within extra images
    # extra_dups, extra_dup_pairs = remove_extra_class_duplicates(extra_images_folder)
    # extra_dups.to_csv('extra_dups.csv',index=False)
    # extra_dup_pairs.to_csv('extra_dup_pairs.csv',index=False)

    # remove duplicates within each class
    # for class in range(5)
        # df.loc[df['label'] == class]['image id']

    # intra_dups, intra_dup_pairs =
    merged_dataset = pd.read_csv('merged_dataset.csv')
    # a = pd.DataFrame([],columns=['duplicates'])
    # b = pd.DataFrame([],columns=['duplicates','duplicate_pairs'])
    # for folder in range(5):
    #     folder = merged_dataset[merged_dataset['label'] == folder]['image_id'].values.tolist()
    #     pathes = img_pathes(folder,PATH_2020,PATH_2019)
    #     intra_dups, intra_dup_pairs = remove_intra_class_duplicates(pathes)  # removed 1256 images
    #     a = pd.concat([a,intra_dups],ignore_index=True)
    #     b = pd.concat([b,intra_dup_pairs],ignore_index=True)
    # a.to_csv('intra_dups.csv', index=False)
    # b.to_csv('intra_dup_pairs.csv', index=False)

    # remove duplicates between classes
    a = pd.read_csv('intra_dups.csv')
    alpha = merged_dataset['image_id'].to_list()
    beta = a['duplicates'].to_list()
    no_intra_dup_list = set(alpha).difference(set(beta))

    # cross_dups, cross_dup_pairs = remove_cross_class_duplicates(img_pathes(no_intra_dup_list,PATH_2020,PATH_2019))  # removed 174 images
    # cross_dups.to_csv('cross_dups.csv',index=False)
    # cross_dup_pairs.to_csv('cross_dup_pairs.csv',index=False)


    a = pd.read_csv('cross_dups.csv')
    beta = a['duplicates'].to_list()
    no_cross_dup_list = set(no_intra_dup_list).difference(set(beta))
    extra_dups = pd.read_csv('extra_dups.csv')
    extra_im_list = set(os.listdir(extra_images_folder)).difference(set(extra_dups['duplicates'].to_list()))
    no_cross_dup_list_union_extra = sorted([*no_cross_dup_list.union(extra_im_list)])
    cross_dups_extra, cross_dup_pairs_extra = remove_cross_class_duplicates(img_pathes2(no_cross_dup_list_union_extra, PATH_2020, PATH_2019,extra_images_folder))
    cross_dups_extra.to_csv('cross_dups_extra.csv', index=False)
    cross_dup_pairs_extra.to_csv('cross_dup_pairs_extra.csv', index=False)

    print(f"all intra-class duplicates removed in " \
          f"{round((datetime.datetime.now() - start_time).total_seconds() / 60, 1)} mins")