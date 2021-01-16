import pandas as pd
import numpy as np
import os

import datetime
import torch
import imagehash
import glob
from PIL import Image
from tqdm import tqdm
from data_preprocessing import helper_function


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


def compute_similarity(hashes,image_pathes,cross_find=False):

    dups = []
    dups_p = []

    for i in tqdm(range(hashes.shape[0])):
        if not cross_find:                                              # skip paired duplicates for the same class
            if os.path.basename(image_pathes[i]) in dups:
                continue
        sim = (hashes[i] == hashes).sum(dim=1).numpy() / 256                # compute similarity mapping in %
        sim[i] = 0                                                          # fill i-th diagonal element with 0
        duplicates, duplicate_pairs = \
            find_duplicates(sim, image_pathes, i)                           # get duplicates and dup_pairs for i-th row
        dups.extend(duplicates)                                             # obtain list of duplicate
        dups_p.extend(duplicate_pairs)                                      # obtain list of duplicate pairs

    return set(dups), dups_p


def find_duplicates(similarity, image_pathes, row_idx):
    ''' Find duplicates within a single row with corresponding column idx
     (np.array, list, int) >>> list of duplicates, list of duplicate_pairs
     '''

    duplicate_pairs = []

    dup_indices = np.argwhere(similarity > 0.9)[:, 0]                       # return dup_indices of i-th sim-vector
    if any(np.argwhere(similarity > 0.9)):
        duplicate_pairs.append((os.path.basename(image_pathes[row_idx]),    # return i-th image itself
         [os.path.basename(image_pathes[idx]) for idx in dup_indices]))     # return dup_pair of i-th image and its dups

    duplicates = [os.path.basename(image_pathes[idx]) for idx in dup_indices] # return list of base_name duplicates
    return duplicates, duplicate_pairs


def find_pathes(folder):
    image_pathes = []
    pathes = glob.glob(f"{folder}/*")
    for path in pathes:
        image_pathes.append(path)
    return image_pathes


def extract_extra_img_duplicates(folder):

    # find intra-class image pathes
    image_pathes = find_pathes(folder)

    # compute hashes
    hashes = compute_hashes(image_pathes)

    # find duplicates
    duplicates,duplicate_pairs = compute_similarity(hashes,image_pathes)

    dups_to_remove = pd.DataFrame(duplicates,columns=['duplicates'])
    dups_pairs = pd.DataFrame(duplicate_pairs, columns=['duplicates', 'duplicate_pairs'])

    return dups_to_remove, dups_pairs


def img_pathes(base_names, root1, root2, root3, root4):
    ''' (list,str,str) >>> list
        returns abs_pathes for img in 2020, 2019, extra_img_2019, test_img_2019
    '''
    img_list = []
    for file in base_names:
        if file[0].isdigit():
            img_list.append(os.path.join(root1, file))
        elif 'extra' in file:
            img_list.append(os.path.join(root3, file))
        elif 'test' in file:
            img_list.append(os.path.join(root4, file))
        else:
            img_list.append(os.path.join(root2, file.split('-')[1], file))

    return img_list


def extract_intra_class_duplicates(image_pathes):

    # compute hashes
    hashes = compute_hashes(image_pathes)

    # find duplicates
    duplicates, duplicate_pairs = compute_similarity(hashes, image_pathes)

    dups_to_remove = pd.DataFrame(duplicates, columns=['duplicates'])
    dups_pairs = pd.DataFrame(duplicate_pairs, columns=['duplicates', 'duplicate_pairs'])

    return dups_to_remove, dups_pairs

def extract_cross_class_duplicates(image_pathes):

    # compute hashes
    hashes = compute_hashes(image_pathes)

    # compute similiarities
    duplicates,duplicate_pairs = compute_similarity(hashes,image_pathes, cross_find=True)

    dups_to_remove = pd.DataFrame(duplicates, columns=['duplicates'])
    dups_pairs = pd.DataFrame(duplicate_pairs, columns=['duplicates', 'duplicate_pairs'])

    return dups_to_remove, dups_pairs


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    PATH_2019 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2019 COMP\train'
    PATH_2020 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2020-11-27\train_images'
    csv_2020 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2020-11-27\train.csv'
    extra_images_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\extraimages\extraimages'
    test_images_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2019 COMP\test\0'

    # extract duplicates within extra images
    print('Extracting intra-class duplicates across 2019 test images...')
    test_dups, test_dup_pairs = extract_extra_img_duplicates(test_images_folder)
    test_dups.to_csv('test_dups.csv',index=False)
    test_dup_pairs.to_csv('test_dup_pairs.csv',index=False)

    print(f'{len(test_dups)} test-class duplicates have been extracted.')

    # extract duplicates within extra images
    print('Extracting intra-class duplicates across extra images...')
    extra_dups, extra_dup_pairs = extract_extra_img_duplicates(extra_images_folder)
    extra_dups.to_csv('extra_dups2.csv',index=False)
    extra_dup_pairs.to_csv('extra_dup_pairs2.csv',index=False)

    print(f'{len(extra_dups)} extra-class duplicates have been extracted.')

    merged_dataset = helper_function.data_merge(csv_2020,PATH_2019)          # generate merged_dataset

    print('Extracting intra-class duplicates ...')
    a = pd.DataFrame([],columns=['duplicates'])                              # intra_dups placeholder
    b = pd.DataFrame([],columns=['duplicates','duplicate_pairs'])            # intra_dup_pairs placeholder

    for class_fld in range(5):
        fld = merged_dataset[merged_dataset['label']
                                == class_fld]['image_id'].values.tolist()    # return list of img of i-th class
        pathes = img_pathes(fld,PATH_2020,PATH_2019,None,None)               # return list of abs_img of i-th class
        intra_dups, intra_dup_pairs = extract_intra_class_duplicates(pathes)
        a = pd.concat([a,intra_dups],ignore_index=True)                      # concat i-th class to the intra_dups
        b = pd.concat([b,intra_dup_pairs],ignore_index=True)                 # concat i-th class to the intra_dups_pairs
    a.to_csv('intra_dups.csv', index=False)                                  # save intra_dups
    b.to_csv('intra_dup_pairs.csv', index=False)                             # save intra_dups_pairs
    print(f'{len(a)} intra-class duplicates have been extracted.')


    # remove intra-class duplicates from merged dataset before extracting
    print('Removing intra-class duplicates from merged dataset before starting ...')

    intra_dups = pd.read_csv('intra_dups.csv')
    no_intra_dup_list = merged_dataset.loc[~merged_dataset['image_id'].isin(intra_dups['duplicates'])]['image_id']

    print('The intra-class duplicates have been removed!')
    print('Extracting cross class duplicates ...')

    cross_dups, cross_dup_pairs = \
        extract_cross_class_duplicates(img_pathes(no_intra_dup_list,PATH_2020,PATH_2019,None,None)) # removed 174 images

    print(f'{len(cross_dups)} cross class duplicates have been extracted.')

    cross_dups.to_csv('cross_dups.csv',index=False)                           # save cross_dups
    cross_dup_pairs.to_csv('cross_dup_pairs.csv',index=False)                 # save cross_dups_pairs


    #  remove in-class, cross-class, extra_img, test_im duplicates from merged dataset before extracting
    print('Removing duplicates from globally merged dataset before starting ...')

    merged_dataset = helper_function.data_merge(csv_2020, PATH_2019) # generate merged_dataset
    merged_dataset = merged_dataset.drop(columns=['label'])          # drop 'label' col

    # load and merge all found dups
    intra_dups = pd.read_csv('intra_dups.csv')
    cross_dups = pd.read_csv('cross_dups.csv')
    extra_dups = pd.read_csv('extra_dups.csv')
    test_dups = pd.read_csv('test_dups.csv')

    all_dups = pd.concat([intra_dups,cross_dups,extra_dups,test_dups],ignore_index=True)

    # form lists of extra and test images
    extra_img = pd.DataFrame(os.listdir(extra_images_folder), columns=['image_id'])
    test_img = pd.DataFrame(os.listdir(test_images_folder), columns=['image_id'])

    # merge 2020 img, 2019 img, extra_img, and test_img
    global_img_list = pd.concat([merged_dataset,extra_img,test_img])

    # remove all found dups from global img list
    global_no_dups = global_img_list.loc[~global_img_list['image_id'].isin(all_dups['duplicates'])]['image_id']

    print('The duplicates have been removed!')
    print('Extracting global cross class duplicates ...')

    global_dups_extra, global_dup_pairs_extra = extract_cross_class_duplicates(
        img_pathes(global_no_dups, PATH_2020, PATH_2019,extra_images_folder,test_images_folder))
    global_dups_extra.to_csv('global_dups_extra.csv', index=False)
    global_dup_pairs_extra.to_csv('global_dup_pairs_extra.csv', index=False)
    print(f'{len(global_dups_extra)} cross class duplicates have been extracted globally.')

    print(f"all duplicates extracte in " \
          f"{round((datetime.datetime.now() - start_time).total_seconds() / 60, 1)} mins")
