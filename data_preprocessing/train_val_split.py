import os
import random
from glob import glob
from sklearn.model_selection import KFold


def split_train_val_5_fold(root, seed=1337):
    # init splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_dic = {idx: {"train": [], "val": []} for idx in range(5)}
    images_list = []
    train_dir = os.path.join(root, 'train')

    train_class_folders = os.listdir(train_dir)
    cur_idx = 0
    for c in train_class_folders:
        train_class_folder = os.path.join(train_dir, c)
        images = glob(f"{train_class_folder}/*")
        images_list.extend(images)
        kf.get_n_splits(images)
        for idx, (train, val) in enumerate(kf.split(images)):
            fold_dic[idx]["train"].extend(train + cur_idx)
            fold_dic[idx]["val"].extend(val + cur_idx)
        cur_idx += len(images)
    return fold_dic, images_list


if __name__ == '__main__':
    root = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2020-11-29 TRAIN ORIGINAL T-CROSS-FLD DUP RMVD\2020-11-30 TRAIN'
    seed = 1337
    split_train_val_5_fold(root, seed=seed)
