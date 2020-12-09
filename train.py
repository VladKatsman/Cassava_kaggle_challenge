from data_preprocessing.train_val_split import split_train_val_5_fold
from dataset.dataset import dataloader
from dataset.augmentations import tensor_batch2PIL
import numpy as np
from PIL import Image

if __name__ == '__main__':
    ROOT = '/home/noteme/PycharmProjects/comp/data/cassava-disease'
    SEED = 1337
    BATCH_SIZE = 64
    folds, images_list = split_train_val_5_fold(root=ROOT, seed=SEED)
    for key in folds.keys():
        dataset_train = dataloader(split=folds[key]['train'], images_list=images_list, batch_size=BATCH_SIZE)
        dataset_val = dataloader(split=folds[key]['val'], images_list=images_list, batch_size=BATCH_SIZE)

        # for testing purposes, remove later
        for batch in dataset_train:
            batch = tensor_batch2PIL(batch[0])
            for image in batch:
                image = image.numpy().astype(np.uint8)
                image = image.transpose(1, 2, 0)
                Image.fromarray(image).show()
