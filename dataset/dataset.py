from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from dataset.augmentations import get_train_transforms


LBL_DESC_DICT = {'cbb': 0,
                 'cbsd': 1,
                 'cgm': 2,
                 'cmd': 3,
                 'healthy': 4}


class CassavaDataset(Dataset):
    def __init__(self, split, images_list):

        super().__init__()
        self.split = split
        self.images_list = images_list # tmp
        self.augmentations = get_train_transforms()

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index: int):

        path = self.images_list[self.split[index]]
        # path = self.images_list[self.split[0]]
        # label_idx = 0
        image = np.array(Image.open(path))
        label_string = os.path.basename(path).split('-')[1]
        label_idx = LBL_DESC_DICT[label_string]


        if self.augmentations:
            image = self.augmentations(image=image)['image']

        return image, label_idx


def dataloader(split, images_list, batch_size):
    dataset = CassavaDataset(split, images_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
