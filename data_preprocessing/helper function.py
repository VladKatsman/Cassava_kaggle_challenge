import pandas as pd
import numpy as np
import os


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


def data_merge(csv_2020,path2019):
    ''' takes train2020_csv and merge it with the labeled 2019 dataset
    (DataFrame[2 col], str) >> DataFrame[2 col]
    '''
    LBL_DESC_DICT = {0: 'cbb',
                     1: 'cbsd',
                     2: 'cgm',
                     3: 'cmd',
                     4: 'healthy'}

    LIST_2019 = [file for root,dirs,files in os.walk(PATH_2019) for file in files] # retrieve images from 5 folders
    df = pd.DataFrame(LIST_2019,columns=['image_id'])                              # create df 2019
    df['label'] = np.nan                                                           # insert column 'label' with Nan

    for key in LBL_DESC_DICT:
        df.loc[df['image_id'].str.contains(LBL_DESC_DICT[key]), ['label']] = key   # assign labels

    df['label'] = df['label'].astype('Int64')                                      # convert labels-val into int
    csv_2020 = pd.read_csv(csv_2020)
    merged_train = pd.concat([csv_2020,df],ignore_index=True)                      # merge 2020 and 2019 datasets
    return merged_train

if __name__ == '__main__':

    PATH_2019 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2019 COMP\train'
    PATH_2020 = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2020-11-27\train_images'
    csv_2020 =  r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\2020-11-27\train.csv'
    merged_dataset = data_merge(csv_2020,PATH_2019)
    base_names = merged_dataset['image_id'].values.tolist()
    merged_dataset_abs = pd.DataFrame(img_pathes(base_names,PATH_2020,PATH_2019))
    merged_dataset_abs['label'] = merged_dataset['label']
    merged_dataset.to_csv('merged_dataset.csv',index=False)
    merged_dataset = pd.read_csv('merged_dataset.csv')

    print(len(merged_dataset) + len(os.listdir(r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\1. Original Data\extraimages\extraimages')) - 824)

