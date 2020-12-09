import os
from PIL import Image
import numpy as np
from shutil import *
import glob
from tqdm import tqdm
# org_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2019 COMP\train\cbb'
# dup_rmvd_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2020-11-29 TRAIN ORIGINAL T-CROSS-FLD DUP RMVD\cbb'
#
#
#
# org_folder_cont = set(os.listdir(org_folder))
# dup_rmvd_folder_cont = set(os.listdir(dup_rmvd_folder ))
# print(len(org_folder_cont),len(dup_rmvd_folder_cont))
# difference = org_folder_cont.difference(dup_rmvd_folder_cont)
# print(difference)
#
# for picture in difference:
#     image = Image.open(os.path.join(org_folder,picture))
#     image.show()

org_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2020-11-29 TRAIN ORIGINAL T-CROSS-FLD DUP RMVD\2020-11-30 TRAIN'
# dest_folder = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2020-11-29 TRAIN ORIGINAL T-CROSS-FLD DUP RMVD\2020-11-30 TRAIN'
our_sizes = []
print(glob.glob(org_folder))
for i in tqdm(os.listdir(org_folder)):
    our_sizes.append(Image.open(os.path.join(org_folder,i)).size)
our_sizes = np.array(our_sizes)
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.font_manager import FontProperties
fig, ax = plt.subplots()


x, y = our_sizes[:,1],our_sizes[:,0]

ax.scatter(x, y,marker="*",color='m',label="Image size (pxl)")
font = FontProperties()
ax.legend()
ax.grid(True)
plt.xlabel("WIDTH")
plt.ylabel("HEIGHT")
plt.legend(loc='upper right')
ax.set_xlabel("WIDTH",fontsize='large', fontweight='bold')
ax.set_ylabel("HEIGHT",fontsize='large', fontweight='bold')

ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(100))
plt.show()


print()
# k = glob.glob(f"{org_folder}/*")
# our_image = r'D:\CASSAVA COMPET\2020-11-27 CASSAVA-LEAF-CHALLENGE\2. Preparead Data\2020-11-29 TRAIN ORIGINAL\cbsd\train-cbsd-1446'
# ids = [373,916,920,928,959,1178,1221,1405,1482,2250,2295,2404,3212,3378]
# for i in ids:
#     image = Image.open(k[i])
#     image.show()
# print(k[498])

# for folder in glob.glob(f"{org_folder}/*"):
#     # print(*glob.glob(f"{folder}/*"))
#     for image in tqdm(glob.glob(f"{folder}/*")):
#         copy(image,dest_folder)

