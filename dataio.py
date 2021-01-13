#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chenxi
"""
import torchvision.transforms as transforms
import torch.utils as utils
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utility_functions import get_hist2d, get_target, get_separability
from utility_functions import get_coordinate

def scale_percentile_n(matrix):
    # matrix = matrix.transpose(1, 2, 0)
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 0, axis=0)
    # print(mins,"-----",mins[None:])
    maxs = np.percentile(matrix, 100, axis=0)  # - mins
    # print(maxs,"-----",maxs[None:])
    matrix = (matrix - mins[None:]) / (maxs[None:] * 0.5) - 1
    matrix = np.reshape(matrix, [w, h, d]).astype(np.float32)
    matrix = matrix.clip(-1, 1)
    # print(matrix)
    return matrix

class BuildingDataset(utils.data.Dataset):
    def __init__(self, dir, transform=None, scale=True, target=True):
        self.dir = dir
        self.transform = transform
        self.img_list = os.listdir(dir)
        self.scale = scale
        self.target = target
        if self.target:
            self.target_list = os.listdir(dir.replace('img', 'target'))

    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        if self.img_list[index].endswith('tif'):
            image = io.imread(os.path.join(self.dir, self.img_list[index])).astype(np.int16)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]

            if self.scale:
                image = scale_percentile_n(image)
            sample = {}
            sample['image'] = image.transpose(2,0,1)
            sample['patch'] = self.img_list[index].split('.')[0]
            sample['name'] = self.img_list[index]
            if self.target:
                target = io.imread(os.path.join(self.dir.replace('img', 'target'), self.target_list[index])).transpose(
                    2, 0, 1)
                sample['label'] = target

            if self.transform:
                sample = self.transform(sample)
            return sample

def generate_hist(dir, folder_id='', file_id='', indexes=[2,20], positive=True, test=False):
    '''
    :param dir: the csv file containing the path for image and target
    :param folder_id: the folder name
    :param file_id: the file name
    :param indexes: what time steps to include
    :param positive: whether we want positive samples (those recognizable heat maps)
    :param test: if this is training and validation data, it should be false
    :return:
    '''

    ## for corn, only between index 10 and 20 and the JM distance > 0.9 will the histogram has its target

    ## create folders
    for item in ['img', 'target', 'RGB']:
        if not os.path.exists(os.path.join(root_dir, folder_id+ item)):
            os.makedirs(os.path.join(root_dir, folder_id + item))


    img_dir = np.array(pd.read_csv(dir)['img_dir'])
    target_dir = np.array(pd.read_csv(dir)['target_dir'])
    np.random.seed(2000)
    dirs = np.c_[img_dir, target_dir]
    np.random.shuffle(dirs)
    zeros_count = 0

    for index, img_target in enumerate(dirs):
        patch_index = img_target[0].split('.')[:-1][0].split('\\')[-1]
        date_index = int(img_target[0].split('.')[:-1][0].split('\\')[-2].split('_')[0])
        if not test:
            if positive:
                if date_index >= indexes[0] and date_index <= indexes[1]:
                    img = io.imread(img_target[0])/ 10000
                    target = io.imread(img_target[1])
                    ## you should have a folder containing image patches of historical CDL
                    mask = io.imread(os.path.join(r'./target_historical', img_target[1].split('\\')[-1]))
                    # make sure the shape is (channel, height, weight)
                    # each image is a square
                    if img.shape[0] == img.shape[1]:
                        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]

                    """
                    convert the image to 3 classes:
                    0 for background
                    1 for corn
                    2 for soybean
                    """

                    code_class = [1, 5]
                    if (target == 1).sum() == 0:
                        data_target == np.zeros([3,128,128])
                    else:
                        arr_class = np.zeros_like(target)
                        for i_cur, code_cur in enumerate(code_class):
                            arr_class[target == code_cur] = i_cur + 1
                        list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
                        mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
                        sep = get_separability(img, arr_class)
                        if (sep > 0.9).sum()==3:
                            list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                            x_coor = get_coordinate(list_img2)
                            y_coor = get_coordinate(list_img2, x_coor=False)

                            data_target = get_target\
                            (
                                list_img,
                                separability=sep,
                                threshold_separability=0.9,
                                crop='all'
                            )
                            name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                                      np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                                      x_coor[np.newaxis, np.newaxis, :,:],
                                                      y_coor[np.newaxis, np.newaxis, :,:]), axis=1).squeeze().astype(np.int16))
                            if (mask==0).sum() <= 512*256:
                                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                          data_target.squeeze().astype(np.int8))
                            if (mask==0).sum() > 512*256:
                                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                          np.zeros([3, 128, 128]).astype(np.int8))
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                                      np.array(list_img).squeeze().astype(np.int16))
                            zeros_count += 1
                            if zeros_count > 1000:
                                break


            if not positive:
                img = io.imread(img_target[0]) / 10000
                target = io.imread(img_target[1])
                mask = io.imread(os.path.join(r'./target_historical', img_target[1].split('\\')[-1]))
                # make sure the shape is (channel, height, weight)
                # each image is a square
                if img.shape[0] == img.shape[1]:
                    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
                code_class = [1, 5]
                if (target == 1).sum() == 0:
                    data_target == np.zeros([1, 1, 128, 128])
                else:
                    arr_class = np.zeros_like(target)
                    for i_cur, code_cur in enumerate(code_class):
                        arr_class[target == code_cur] = i_cur + 1
                    list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
                    mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
                    sep = get_separability(img, arr_class)

                    if date_index < indexes[0] or date_index > indexes[1]:
                        list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                        x_coor = get_coordinate(list_img2)
                        y_coor = get_coordinate(list_img2, x_coor=False)
                        name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                                  np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                                  x_coor[np.newaxis, np.newaxis, :, :],
                                                  y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                                      np.int16))
                        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                                  np.array(list_img).squeeze().astype(np.int16))
                        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                  np.zeros([3, 128, 128]).astype(np.int8))
                        zeros_count += 1

                    else:
                        if sep.mean() < 0.9:
                            list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                            x_coor = get_coordinate(list_img2)
                            y_coor = get_coordinate(list_img2, x_coor=False)
                            data_target = get_target(
                                list_img,
                                separability=sep,
                                threshold_separability=0.9,
                                crop='all'
                            )
                            name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                                      np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                                      x_coor[np.newaxis, np.newaxis, :, :],
                                                      y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                                          np.int16))
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                      data_target.squeeze().astype(np.int8))
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                                      np.array(list_img).squeeze().astype(np.int16))
                            zeros_count += 1
                    if zeros_count > 100:
                        break

        if test:
            candidate = np.zeros(68)
            candidate[0] = 660
            for i_c in range(1, 68):
                if i_c % 4 != 0:
                    candidate[i_c] = candidate[i_c - 1] + 1
                if i_c % 4 == 0:
                    candidate[i_c] = candidate[i_c - 1] + 19
            # if int(patch_index) in candidate:
            img = io.imread(img_target[0]) / 10000
            target = io.imread(img_target[1])
            mask = io.imread(os.path.join(r'./target_historical', img_target[1].split('\\')[-1]))
            if img.shape[0] == img.shape[1]:
                img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
            code_class = [1, 5]
            if (target == 1).sum() == 0:
                data_target == np.zeros([1, 1, 128, 128])
            else:
                arr_class = np.zeros_like(target)
                for i_cur, code_cur in enumerate(code_class):
                    arr_class[target == code_cur] = i_cur + 1
                list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
                list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                x_coor = get_coordinate(list_img2)
                y_coor = get_coordinate(list_img2, x_coor=False)
                mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
                name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                          np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                          x_coor[np.newaxis, np.newaxis, :, :],
                                          y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(np.int16))

                # io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                #           np.array(list_img).squeeze().astype(np.int16))
                # io.imsave(os.path.join(os.path.join(root_dir, folder_id + '_target'), name),
                #           data_target.squeeze().astype(np.int8))

def remove_file(dir):
    [os.remove(os.path.join(dir, file)) for file in os.listdir(dir)]

def match_folders(f1_dir, f2_dir, identifier='', rename=False):
    f1_list = np.array(os.listdir(f1_dir))
    f2_list = np.array(os.listdir(f2_dir))
    union = np.intersect1d(f1_list, f2_list)
    diff_f1 = np.setdiff1d(f1_list, union)
    diff_f2 = np.setdiff1d(f2_list, union)
    [os.remove(os.path.join(f1_dir, file)) for file in diff_f1]
    [os.remove(os.path.join(f2_dir, file)) for file in diff_f2]
    # if rename:
    #     [os.rename(os.path.join(f1_dir, file), os.path.join(f1_dir,
    #                                                         file.replace(file.split('.')[0], file.split('.')[0]+identifier))) for file in os.listdir(f1_dir)]
    #     [os.rename(os.path.join(f2_dir, file), os.path.join(f2_dir,
    #                                                         file.replace(file.split('.')[0], file.split('.')[0]+identifier))) for file in os.listdir(f2_dir)]



def check_data_target_RGB(data_dir, identifier='training_ls7', sensor='landsat'):
    if sensor == 'landsat':
        xticks = {'raw':[0, 31, 63, 95, 127], 'scale': [0, 0.2, 0.4, 0.6, 0.8]}
        yticks = {'raw': [0, 31, 63, 95, 127], 'scale': [0.6, 0.45, 0.3, 0.15, 0]}
    else:
        xticks = {'raw': [0, 31, 63, 95, 127], 'scale': [0, 0.075, 0.15, 0.225, 0.3]}
        yticks = {'raw': [0, 31, 63, 95, 127], 'scale': [0.6, 0.45, 0.3, 0.15, 0]}
    img_dir = os.path.join(data_dir, identifier+'_img')
    for dir in os.listdir(img_dir):
        img = io.imread(os.path.join(img_dir, dir))
        target = io.imread(os.path.join(img_dir.replace('img', 'target'), dir))
        rgb = io.imread(os.path.join(img_dir.replace('img', 'RGB'), dir))
        ##
        # font1 = {
        #     'family': 'Arial',
        #     'weight': 'bold',
        #     'size': 16
        #     # 'style': 'italic'
        # }
        # ax = plt.gca()
        # labels = ax.get_xticklabels() + ax.get_yticklabels()
        # for label in labels:
        #     label.set_fontname('Arial')
        #     # label.set_style('italic')
        #     label.set_fontsize(16)
        #     label.set_weight('bold')
        # ax.set_ylabel('SWIR ', font1)
        # ax.set_xlabel('RDED', font1)
        # plt.xticks([0, 31, 63, 95, 127], [0, 0.075, 0.15, 0.225, 0.3])
        # plt.yticks([0, 31, 63, 95, 127], [0, 0.15, 0.3, 0.45, 0.6])
        fig = plt.figure(1, figsize=(14, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(img[:,:, 0], cmap='gist_ncar_r')
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        plt.subplot(1, 4, 2)
        plt.imshow(img[:,:, 1], cmap='gist_ncar_r')
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        plt.subplot(1, 4, 3)
        plt.imshow(target.sum(axis=2))
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        plt.subplot(1, 4, 4)
        for i in range(rgb.shape[2]):
            min_v = rgb[:, :, i].min()
            max_v = rgb[:, :, i].max()
            rgb[:, :, i] = (rgb[:, :, i] - min_v) / (max_v - min_v) * 255
        plt.imshow(rgb)
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        folder_name = r'visualize_' + identifier
        if not os.path.exists(os.path.join(root_dir, folder_name)):
            os.makedirs(os.path.join(root_dir, folder_name))
        plt.savefig(os.path.join(os.path.join(root_dir, folder_name), dir))

def combine_folders(src, dst):
    import shutil
    for file in os.listdir(src):
        shutil.copyfile(os.path.join(src, file), os.path.join(dst, file))


global root_dir
root_dir = r'./'
bins_range = [0.3, 0.6]
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

if __name__ == '__main__':
    operations = {'remove': False, 'match': False, 'combine': False, 'generate': False}
    op = [4]
    for index, key in enumerate(operations.keys()):
        if index + 1 in op:
            operations[key] = True

    if operations['remove']:
        remove_file(os.path.join(root_dir, 'testing_1_img'))
        remove_file(os.path.join(root_dir, 'testing_1_target'))

    if operations['match']:
        for year in range(2017, 2018):
            match_folders(os.path.join(root_dir, r'training_img'),
                          os.path.join(root_dir, r'visualize_training'))
            match_folders(os.path.join(root_dir, r'training_img'),
                          os.path.join(root_dir, r'training_target'))

    if operations['combine']:
        type = 'rgb'
        folder_names = ['training_ls7_' + type, 'training_ls8_' + type]
        # folder_names = ['training_18_negative2_' + type]
        for year in range(2013, 2018):
            for folder in folder_names:
                src = os.path.join(os.path.join(r'E:\DigitalAG\liheng\IW\landsat\2nd', str(year)), folder)
                dst = r'E:\DigitalAG\liheng\IW\landsat\training\\' + type
                combine_folders(src, dst)

    if operations['generate']:
        generate_hist(r'./raw_time_series/training_s2_nir_2019_random.csv', folder_id='training_',
                      file_id='', positive=False, test=True)
        check_data_target_RGB(root_dir, identifier='training', sensor='sentinel-2')

