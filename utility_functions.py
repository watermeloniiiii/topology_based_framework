#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from gdalconst import *
import pandas as pd
from skimage import io
import os
from torchvision.utils import make_grid

np.random.seed(1003)


def get_hist2d(arr, arr_class=None, bins=129, bins_range=[0.3, 0.6], normed=True, b_invert_yaxis=True):
    if arr_class is None:
        arr_class = np.zeros((arr.shape[2], arr.shape[3]))

    code_class = np.unique(arr_class)
    n_class = len(code_class)

    n_img = arr.shape[0]  ## time phase
    list_img = []
    list_xedges, list_yedges = [], []
    for i_img in range(n_img):
        arr_cur = arr[i_img].transpose((1, 2, 0))
        list_class = []
        list_xedges_class, list_yedges_class = [], []
        for i_class in range(0, n_class):
            ## return point pair (n by 2)
            data_flat = arr_cur[arr_class == i_class]
            indi_valid = np.logical_and(data_flat[:, 0] != 0, data_flat[:, 1] != 0)
            data_flat = data_flat[indi_valid, :]
            if not isinstance(bins, list):
                bins0 = np.linspace(0, bins_range[0], bins)
                bins1 = np.linspace(0, bins_range[1], bins)
                new_bins = [bins1, bins0]
            if isinstance(bins, list):
                new_bins = bins
            # x = first dim -> x = cols, y = second dim -> y = reverted rows
            data_hist, yedges, xedges = np.histogram2d(
                data_flat[:, 1], data_flat[:, 0], bins=new_bins
            )
            if b_invert_yaxis:
                data_hist = data_hist[::-1]
                yedges = yedges[::-1]
            list_class.append(data_hist)
            list_xedges_class.append(xedges)
            list_yedges_class.append(yedges)
        list_img.append(list_class)
        list_xedges.append(list_xedges_class)
        list_yedges.append(list_yedges_class)
    return np.array(list_img), np.array(list_xedges), np.array(list_yedges)

def JM_distance(x, y):
    '''
    Calculate the Jeffries-Matusita Distance between x and y
    x and y have the same number of variables (columns).
    Each row is an observation.
    '''
    # dif_mean = np.mean(x, axis=0, keepdims=True) - np.mean(y, axis=0, keepdims=True)
    dif_mean = np.empty((1, x.shape[1]))
    for i in range(x.shape[1]):
        dif_mean[0, i] = x[:, i].mean() - y[:, i].mean()

    comatrix_x = np.cov(x, rowvar=False)
    comatrix_y = np.cov(y, rowvar=False)
    comatrix_mean = 0.5 * (comatrix_x + comatrix_y)
    alpha = (
            0.125 * np.dot(
        np.dot(dif_mean, np.linalg.inv(comatrix_mean)),
        dif_mean.T
    )
            + 0.5 * np.log(
        np.linalg.det(comatrix_mean) /
        np.sqrt(np.linalg.det(comatrix_x) * np.linalg.det(comatrix_y))
    )
    )
    output = np.sqrt(2 * (1 - np.exp(-alpha)))[0, 0]
    return (output)

def get_separability(arr, arr_class):
    """
    :param arr:
    :param arr_class:
    :return:
    """
    code_class = np.unique(arr_class)
    n_class = len(code_class)
    n_img = arr.shape[0]
    list_separability = []
    for i_img in range(n_img):
        arr_cur = arr[i_img].transpose((1, 2, 0))
        list_class = []
        for i_class in range(n_class):
            for j_class in range(n_class):
                indi_valid = ~(np.isnan(arr_cur[:, :, 0]) | np.isnan(arr_cur[:, :, 1]))
                data_flat_pos = arr_cur[(arr_class == code_class[i_class]) & indi_valid]
                # data_flat_neg = arr_cur[(arr_class != code_class[i_class]) & indi_valid]
                data_flat_neg = arr_cur[(arr_class == code_class[j_class]) & indi_valid]
                separability = JM_distance(data_flat_pos, data_flat_neg)
                list_class.append(separability)
        list_separability.append(list_class)
    return np.array(list_separability)[:,[1,2,5]]

def get_target(
        data_hist,
        percentile_pure=50,
        crop='corn',
        separability=None,
        threshold_separability=None):

    n_img, n_class, n_y, n_x = data_hist.shape
    list_img = []
    for i_img in range(n_img):
        '''
        1. find which class has most pixels in each grid
        2. data_hist for image at each date should have size of (classes, bins1, bins2) in our case, class is 3
        3. for each pixel in the bins1-bins2 grid, three classes 
        could all have values, which means there's overlap in the feature combination
        4. when all classes have no value in a grid, the idx_max is assigned to 0
        '''
        idx_max = np.argmax(data_hist[i_img], axis=0)
        list_class = []
        crop_index = {
            'background':[0],
            'corn':[1],
            'soybean':[2],
            'all':[0,1,2]
        }
        i_class_list = crop_index[crop]
        for i_class in i_class_list:
            if separability.mean() > threshold_separability:
                indi_cur = (idx_max == i_class)
                ## non-zero pixel in 2d histogram
                indi_pos = 0 < data_hist[i_img, i_class]
                ## make sure the target is the class that has the largest value and exclude grids having no values
                data_target_cur = data_hist[i_img, i_class] * indi_cur * indi_pos
                data_flat_cur = data_hist[i_img, i_class][indi_cur & indi_pos]
                data_flat_cur = np.sort(data_flat_cur)
                cumsum_flat_cur = np.cumsum(data_flat_cur)
                try:
                    cumsum_pure = cumsum_flat_cur[-1] * (100 - percentile_pure) / 100
                    idx_threshold_pure = int(np.clip(
                        np.where(cumsum_flat_cur > cumsum_pure)[0][0],
                        0, len(data_flat_cur) - 1
                    ))
                    threshold_pure = data_flat_cur[idx_threshold_pure]
                    candidate = data_target_cur > threshold_pure
                    data_target_cur[candidate] = i_class
                    data_target_cur[~candidate] = 0
                except:
                    pass
                #                 data_target_cur = data_target_cur / threshold_pure
            else:
                # data_target_cur = np.ones((n_y, n_x)) * 3
                data_target_cur = np.zeros((n_y, n_x))
            list_class.append(data_target_cur)
        list_img.append(list_class)
    return np.array(list_img)

def get_class_ratio(arr, idx_class):
    '''
    :param arr:
    :param idx_class:
    :return:
    '''
    h, w = arr.shape
    ratios = []
    for idx in idx_class:
        count = (arr==idx).sum()
        ratios.append(count / (h * w))
    return ratios

def get_coordinate(arr, x_coor=True):
    coor1 = arr.copy().squeeze()
    coor1[coor1!=0] = 1
    coor2 = np.zeros_like(arr).squeeze()
    row, col = arr.shape[2], arr.shape[3]
    if x_coor:
        for i in range(row):
            try:
                coor2[i,:] = np.arange(0, col)
            except:
                pass
    if not x_coor:
        for i in range(col):
            coor2[:,i] = np.arange(0, row)
    return coor1*coor2


def confusion_matrix_from_hist(data_hist_cur):
    ''' data_hist_cur: (n_class, row, col)'''
    list_class = []
    list_img = []
    n_class = data_hist_cur.shape[0]
    data_argmax = np.argmax(data_hist_cur, axis=0)
    cm = np.zeros((n_class, n_class))
    for i_class in range(n_class):
        for j_class in range(n_class):
            cm[i_class, j_class] = (
                    data_hist_cur[i_class]
                    * (data_argmax == j_class)
            ).sum()
    TP = np.zeros(n_class)
    TN = np.zeros(n_class)
    FP = np.zeros(n_class)
    FN = np.zeros(n_class)
    F1 = np.zeros(n_class)
    for c in range(n_class):
        TP[c] = cm[c, c]
        TN[c] = np.array([cm[i, i] for i in range(n_class)]).sum() - cm[c, c]
        FP[c] = cm[:, c].sum() - cm[c, c]
        FN[c] = cm[c, :].sum() - cm[c, c]
        if TP[c] != 0 and TN[c] != 0:
            precision = TP[c] / (TP[c] + FP[c])
            recall = TP[c] / (TP[c] + FN[c])
            F1[c] = 2 * precision * recall / (precision + recall)
        else:
            F1[c] = 0

    for i_class in range(n_class - 1):
        if (F1 > 0.8).sum() == 3:
            indi_cur = data_argmax == i_class
            ## non-zero pixel in 2d histogram
            indi_pos = 0 < data_hist_cur[i_class]
            ## make sure the target is the class that has the largest value and exclude grids having no values
            data_target_cur = data_hist_cur[i_class] * indi_cur * indi_pos
            data_flat_cur = data_hist_cur[i_class][indi_cur & indi_pos]
            data_flat_cur = np.sort(data_flat_cur)
            cumsum_flat_cur = np.cumsum(data_flat_cur)

            cumsum_pure = cumsum_flat_cur[-1] * (100 - 50) / 100
            idx_threshold_pure = int(np.clip(
                np.where(cumsum_flat_cur > cumsum_pure)[0][0],
                0, len(data_flat_cur) - 1
            ))
            threshold_pure = data_flat_cur[idx_threshold_pure]
            candidate = data_target_cur > threshold_pure
            data_target_cur[candidate] = i_class
            data_target_cur[~candidate] = 0
        else:
            data_target_cur = np.zeros([data_hist_cur.shape[1], data_hist_cur.shape[2]])
        list_class.append(data_target_cur)
    list_img.append(list_class)
    return np.array(list_img).sum(axis=1)[:, np.newaxis, :, :], F1







def confusion_matrix(arr, target, classes):
    assert isinstance(classes, list), 'please input a class list'
    class_num = len(classes)
    cm = np.zeros([class_num, class_num])
    arr = arr.squeeze()
    target = target.squeeze()
    for pred_c in classes:
        for targ_c in classes:
            cm[targ_c, pred_c] = ((arr==pred_c)*(target==targ_c)).sum()
    return cm


def project_to_target(arr, patch_index, patch_size, classes=[0,1], crop='corn', task='landsat', use_cm=False, use_mask=True):

    '''
    :param arr: the predicted target, should be integer
    :param patch_index:
    :param patch_size:
    :return:
    '''

    b,c,h,w = arr.shape
    arr = arr.cpu().data.numpy()
    pred = np.zeros([b, patch_size, patch_size])
    h_interval = 0.6 / h
    w_interval = 0.8 / h
    cm = np.zeros([len(classes), len(classes)])
    for idx_b in range(0, b):
        '''
        get corresponding CDL
        '''
        date = patch_index[idx_b].split('_')[0]
        pt = patch_index[idx_b].split('_')[1]
        if task == 'landsat':
            raw_img = io.imread(os.path.join(r'E:\DigitalAG\liheng\IW\landsat\2018\\' + str(date)+'_ls8', str(pt) + '.tif'))
            target = io.imread(os.path.join(r'E:\DigitalAG\liheng\IW\landsat\2018\target', str(pt) + '.tif'))
        if task == 's2_nir':
            raw_img = io.imread(os.path.join(r'E:\DigitalAG\liheng\IW\sentinel-2\NIR\2019\\' + str(date), str(pt) + '.tif'))
            target = io.imread(os.path.join(r'E:\DigitalAG\liheng\IW\sentinel-2\NIR\2019\target', str(pt) + '.tif'))
        if task == 's2_rdeg':
            raw_img = io.imread(os.path.join(r'G:\My Drive\Digital_Agriculture\Liheng\iowa\sentinel-2\RDEG\2019\\' + str(date), str(pt) + '.tif'))
            target = io.imread(os.path.join(r'G:\My Drive\Digital_Agriculture\Liheng\iowa\sentinel-2\RDEG\2019\target', str(pt) + '.tif'))
        mask = io.imread(os.path.join(r'G:\My Drive\Digital_Agriculture\Liheng\iowa\target_historical', str(pt) + '.tif'))
        mask[mask!=0]=1
        arr_cur = arr[idx_b].squeeze()
        coord = np.argwhere(arr_cur == 1)
        if len(coord) != 0:
            for item in coord:
                swir_candi = np.logical_and(raw_img[:,:,1] > (h - item[0] -1) * h_interval * 10000, raw_img[:,:,1] < (h - item[0]) * h_interval * 10000)
                rded_candi = np.logical_and(raw_img[:,:,0] > item[1]  * w_interval * 10000, raw_img[:,:,0] < (item[1] + 1) * w_interval * 10000)
                pred[idx_b, swir_candi*rded_candi] = 1
                if use_mask:
                    pred = pred * mask
        if use_cm:
            if crop == 'corn':
                target[target != 1] = 0
            if crop == 'soybean':
                target[target != 5] = 0
                target[target != 0] = 1
            cm += confusion_matrix(pred, target, classes)
            return cm, pred, target
        if not use_cm:
            if crop == 'corn':
                target[target != 1] = 0
            if crop == 'soybean':
                target[target != 5] = 0
                target[target != 0] = 1
            return pred, target

def save_fig(pred_hist, label, image, pred_img, target_img, save_dir, fig_name, mode='single', title=None):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from matplotlib import cm
    fig1 = plt.figure(1, figsize=(14, 14))
    if mode == 'all':
        # plt.subplot(2, 2, 1)
        # plt.imshow(make_grid(pred_hist, nrow=8).cpu().numpy().transpose(1, 2, 0))
        # plt.subplot(2, 2, 2)
        # plt.imshow(make_grid(target_hist, nrow=8).cpu().numpy().transpose(1, 2, 0))
        gist_ncar_r = cm.get_cmap('gist_ncar_r', 256)
        cmap1 = gist_ncar_r(np.arange(0, 256))
        cmap1[:1, :] = np.array([1, 1, 1, 1])
        cmap1 = ListedColormap(cmap1)
        cmap = ListedColormap(np.array([[255, 255, 255, 255],
                                        # [255, 211, 0, 255]
                                        [38, 112, 0, 255]
                                        ]) / 255.0)
        if title:
            plt.rcParams["figure.titlesize"] = 'large'
            plt.rcParams["figure.titleweight"] = 'bold'
            plt.rcParams["font.family"] = 'Arial'
            plt.rcParams["font.size"] = '18'
            plt.suptitle(title)



        plt.subplot(2, 2, 1)
        plt.imshow(make_grid(image, nrow=2, padding=10).cpu().numpy().transpose(1, 2, 0)[:, :, 0], cmap=cmap1)
        plt.xticks([0, 31, 63, 95, 127], np.round(np.linspace(0, 0.3, 5), 4))
        plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')

        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

        plt.subplot(2, 2, 2)
        test = pred_hist.cpu().numpy()
        plt.imshow(pred_hist.cpu().numpy().squeeze(), cmap=cmap)
        plt.xticks([0, 31, 63, 95, 127], np.round(np.linspace(0, 0.3, 5), 4))
        plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')

        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

        # plt.subplot(2, 3, 3)
        # for i in range(label.shape[2]):
        #     min_v = label[:, :, i].min()
        #     max_v = label[:, :, i].max()
        #     label[:, :, i] = (label[:, :, i] - min_v) / (max_v - min_v) * 255
        # plt.imshow(label)
        # plt.xticks([0, 31, 63, 95, 127], [0, 0.075, 0.15, 0.225, 0.3])
        # plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])

        plt.subplot(2, 2, 3)
        plt.imshow(pred_img.squeeze(), cmap=cmap)
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')
        plt.yticks([12, 112, 212, 312, 412, 512], [500, 400, 300, 200, 100, 0])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

        plt.subplot(2, 2, 4)
        plt.imshow(np.array(target_img).squeeze(), cmap=cmap)
        plt.yticks([12, 112, 212, 312, 412, 512], [500, 400, 300, 200, 100, 0])
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

    if mode == 'single':
        plt.imshow(pred_img.squeeze())
    # plt.grid()
    # plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,fig_name))

def save_fig_spectral(pred_hist, label, image, pred_img, target_img, save_dir, fig_name, mode='single'):
    if mode == 'all':
        plt.subplot(2, 3, 1)
        test = make_grid(label, nrow=2, padding=10).cpu().numpy()
        plt.imshow(make_grid(label, nrow=2, padding=10).cpu().numpy().squeeze()[:, :, 0], cmap='rainbow')
        plt.xticks([0, 31, 63, 95, 127], [0, 0.075, 0.15, 0.225, 0.3])
        plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])

        plt.subplot(2, 3, 2)
        plt.imshow(label.cpu().numpy().squeeze()[:, :, 1], cmap='rainbow')
        plt.xticks([0, 31, 63, 95, 127], [0, 0.075, 0.15, 0.225, 0.3])
        plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])

        plt.subplot(2, 3, 3)
        plt.imshow(label.cpu().numpy().squeeze()[:, :, 2], cmap='rainbow')
        plt.xticks([0, 31, 63, 95, 127], [0, 0.075, 0.15, 0.225, 0.3])
        plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])

        plt.subplot(2, 3, 4)
        plt.imshow(pred_img.squeeze())
        plt.subplot(2, 3, 5)
        plt.imshow(np.array(target_img).squeeze())
    if mode == 'single':
        plt.imshow(pred_img.squeeze())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,fig_name))

def writeTif(bands, path, require_proj=False, transform=None, proj=None):
    if bands is None or bands.__len__() == 0:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            if require_proj:
                dataset.SetGeoTransform(transform)  # 写入仿射变换参数
                dataset.SetProjection(proj)  # 写入投影
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")

def read_json(dir):
    import json
    import os
    site_dir = os.listdir(dir)
    for site in site_dir:
        for item in os.listdir(os.path.join(dir, site)):
            if item.endswith('json'):
                with open(os.path.join(os.path.join(dir, site), item)) as f:
                    try:
                        data = json.load(f)
                        cloud_cover = data['properties']['cloud_cover']
                        date = data['properties']['acquired'][:10]
                        print ('site: {}, date: {}, cloud cover: {}'.format(site, date, cloud_cover))
                    except:
                        print(site + ':', 'no data')

def create_map(corn_dir, soybean_dir, CDl_dir, filename):
    '''
    generate the training sample map for soybean and corn
    :param corn_dir:
    :param soybean_dir:
    :return:
    '''
    x_range = 512
    y_range = 512

    img = gdal.Open(CDl_dir, GA_ReadOnly)
    img_geotrans = img.GetGeoTransform()
    img_proj = img.GetProjection()

    top_left_x = img_geotrans[0]
    w_e_pixel_resolution = img_geotrans[1]
    top_left_y = img_geotrans[3]
    n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range
    y_num = img.RasterYSize // y_range

    x_size = x_num * x_range
    y_size = y_num * y_range
    x_off = 0
    y_off = 0
    cdl = img.ReadAsArray(0, 0, x_size, y_size)
    corn_sum = (cdl==1).sum()
    soybean_sum = (cdl==5).sum()
    weight_map = np.zeros([2, y_size, x_size])

    if not os.path.exists(os.path.join(root_dir, filename+'.csv')):
        df = pd.DataFrame(data=[], columns=['time index', 'corn_number', 'corn_precision', 'corn_recall',
                                            'soybean_number', 'soybean_precision', 'soybean_recall'])
        df.to_csv(os.path.join(root_dir, filename+'.csv'), index=False)
    time_index = []
    for ts in range(1, 8):
        weight_map = np.zeros([2, y_size, x_size])
        # time_index.append(ts)
        time_index = [ts]
        for crop_index, crop_dir in enumerate([corn_dir, soybean_dir]):
            crop_list = os.listdir(crop_dir)
            for i in crop_list:
                if i.endswith('.tif'):
                    if int(i.split('.')[0].split('_')[0]) not in time_index:
                        continue
                    img = io.imread(os.path.join(crop_dir, i))
                    index = int(i.split('.')[0].split('_')[1])
                    col = index // y_num
                    row = index % y_num
                    # pred_collaged[crop_index, row * y_range : (row + 1) * y_range , col * x_range : (col + 1) * x_range ] = img
                    weight_map[crop_index, row * y_range : (row + 1) * y_range , col * x_range : (col + 1) * x_range] += img
                    # if img.sum() !=0:
                    #     corn_sum += (cdl[row * y_range : (row + 1) * y_range , col * x_range : (col + 1) * x_range]==1).sum()
                    #     soybean_sum += (cdl[row * y_range: (row + 1) * y_range, col * x_range: (col + 1) * x_range] == 5).sum()

        background = (np.sum(weight_map, axis=0) == 0)
        corn = (np.argmax(weight_map, axis=0) == 0)
        soybean = (np.argmax(weight_map, axis=0) == 1)
        # corn = (pred_collaged[0] == 1) * (pred_collaged[1] == 0)
        # soybean = (pred_collaged[0] == 0) * (pred_collaged[1] == 1)
        classification = np.zeros([y_size, x_size])
        classification[corn] = 1
        classification[soybean] = 2
        classification[background] = 0
        print ((classification==1).sum())
        print (((classification==1)*(cdl==1)).sum()/(classification==1).sum())
        print(((classification == 1) * (cdl == 1)).sum() / corn_sum)

        print((classification == 2).sum())
        print(((classification == 2) * (cdl == 5)).sum() / (classification == 2).sum())
        print(((classification == 2) * (cdl == 5)).sum() / soybean_sum)

        df = pd.read_csv(os.path.join(root_dir, filename+'.csv'))
        df = df.append(pd.DataFrame.from_dict(
            data={0: [time_index, (classification==1).sum(),
                      ((classification==1)*(cdl==1)).sum()/(classification==1).sum(),
                      ((classification == 1) * (cdl == 1)).sum() / corn_sum,
                      (classification == 2).sum(),
                      ((classification == 2) * (cdl == 5)).sum() / (classification == 2).sum(),
                      ((classification == 2) * (cdl == 5)).sum() / soybean_sum]},
            orient='index', columns=['time index', 'corn_number', 'corn_precision', 'corn_recall',
                                                'soybean_number', 'soybean_precision', 'soybean_recall']), ignore_index=True)
        df.to_csv(os.path.join(root_dir, filename+'.csv'), index=False)
        # new_top_left_x = top_left_x + x_off * np.abs(w_e_pixel_resolution)
        # new_top_left_y = top_left_y - y_off * np.abs(n_s_pixel_resolution)
        #
        # dst_transform = (
        #     new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4],
        #     img_geotrans[5])
        # driver = gdal.GetDriverByName("GTiff")
        # path = os.path.join(root_dir, r'2018_sample_' + str(ts)+'.tif')
        # dataset = driver.Create(path, classification.shape[1], classification.shape[0], 1, gdal.GDT_Float32)
        # if dataset is not None:
        #     dataset.SetGeoTransform(dst_transform)  # 写入仿射变换参数
        #     dataset.SetProjection(img_proj)  # 写入投影
        #     for i in range(1):
        #         dataset.GetRasterBand(i + 1).WriteArray(classification[:, :])



if __name__ == '__main__':
    root_dir = r'E:\DigitalAG\liheng\IW\landsat'
    create_map(os.path.join(root_dir, r'result_corn_2018'), os.path.join(root_dir, r'result_soybean_2018'),
               os.path.join(root_dir, 'IW_CDL_2018.tif'), 'sample_map_accuracy_single_2018')


    # dir = np.array(pd.read_csv(r'D:\My Drive\Digital_Agriculture\Liheng\iowa\2019\training.csv')['target_dir'])
    # for idx, target in enumerate(dir):
    #     patch_index = target.split('.')[:-1][0].split('\\')[-1]
    #     ratios = get_class_ratio(io.imread(target), [1])
    #     print (patch_index, ratios)
    # read_json(r'D:\My Drive\Digital_Agriculture\Morocco\Planet_Morocco\temporal\site\site2')
