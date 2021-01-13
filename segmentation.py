#!/usr/bin/env python
# coding: utf-8
###################################################
#### this is step 1: segment time series imagery into small image patch ####

###########################################################################################################
###########################################################################################################
#### **********************           READ ME FIRST                               *********************####
#### the data that will be processed are downloaded satellite imagery of different time steps          ####
#### e.g., for sentinel-2, from June to September, we have 24 time steps if we use 5 days as interval  ####
####                                 NAMING CONVENTION                                                 ####
#### using location_year_timestep will be a good practice, e.g., the first time step in Iowa in 2019   ####
#### will be IW_2019_0. 0 is used to match python's indexing mechanism                                 ####
###########################################################################################################
###########################################################################################################


import numpy as np
from osgeo import gdal
from gdalconst import *
import os


def readTifImageWithWindow(img_path, x_range, y_range, index, target=False, require_proj=True, continue_count=True):
    '''

    :param img_path:
    :param x_range: x size of the image patch
    :param y_range: y size of the image patch
    :param index: time step
    :param target: whether the image is target (in this case, the CDL)
    :param require_proj: whether to maintain projection when exporting small image patches
    :param continue_count: if not, will delete all image patches in the current folder
    :return:
    '''
    img = gdal.Open(img_path, GA_ReadOnly)
    # if require_proj:
    #     img_geotrans = img.GetGeoTransform()  # crs transform information
    #     img_proj = img.GetProjection()  # projection
    #     top_left_x = img_geotrans[0]  # x coordinate of upper lefe corner
    #     w_e_pixel_resolution = img_geotrans[1]  # horizontal resolution
    #     top_left_y = img_geotrans[3]  # y coordinate of upper lefe corner
    #     n_s_pixel_resolution = img_geotrans[5]  # vertical resolution
    if not target:
        folder = os.path.join(save_dir, str(index))
    else:
        folder = os.path.join(save_dir, r'target')
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(folder + ' has been created')
    if not continue_count:
        [os.remove(os.path.join(folder, file)) for file in os.listdir(folder)]
    non_overlap_segmentation(img, folder, x_range, y_range, target, require_proj, continue_count)


def non_overlap_segmentation(img, folder, x_range, y_range, target=False, require_proj=False, continue_count=False):
    '''
    :param img:
    :param folder: the folder to save image patches
    :param x_range:
    :param y_range:
    :param target:
    :param require_proj:
    :param continue_count:
    :return:
    '''
    if require_proj:
        img_geotrans = img.GetGeoTransform()
        img_proj = img.GetProjection()
        top_left_x = img_geotrans[0]
        w_e_pixel_resolution = img_geotrans[1]
        top_left_y = img_geotrans[3]
        n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range ## calculate how many patches can be segmented along the x axis
    y_num = img.RasterYSize // y_range
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)

    ## determine the original count
    ## if continue_count is false, then the index image patch in the upper left corner should be 0
    if continue_count:
        original_index = len(os.listdir(os.path.join(folder, 'input')))
    else:
        original_index = 0

    for i in range(0, x_num):
        for j in range(0, y_num):
            x_off_patch = i * x_range
            y_off_patch = j * y_range
            if not target:
                patch = img_array[:, y_off_patch:y_off_patch + y_range, x_off_patch:x_off_patch + x_range]
                ## determine if the patch has enough valid pixel
                ## invalid pixels may from cloud coverage
                valid_ratio = (patch != 0).mean()
                if valid_ratio < 0.8:
                    continue
            if target:
                patch = img_array[y_off_patch:y_off_patch + y_range, x_off_patch:x_off_patch + x_range][np.newaxis, :,
                        :]

            patch_name = os.path.join(folder, str(i * y_num + j + original_index) + '.tif')

            if require_proj:
                new_top_left_x = top_left_x + x_off_patch * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - y_off_patch * np.abs(n_s_pixel_resolution)
                dst_transform = (
                new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4], img_geotrans[5])
                writeTif(patch, patch_name, require_proj, dst_transform, img_proj)
            else:
                writeTif(patch, patch_name)


def writeTif(bands, path, require_proj=False, transform=None, proj=None):
    if bands is None or bands.__len__() == 0:
        return
    else:
        band1 = bands[0]
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            if require_proj:
                dataset.SetGeoTransform(transform)
                dataset.SetProjection(proj)
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")

if __name__ == '__main__':
    root_dir = r'.' ## make sure raw time series images are in the root dir
    save_dir = r'.\raw_time_series'

    for i in range(0, 24):
        img_path = os.path.join(root_dir, 'IW_2019_' + str(i) + '.tif')
        readTifImageWithWindow(img_path, 512, 512, i, target=False, continue_count=False)

    target_path = os.path.join(root_dir, 'IW_CDL_2019.tif')
    readTifImageWithWindow(target_path, 512, 512, 0, target=True, continue_count=False)




