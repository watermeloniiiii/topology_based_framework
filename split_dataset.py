#!/usr/bin/env python
# coding: utf-8
###################################################
#### this is step 2: split the segmented image patches into different sets ####

import pandas as pd
import numpy as np
import os
from skimage import io

def split_data(dir, identifier='',remove=True, sampling='random'):
    '''
    :param dir: the directory to read raw data from
    :param identifier: use to distinguish between different dataset, for example, landsat-7, landsat-8 and sentinel-2
    :param year:
    :param remove: if remove, delete the already existed csv files to avoid repeatative operations
    :return:
    '''
    training_path = os.path.join(dir, 'training'+identifier+'_'+sampling+'.csv')
    validation_path = os.path.join(dir, 'validation'+identifier+'_'+sampling+'.csv')
    testing_path = os.path.join(dir, 'testing'+identifier+'_'+sampling+'.csv')
    for csv in [training_path, validation_path, testing_path]:
        if os.path.exists(csv) and remove:
            os.remove(csv)

    if sampling == 'grid':
        ## reference is used to determine the size the aoi and then help to divide the area into several grids
        reference = io.imread(r'./IW_CDL_2019.tif')
        x_num = reference.shape[1] // 512 ## the number of image 512 * 512 patches along the x axis
        y_num = reference.shape[0] // 512 ## the number of image 512 * 512 patches along the y axis
    training = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    validation = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    testing = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])

    time_steps = 24 ## sentinel-2 has 24 time steps whereas landsat just has 8
    for date_index, date in enumerate(range(0, time_steps)):
        date_dir = os.path.join(dir, str(date))
        target_dir = os.path.join(dir, 'target')
        for image in os.listdir(date_dir):
            if image.endswith('tif'):
                img_index = int(image.split(".")[0])
                assert sampling in ['random', 'grid'], \
                'please choose one of the two sampling stragety: random or grid'
                if sampling == 'random':
                    sign = np.random.choice([0,1,2], p=[0.7, 0.1, 0.2])
                    if sign == 0:
                        training = training.append(pd.DataFrame.from_dict(
                            data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                            orient='index', columns=['img_dir', 'target_dir']), ignore_index=True)
                    elif sign == 1:
                        validation = validation.append(pd.DataFrame.from_dict(
                            data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                            orient='index', columns=['img_dir', 'target_dir']), ignore_index=True)
                    else:
                        testing = testing.append(pd.DataFrame.from_dict(
                            data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                            orient='index', columns=['img_dir', 'target_dir']), ignore_index=True)

                if sampling == 'grid':
                    ## here I used 3*3 grid
                    i = img_index // y_num
                    j = img_index - (y_num * i)
                    grid_x = i // (x_num // 3)
                    grid_y = j // (y_num // 3)
                    if grid_x == 3:
                        grid_x = 2
                    if grid_y == 3:
                        grid_y = 2
                    grid_index = grid_x * 3 + grid_y + 1

                    if grid_index in [1, 2, 3, 4, 5, 6]:
                        training = training.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                                                                                      orient='index',columns=['img_dir', 'target_dir']), ignore_index=True)
                    if grid_index in [7]:
                        validation = validation.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                                                                                      orient='index',columns=['img_dir', 'target_dir'] ), ignore_index=True)
                    if grid_index in [8, 9]:
                        testing = testing.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                                                                                      orient='index',columns=['img_dir', 'target_dir'] ), ignore_index=True)
    testing.to_csv(os.path.join(dir, 'testing'+identifier+'_'+sampling+'.csv'), index=False)
    training.to_csv(os.path.join(dir, 'training'+identifier+'_'+sampling+'.csv'), index=False)
    validation.to_csv(os.path.join(dir, 'validation'+identifier+'_'+sampling+'.csv'), index=False)

if __name__ == '__main__':
    root_dir = r'./raw_time_series'
    split_data(root_dir, identifier='_s2_nir_2019', sampling='grid')





