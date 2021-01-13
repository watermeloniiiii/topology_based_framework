import numpy as np
import shutil
import os
import random

def make_partial_copy(src, dst, number):
    '''
    :param src: source files
    :param dst: the destination to copy files to
    :param number: how many files you want to copy
    :return:
    '''
    if not os.path.exists(dst):
        os.makedirs(dst)
    file_list = os.listdir(src)
    random.shuffle(file_list)
    count = 0
    for file in file_list:
        shutil.copyfile(os.path.join(src, file), os.path.join(dst, file))
        count += 1
        if count >= number:
            break

if __name__ == '__main__':
    make_partial_copy(r'E:\DigitalAG\liheng\IW\sentinel-2\NIR\testing\img', r'E:\DigitalAG\liheng\IW\sentinel-2\NIR\msi_testing\img', 1000)