#!/usr/bin/env python
# coding: utf-8

# In[70]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from dataio import BuildingDataset
import torch
import torch.utils as utils
from model import Unet
import wandb
from trainer import Trainer
from torch.autograd import Variable
from utility_functions import project_to_target, save_fig, writeTif, save_fig_spectral
import pandas as pd
from dataio import check_data_target
from skimage import io
from dataio import scale_percentile_n
from ResNet_fromGit import ResNetUNet

global training, prediction, cuda
training = False
prediction = True
cuda = True  #是否使用GPU
seed = 11
mode = 'multiple'
crop = 'corn'
task = 'landsat'
if task == 'landsat':
    root_dir = r'E:\DigitalAG\liheng\IW\landsat'
    model_dir = r'E:\DigitalAG\liheng\IW\landsat\model_' + crop
if task == 's2_rdeg':
    root_dir = r'E:\DigitalAG\liheng\IW\sentinel-2\RDEG\1718'
    model_dir = r'E:\DigitalAG\liheng\IW\sentinel-2\RDEG\1718\model_' + crop
if task == 's2_nir':
    root_dir = r'E:\DigitalAG\liheng\IW\sentinel-2\NIR'
    model_dir = r'E:\DigitalAG\liheng\IW\sentinel-2\NIR\model_' + crop

train_dir = os.path.join(root_dir, r'training\img')
vali_dir = os.path.join(root_dir, r'validation\img')
test_dir = os.path.join(root_dir, r'2018_testing\ls8_img')

hyperparameter = dict(
    batch_size=16,
    epochs=200,
    lr=0.001,  ## this param does not work for the CLR scheduler
    optimizer='SGD',
    lr_scheduler='CLR',
    lr_scheduler_params=[0.006, 0.011, 25],
    weight_decay=0,
    model='resnet',
    hidden_layer=[48, 64],
    weight=140,
    crop=crop,
    model_index='2',
    state='iowa',
    year='2013-2017',
    # info='147 nagative samples were added to the dataset'
)


def accuracy_access(trainer, test_dir, model, model_folder):
    trainer.restore_model(model, True)
    test_data = BuildingDataset(dir=test_dir, transform=None, target=False)
    test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
    precision = np.zeros(len(test_loader))
    recall = np.zeros(len(test_loader))
    count = 0
    for l, sample in enumerate(test_loader):
        image = Variable(sample['image'], requires_grad=False)
        # label = Variable(sample["label"], requires_grad=False)
        # label[label != 0] = 1
        patch = sample['patch']
        label = io.imread(os.path.join(test_dir.replace('_img', '_target'), patch[0] + '.tif'))
        # candidate = np.zeros(68)
        # candidate[0] = 660
        # for i_c in range(1, 68):
        #     if i_c % 4 != 0:
        #         candidate[i_c] = candidate[i_c-1] + 1
        #     if i_c % 4 == 0:
        #         candidate[i_c] = candidate[i_c-1] + 19
        if cuda:
            image = image.cuda()
            # label = label.cuda()
        pred = trainer.predict(image)
        balance = 0.8
        pred = torch.ge(pred, balance).type(torch.cuda.FloatTensor)
        cm, pred_img, target_img = project_to_target(pred, patch, 512, crop=hyperparameter['crop'], task=task, use_cm=True, use_mask=False)
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        if (pred_img != 0).sum() > 512 ** 2 * 0.1:
            precision[l] = TP / (TP + FP)
            recall[l] = TP / (TP + FN)
            # print (precision[l], recall[l], (target_img!=0).sum() / (target_img.shape[0] * target_img.shape[1]))
            # print ((pred_img!=0).sum(), TP, precision[l], (target_img!=0).sum() / (target_img.shape[0] * target_img.shape[1]))
            # txt = open((os.path.join(root_dir, r'result_soybean\accuracy.txt')), 'a')
            # txt.writelines('{}, {}, \n'.format(round(precision[l], 5), round(recall[l], 5)))
            # txt.close()
            count += 1
            result_folder = os.path.join(root_dir, r'result_' + hyperparameter['crop'] + '_2018')
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            writeTif(pred_img, os.path.join(result_folder, str(patch[0]) + '.tif'))
            # if int(patch[0].split('_')[0]) in [3,5,6]:
            # #     save_fig(pred, label, image, pred_img, target_img, os.path.join(root_dir, r'4_channel\visualize_testing_356'), patch[0], 'all')
            # save_fig(pred, label, image, pred_img, target_img, r'E:\DigitalAG\liheng\IW\sentinel-2\NIR\\'+crop+'_sample',
            #          patch[0], 'all', title = 'pre:{}, rec:{}'.format(np.round(precision[l], 4), np.round(recall[l], 4)))

    print(
        'the mean precision is {}, the mean recall is {}'.format(precision.sum() / count, recall.sum() / count))
    if not os.path.exists(os.path.join(model_dir, 'accuracy.csv')):
        df = pd.DataFrame(data=[], columns=['model_name', 'precision', 'recall', 'info'])
        df.to_csv(os.path.join(model_dir, 'accuracy.csv'), index=False)
    df = pd.read_csv(os.path.join(model_dir, 'accuracy.csv'))
    df = df.append(pd.DataFrame.from_dict(
        data={0: [model_folder, precision.sum() / count, recall.sum() / count, 'for 2018 soybean']},
        orient='index', columns=['model_name', 'precision', 'recall', 'info']), ignore_index=True)
    df.to_csv(os.path.join(model_dir, 'accuracy.csv'), index=False)

def run(gpu=0):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print('-------------------------')
        print (torch.backends.cudnn.version())
        print (torch.__version__)
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            # os.environ['CUDA_ENABLE_DEVICES'] = '0'
            print("There are {} CUDA devices".format(torch.cuda.device_count()))
            print("Setting torch GPU to {}".format(gpu))
            torch.cuda.set_device(gpu)
            print("Using device:{} ".format(torch.cuda.current_device()))
            torch.cuda.manual_seed(seed)
            # torch.backends.cudnn.enabled = False

    # building the net
    model = Unet(features=hyperparameter['hidden_layer'])
    # model = ResNetUNet(1)
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    if cuda:
        model = model.cuda()
    identifier = hyperparameter['state'] + '_' + hyperparameter['crop'] + '_' + hyperparameter['model_index']
    trainer = Trainer(net=model, file_path=root_dir, train_dir=train_dir, vali_dir=vali_dir, test_dir=test_dir, model_dir=model_dir,
                     hyperparams=hyperparameter, identifier=identifier, cuda=cuda)


    # training
    if training:
        print ("begin training!")
        wandb.init(entity='chenxilin', project='nc_landsat_1317', name=identifier,
                   config=hyperparameter)
        trainer.train_model(epoch=hyperparameter['epochs'], bs=hyperparameter['batch_size'])

    if prediction:
        print("restore the model")
        for model_folder in os.listdir(model_dir):
            if model_folder.split('_')[-1] in ['notselected']:
                continue
            model_folder_path = os.path.join(model_dir, model_folder)
            if not os.path.isdir(model_folder_path):
                continue
            if mode == 'single':
                model = os.path.join(model_folder_path, str(model_folder) + '.pkl')
                accuracy_access(trainer, test_dir, model, model_folder)
            if mode == 'multiple':
                for epoch in range(197, 198):
                    current_epoch = model_folder.split('_')[-1]
                    model = os.path.join(model_folder_path, model_folder.replace(current_epoch, str(epoch)) + '.pkl')
                    accuracy_access(trainer, test_dir, model, model_folder.replace(current_epoch, str(epoch)) + '.pkl')



if __name__ == "__main__":
    run()





