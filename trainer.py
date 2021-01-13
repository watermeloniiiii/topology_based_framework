#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: huijian
@modified by Chenxi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from dataio import BuildingDataset
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import os
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau, MultiStepLR
import matplotlib.pyplot as plt
import numpy as np
import wandb
from utility_functions import project_to_target


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):
    def __init__(self, net, file_path, train_dir, vali_dir, test_dir, model_dir, cuda=False, identifier=None, hyperparams=None):
        self.file_path = file_path
        self.train_dir = train_dir
        self.vali_dir = vali_dir
        self.test_dir = test_dir
        self.model_dir = model_dir
        self.net = net
        self.hyperparams = hyperparams
        self.opt = hyperparams['optimizer']
        self.learn_rate = hyperparams['lr']
        self.cuda = cuda
        self.identifier = identifier
        self.lr_schedule = hyperparams['lr_scheduler']
        self.weight = hyperparams['weight']
        self.wd = hyperparams['weight_decay']

    def select_optimizer(self):
        optimizer = None
        if (self.opt == 'Adam'):
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                   lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'RMS'):
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'SGD'):
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                  lr=self.learn_rate, momentum=0.9, weight_decay=self.wd)
        elif (self.opt == 'Adagrad'):
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'Adadelta'):
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.net.parameters()),
                                       lr=self.learn_rate, weight_decay=self.wd)
        return optimizer

    def makefolders(self):
        '''
        This function is used to create necessary folders to save models, textbooks and images
        :return:
        '''
        model_folder = self.model_dir
        model_path = os.path.join(model_folder, self.identifier+'_notselected')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_folder = model_folder
        self.model_path = model_path


    def train_model(self, epoch, bs):
        torch.backends.cudnn.deterministic = True
        self.makefolders()
        since = time.time()
        optimizer = self.select_optimizer()
        txt = open(os.path.join(self.model_path, self.identifier + ".txt"), 'w')
        txt.writelines(self.identifier + '\n')
        txt.close()
        train_data = BuildingDataset(dir=self.train_dir, transform=None)
        train_loader = utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=8)
        self.net.apply(inplace_relu)
        if self.lr_schedule == 'CLR':
            # scheduler = StepLR(optimizer, step_size=4 * len(train_loader), gamma=0.75)
            # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
            scheduler = CyclicLR(optimizer,
                                 base_lr=self.hyperparams['lr_scheduler_params'][0],
                                 max_lr=self.hyperparams['lr_scheduler_params'][1],
                                 step_size_up= self.hyperparams['lr_scheduler_params'][2] * len(train_loader))
        vali_data = BuildingDataset(dir=self.vali_dir, transform=None)
        vali_loader = utils.data.DataLoader(vali_data, batch_size=bs, shuffle=True, num_workers=8)
        train_loss = np.zeros([epoch])
        vali_loss = np.zeros([epoch])
        train_F1 = np.zeros([epoch])
        vali_F1 = np.zeros([epoch])
        test_F1 = np.zeros([epoch])
        for i in range(epoch):
            self.net.train()
            print(get_lr(optimizer))
            txt = open(os.path.join(self.model_path, self.identifier + ".txt"), 'a')
            loss_all_training = 0
            loss_all_vali = 0

            TP_train, FP_train, TN_train, FN_train = 0, 0, 0, 0
            TP_vali, FP_vali, TN_vali, FN_vali = 0, 0, 0, 0
            TP_test, FP_test, TN_test, FN_test = 0, 0, 0, 0
            vali_pre, vali_rec = 0, 0
            train_pre, train_rec = 0, 0
            test_pre, test_rec = 0, 0

            for j, sample in enumerate(train_loader, 0):
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False)[:,:, :, :]
                if self.hyperparams['crop'] == 'soybean':
                    label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)[:, 2:3, :, :]
                if self.hyperparams['crop'] == 'corn':
                    label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)[:, 1:2, :, :]
                label[label != 0] = 1
                weights = torch.zeros([label.size(0), 1, 128, 128])
                weights[label == 1] = self.hyperparams['weight']
                weights[label == 0] = 1
                ## when your positive samples and negative samples have unbalanced size, using weight parameter
                criterion = nn.BCELoss(weight=weights.cuda())
                if self.cuda:
                    image = image.cuda()
                    label = label.cuda()

                prediction = self.net(image)
                loss = criterion(prediction, label)
                loss_all_training += loss
                balance = 0.9
                prediction = torch.ge(prediction, balance).type(torch.cuda.FloatTensor)
                TP_train += torch.eq(label + prediction, 2).sum().cpu().detach().data.numpy()
                FP_train += torch.eq(label - prediction, -1).sum().cpu().detach().data.numpy()
                FN_train += torch.eq(prediction - label, -1).sum().cpu().detach().data.numpy()
                TN_train += torch.eq(prediction + label, 0).sum().cpu().detach().data.numpy()

                loss.backward()
                optimizer.step()
                if self.lr_schedule != 'none':
                    scheduler.step()

            loss_all_training = loss_all_training.cpu().detach().data.numpy()
            train_loss[i] = loss_all_training / len(train_loader)
            try:
                train_pre = TP_train / (TP_train + FP_train)
            except:
                pass
            try:
                train_rec = TP_train / (TP_train + FN_train)
            except:
                pass
            try:
                train_F1[i] = 2 * (train_pre * train_rec) / (train_pre + train_rec)
            except:
                pass
            wandb.log({"train_loss": train_loss[i].item(), "train_F1": train_F1[i].item()}, step=i)

            self.net.eval()
            with torch.no_grad():
                for k, sample in enumerate(vali_loader, 0):
                    image = Variable(sample["image"], requires_grad=False)[:, :, :, :]
                    if self.hyperparams['crop'] == 'soybean':
                        label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)[:, 2:3, :, :]
                    if self.hyperparams['crop'] == 'corn':
                        label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor)[:, 1:2, :, :]
                    label[label != 0] = 1
                    criterion = nn.BCELoss()
                    if self.cuda:
                        image = image.cuda()
                        label = label.cuda()
                    prediction = self.net(image)
                    loss_all_vali += criterion(prediction, label).item()
                    balance = 0.9
                    prediction = torch.ge(prediction, balance).type(torch.cuda.FloatTensor)
                    TP_vali += torch.eq(label + prediction, 2).sum().item()
                    FP_vali += torch.eq(label - prediction, -1).sum().item()
                    FN_vali += torch.eq(prediction - label, -1).sum().item()
                    TN_vali += torch.eq(prediction + label, 0).sum().item()
                    # accuracy_all_vali += torch.eq(prediction, label).type(torch.FloatTensor).mean().item()

                vali_loss[i] = loss_all_vali / len(vali_loader)
                try:
                    vali_pre = TP_vali / (TP_vali + FP_vali)
                except:
                    pass
                try:
                    vali_rec = TP_vali / (TP_vali + FN_vali)
                except:
                    pass
                try:
                    vali_F1[i] = 2 * (vali_pre * vali_rec) / (vali_pre + vali_rec)
                except:
                    pass
                wandb.log({"vali_loss": vali_loss[i].item(), "vali_pre":vali_pre, "vali_rec": vali_rec, "vali_F1": vali_F1[i].item()}, step=i)
                self.save_model(i)

                elapse = time.time() - since
                print(
                    "Epoch:{}/{}\n"
                    "Train_loss:{}, Train_pre:{}, Train_rec:{}, Train_F1:{}\n"
                    "Vali_loss:{}, Vali_pre:{}, Vali_rec:{}, Vali_F1:{}\n"
                    "Time_elapse:{}\n'".format(
                    i + 1, epoch,
                    round(train_loss[i], 5),
                    round(train_pre, 5),
                    round(train_rec, 5),
                    round(train_F1[i], 5),
                    round(vali_loss[i], 5),
                    round(vali_pre, 5),
                    round(vali_rec, 5),
                    round(vali_F1[i], 5),
                    elapse))

            # if i >= 0:
            #     test_data = BuildingDataset(dir=self.test_dir, transform=None, target=False)
            #     test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
            #     precision = np.zeros(len(test_loader))
            #     recall = np.zeros(len(test_loader))
            #     count = 0
            #     for l, sample in enumerate(test_loader, 0):
            #         image = Variable(sample["image"], requires_grad=False)[:, :, :,:]
            #         patch = sample['patch']
            #         if self.cuda:
            #             image = image.cuda()
            #         pred = self.net(image)
            #         balance = 0.9
            #         pred = torch.ge(pred, balance).type(torch.cuda.FloatTensor)
            #         cm, pred_img, target_img = project_to_target(pred, patch, 512, crop=self.hyperparams['crop'], use_cm=True, use_mask=False)
            #         TP = cm[1, 1]
            #         TN = cm[0, 0]
            #         FP = cm[0, 1]
            #         FN = cm[1, 0]
            #         if (pred_img != 0).sum() > 512 ** 2 * 0.1:
            #             precision[l] = TP / (TP + FP)
            #             recall[l] = TP / (TP + FN)
            #             count += 1
            #     wandb.log({"test_pre": precision.sum()/count, "test_rec": recall.sum()/count}, step=i)

    def save_model(self, epoch):
        torch.save(self.net, os.path.join(self.model_path, self.identifier + 'e_' + str(epoch) + ".pkl"))

    def restore_model(self, dir=None, user_defined=False):
        if not user_defined:
            self.net = torch.load(os.path.join(self.model_path, self.identifier + ".pkl"))
        if user_defined:
            self.net = torch.load(dir)


    def predict(self, image):
        self.net.eval()
        prediction = self.net(image)
        return prediction
