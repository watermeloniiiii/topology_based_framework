#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chenxi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from dataio import BuildingDataset
from torch.autograd import Variable
import time
import os
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau, MultiStepLR
import numpy as np
import wandb
from utility_functions import project_to_target, confusion_matrix

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def collate_fn(data):
    for i in range(0, len(data)):
        data[i]['label'] = data[i]['label'].sum(axis=0)
    patch = []
    name = []
    image = torch.stack([torch.from_numpy(b['image']) for b in data], 0)
    label = torch.stack([torch.from_numpy(b['label']) for b in data], 0)[:, np.newaxis, :, :]
    patch = patch.append(b['patch'] for b in data)
    name = name.append(b['name'] for b in data)
    return {'image': image, 'patch': patch, 'name':name, 'label':label}


class Trainer(object):
    def __init__(self, net, file_path, train_dir, vali_dir, test_dir, model_dir, cuda=False, identifier=None,
                 hyperparams=None):
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
        self.bs = hyperparams['batch_size']
        self.epoch = hyperparams['epochs']
        self.crop = hyperparams['crop']
        self.train_data = BuildingDataset(dir=self.train_dir, transform=None)
        self.train_loader = utils.data.DataLoader(self.train_data, batch_size=self.bs, shuffle=True, num_workers=8,
                                             collate_fn=collate_fn)
        self.vali_data = BuildingDataset(dir=self.vali_dir, transform=None)
        self.vali_loader = utils.data.DataLoader(self.vali_data, batch_size=self.bs, shuffle=True, num_workers=8)
        self.makefolders()
        self.net.apply(inplace_relu)

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
        model_path = os.path.join(model_folder, self.identifier + '_notselected')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_folder = model_folder
        self.model_path = model_path

    def select_scheduler(self, optimizer):
        if self.lr_schedule == 'SLR':
            scheduler = StepLR(optimizer,
                               step_size=4 * len(self.train_loader),
                               gamma=self.hyperparams['gamma'])
        elif self.lr_schedule == 'CLR':
            scheduler = CyclicLR(optimizer,
                                 base_lr=self.hyperparams['CLR_params'][0],
                                 max_lr=self.hyperparams['CLR_params'][1],
                                 step_size_up=self.hyperparams['CLR_params'][2] * len(self.train_loader))
        elif self.lr_schedule == 'MSLR':
            scheduler = MultiStepLR(optimizer,
                                    milestones=self.hyperparams['milestones'],
                                    gamma=self.hyperparams['gamma'])
        return scheduler

    def train_model(self):
        ############ parameters initialization ############
        torch.backends.cudnn.deterministic = True
        since = time.time()
        optimizer = self.select_optimizer()
        scheduler = self.select_scheduler(optimizer)
        softmax = torch.nn.functional.softmax
        number_of_crop = len(self.crop)
        cm = np.zeros([number_of_crop + 1, number_of_crop + 1])
        train_loss = np.zeros([self.epoch])
        vali_loss = np.zeros([self.epoch])
        PA_training = np.zeros([self.epoch])
        UA_training = np.zeros([self.epoch])
        PA_vali = np.zeros([self.epoch])
        UA_vali = np.zeros([self.epoch])

        ############ network training ############
        for i in range(self.epoch):
            self.net.train()
            accu_loss_training = 0
            accu_loss_vali = 0
            for j, sample in enumerate(self.train_loader, 0):
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False)
                label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor).sum(dim=1)
                weights = torch.FloatTensor(self.hyperparams['weight']).cuda()
                criterion = nn.CrossEntropyLoss(weight=weights)
                if self.cuda:
                    image = image.cuda()
                    label = label.cuda()
                prediction = self.net(image)
                loss = criterion(prediction, label.long())
                accu_loss_training += loss
                filter = torch.ge(torch.max(softmax(prediction, dim=1), dim=1)[0], 0.995).type(torch.cuda.FloatTensor)
                prediction = torch.argmax(softmax(prediction, dim=1), dim=1) * filter
                cm += confusion_matrix(pred=prediction, target=label, classes=[0, 1, 2, 3])
                loss.backward()
                optimizer.step()
                if self.lr_schedule != 'none':
                    scheduler.step()
            for c in range(1, number_of_crop+1):
                PA_training[i] = cm[c, c] / cm[c, :].sum()
                UA_training[i] = cm[c, c] / cm[:, c].sum()
            accu_loss_training = accu_loss_training.cpu().detach().numpy()
            train_loss[i] = accu_loss_training / len(self.train_loader)
            wandb.log({"train_loss": train_loss[i].item()}, step=i)
            for c in range(number_of_crop):
                wandb.log({self.crop[c] + "_UA": UA_training[c], self.crop[c] + "_PA": PA_training[c]}, step=i)

            ############ network validation ############
            self.net.eval()
            with torch.no_grad():
                for k, sample in enumerate(self.vali_loader, 0):
                    image = Variable(sample["image"], requires_grad=False)
                    label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor).sum(
                            axis=1)
                    criterion = nn.CrossEntropyLoss()
                    if self.cuda:
                        image = image.cuda()
                        label = label.cuda()
                    prediction = self.net(image)
                    loss_all_vali += criterion(prediction, label.long())
                    prediction = torch.argmax(softmax(prediction, dim=1), dim=1)
                    cm += confusion_matrix(pred=prediction, target=label, classes=[0, 1, 2, 3])
                for c in range(1, number_of_crop + 1):
                    PA_vali.append(cm[c, c] / cm[c, :].sum())
                    UA_vali.append(cm[c, c] / cm[:, c].sum())
                loss_all_vali = loss_all_vali.cpu().detach().data.numpy()
                vali_loss[i] = loss_all_vali / len(self.vali_loader)
                wandb.log({"vali_loss": vali_loss[i].item()}, step=i)
                for c in range(number_of_crop):
                    wandb.log({self.crop[c] + "_UA_vali": UA_vali[c], self.crop[c] + "_PA_vali": PA_vali[c]}, step=i)
                wandb.log({"vali_loss": train_loss[i].item()}, step=i)
                for c in range(number_of_crop):
                    wandb.log({self.crop[c] + "_UA": UA_vali[c], self.crop[c] + "_PA": PA_vali[c]}, step=i)
            self.save_model(i)

            elapse = time.time() - since
            print(
                "Epoch:{}/{}\n"
                "training_loss:{}\n"
                "vali_loss:{}\n"
                "Time_elapse:{}\n'".format(
                i + 1, self.epoch,
                round(train_loss[i], 5),
                round(vali_loss[i], 5),
                elapse))


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