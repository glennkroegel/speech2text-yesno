'''
created_by: Glenn Kroegel
date: 3 February 2019

'''

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torchvision
from tqdm import tqdm
import shutil

from loading import *
from utils import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
status_properties = ['loss', 'accuracy']

NUM_EPOCHS = 20

#############################################################################################################################

class Dense(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.bn = nn.BatchNorm1d(out_size)
        self.drop = nn.Dropout(0.5)
        self.act = nn.LeakyReLU()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.bn(self.drop(self.act(self.fc(x))))
        return x

class Conv(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, bias=bias, padding=padding)
        self.norm = nn.BatchNorm2d(out_c)
        # self.norm = nn.GroupNorm(min(out_c, out_c), out_c)
        self.drop = nn.Dropout(0.5)
        self.act = nn.LeakyReLU()
        self.in_size = in_c
        self.out_size = out_c

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, n):
        super(ResBlock, self).__init__()
        self.c1 = Conv(n, n)
        self.c2 = Conv(n, n)

    def forward(self, x):
        return x + self.c2(self.c1(x))

class ConvResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvResBlock, self).__init__()
        self.conv = Conv(in_c=in_c, out_c=out_c, stride=1)
        self.res_block = ResBlock(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block(x)
        return x

#############################################################################################################################

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.clf = nn.Sequential(*[x for x in resnet.children()][:-2])
        self.pool = nn.AdaptiveMaxPool2d(1)
        # self.clf.requires_grad_(False)

    def forward(self, x):
        bs = x.size(0)
        x = self.clf(x)
        x = self.pool(x)
        x = x.view(bs,-1)
        return x

#############################################################################################################################

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        c = 10
        self.downsample = nn.Sequential(ConvResBlock(1,c),
                                        ConvResBlock(c, c), 
                                        ConvResBlock(c, c), 
                                        ConvResBlock(c, c))
    def forward(self, x):
        bs = x.size(0)
        # x = self.norm(x)
        x = self.downsample(x)
        return x

###########################################################################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

#######################################################################################

class BaseLearner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = Resnet().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-2, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-3)
        self.epochs = epochs

        self.train_loader = torch.load('train_loader.pt')
        self.cv_loader = torch.load('cv_loader.pt')

        self.train_loss = []
        self.cv_loss = []

        self.best_loss = 1e3
        print('Model Parameters: ', count_parameters(self.model))

    def iterate(self, loader, model, criterion, optimizer, training=True):
        if training:
            model.train()
        else:
            model.eval()
        props = {k:0 for k in status_properties}
        for i, data in enumerate(loader):
            if i % 100 == 0:
                print(i)
            x, targets = data
            targets = targets.view(-1).to(device)
            preds = model(x.to(device))
            loss = criterion(preds, targets)
            props['loss'] += loss.item()
            props['accuracy'] += accuracy(preds, targets).item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_grad_norm_(model.parameters(), 0.5)
            L = len(loader)
        props = {k:v/L for k,v in props.items()}
        return props

    def step(self):
        '''Actual training loop.'''
        for epoch in tqdm(range(self.epochs)):
            train_props = self.iterate(self.train_loader, self.model, self.criterion, self.optimizer, training=True)
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()[0]
            self.train_loss.append(train_props['loss'])
            # cross validation
            with torch.no_grad():
                cv_props = self.iterate(self.cv_loader, self.model, self.criterion, self.optimizer, training=False)
                L = len(self.cv_loader)
                self.cv_loss.append(cv_props['loss'])
                if epoch % 1 == 0:
                    self.status(epoch, train_props, cv_props)
                if cv_props['loss'] < self.best_loss:
                    print('dumping model...')
                    path = 'model' + '.pt'
                    torch.save(self.model, path)
                    self.best_loss = cv_props['loss']
                    is_best = True
                save_checkpoint(
                    {'epoch': epoch + 1,
                    'lr': lr, 
                    'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(), 
                    'best_loss': self.best_loss}, is_best=is_best)
                is_best=False

    def status(self, epoch, train_props, cv_props):
        s0 = 'epoch {0}/{1}\n'.format(epoch, self.epochs)
        s1, s2 = '',''
        for k,v in train_props.items():
            s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
        for k,v in cv_props.items():
            s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
        print(s0 + s1 + s2)

if __name__ == "__main__":
    try:
        mdl = BaseLearner()
        mdl.step()
    except KeyboardInterrupt:
        print('Stopping')