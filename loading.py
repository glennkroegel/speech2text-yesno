'''
created_by: Glenn Kroegel
date: 3 February 2020

description: Create dataloaders to feed for training

'''
import pandas as pd
import numpy as np
import glob
import random

import torch
import torch.nn.functional as F
import os
from torch.distributions import Bernoulli
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
import torchaudio
from sklearn.model_selection import train_test_split

from config import sample_rate
from utils import collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WaveDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = glob.glob('../waves_yesno/*.wav')

    def __getitem__(self, i):
        infile = self.data[i]
        waveform, sr = torchaudio.load(infile, normalization=True)
        return waveform
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    batch_size = 1
    train_ds = WaveDataset()
    cv_ds = WaveDataset()
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, shuffle=True)
    cv_loader = DataLoader(cv_ds, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
    torch.save(train_loader, 'train_loader.pt')
    torch.save(cv_loader, 'cv_loader.pt')