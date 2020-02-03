import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
import json
import os
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from collections import defaultdict
from config import sample_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_fn(data, device=device):
    data = torch.stack(data)
    x = MelSpectrogram(sample_rate=sample_rate)(data)
    x = AmplitudeToDB(stype='power', top_db=80)(x)
    maxval = x.max()
    minval = x.min()
    x = (x-minval)/(maxval - minval)
    return x

def fmt_inps(info):
    t1 = info['t1']
    t1 = torch.FloatTensor(t1).squeeze()
    t2 = torch.FloatTensor(np.arange(t1[-1]+1, t1[-1]+21, 1)).squeeze()
    obsvd_data = torch.FloatTensor(info['x']).view(1,-1,1)
    obsvd_mask = torch.ones(1, len(t1), 1)
    return t1, t2, obsvd_data, obsvd_mask