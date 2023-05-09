#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: CNN with imagenet finetuning implementation 
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

import torch
import argparse
import os
import numpy as np

from data.imagenette import ImagenetteDataset
from data.utils.imagenet_utils import *
from utils.weight_sharing import *

# net params
BATCH_SIZE = 32
NET_REPO = 'pytorch/vision:v0.10.0'
DEVICE = 'cpu'
LAYER_CLUSTERS = [98, 95, 77, 67, 115, 106, 55, 98, 110, 55, 52, 44, 113, 61, 50, 19, 40, 107, 87, 10, 60, 22, 95, 31, 12, 51, 37, 102, 45, 31, 65, 115, 62, 13, 43, 112, 101, 62, 72, 59, 76, 89, 29, 38, 41, 112, 23, 115, 44, 13, 106, 79, 86]
NET_TYPE = 'mobilenet_v2'
CLUST_ALG = 'minibatch-kmeans'

# dataset settings
DATA_PATH = './data/imagenette'

parser = argparse.ArgumentParser(prog='net_range_opt.py', description='Finetunig metod')
parser.add_argument('-lo', '--low', type=int, default=0, help='focus min')
parser.add_argument('-up', '--upper', type=int, default=10, help='focus max')
parser.add_argument('-st', '--step', type=float, default=0.2, help='focus step')
parser.add_argument('-fp', '--folder_path', type=str, default='./results/finetuning', help='output file save path')
parser.add_argument('-pr', '--precision', choices=['f4', 'f2', 'f1'], default='f4', help='precision reduction')

args = parser.parse_args()

if not os.path.isdir(args.folder_path):
    os.makedirs(args.folder_path)

dataset = ImagenetteDataset(BATCH_SIZE, DATA_PATH, val_split=0.3)
model = torch.hub.load(NET_REPO, NET_TYPE, pretrained=True)

# defining model hooks
lam_opt = None
lam_train = None
lam_test = lambda : get_accuracy(model, dataset.test_dl, DEVICE, topk=1)

# initing weightsharing
print('initing weight sharing')
ws_controller = WeightShare(model, lam_test, lam_opt, lam_train)

# range optimization
print('fine-tuning')
ws_controller.finetuned_mod(
    layer_clusters = LAYER_CLUSTERS,
    mods_focus = np.arange(args.low, args.upper, args.step),
    mods_spread = [2 for _ in ws_controller.model_layers],
    prec_reduct = [args.precision for _ in ws_controller.model_layers],
    savefile = os.path.join(args.folder_path, f'{NET_TYPE}_{args.precision}.csv'),
    clust_alg = CLUST_ALG
)