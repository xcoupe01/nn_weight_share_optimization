#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: CNN with imagenet range optimization implementation 
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

import torch
import argparse

from data.imagenette import ImagenetteDataset
from data.utils.imagenet_utils import *
from utils.weight_sharing import *

# net train params - before compression
BATCH_SIZE = 32
NET_REPO = 'pytorch/vision:v0.10.0'
# dataset settings
DATA_PATH = './data/imagenette'

DEVICE = 'cpu'
NET_TYPE = 'mobilenet_v2'
CLUST_ALG = 'minibatch-kmeans'

parser = argparse.ArgumentParser(prog='net_range_opt.py', description='Range optimization for WS optimization')
parser.add_argument('-lo', '--low', type=int, default=1, help='range max')
parser.add_argument('-up', '--upper', type=int, default=10, help='range low')
parser.add_argument('-fn', '--filenumber', type=int, default=1, help='file number for splitting')
parser.add_argument('-pr', '--precision', choices=['f4', 'f2', 'f1'], default='f4', help='precision reduction')

args = parser.parse_args()

dataset = ImagenetteDataset(BATCH_SIZE, DATA_PATH, val_split=0.3)
model = torch.hub.load(NET_REPO, NET_TYPE, pretrained=True)

# defining model hooks
lam_opt = None
lam_train = None
lam_test = lambda : get_accuracy(model, dataset.test_dl, DEVICE, topk=1)

# initing weightsharing
print('initing weight sharing')
ws_controller = WeightShare(model, lam_test, lam_opt, lam_train)
ws_controller.set_reset()

# creating and optimizing search ranges
search_ranges = [range(args.low, args.upper) for _ in ws_controller.model_layers]

# range optimization
lam_test_inp = lambda _ : lam_test()
print('range optimization')
ws_controller.get_optimized_layer_ranges(
    search_ranges, 
    lam_test_inp, 
    0.1, 
    savefile=f'./models/mobilenet_v2/saves/mobilenet_v2_layer_perf_{args.precision}_full-{args.filenumber}.csv',
    prec_rtype=args.precision,
    clust_alg = CLUST_ALG
)