PATH_PREFIX = '../../'
import sys
sys.path.append(PATH_PREFIX)

#------------------------

import torch
import os
from utils.weight_sharing import *
from data.imagenette import *
from data.utils.download import *
from data.utils.imagenet_utils import *

BATCH_SIZE = 32

#--------------------------------------

net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

dat = ImagenetteDataset(BATCH_SIZE, os.path.join(PATH_PREFIX, 'data/imagenette/'), val_split=0.9)

#----------------------

ws_controller = WeightShare(net, lambda: get_accuracy(net, dat.test_dl, 'cpu', topk=5))
ws_controller.print_layers_info()

#------------------------

ws_controller.share([100 for _ in ws_controller.model_layers], verbose=True, parallel=True)
