import pandas as pd
import torch.nn as nn
import os
import math
from utils.train import *
from utils.weight_sharing import *

class CompressConfig:
    # shared config
    SAVE_EVERY = 1
    VERBOSE = True
    SHARE_ORDER = [0, 1, 2, 3, 4]
    RETRAIN_AMOUNT = [0, 0, 0, 0, 0]
    PRECISION_REDUCTION = ['f4', 'f4', 'f4', 'f4', 'f4']

    # genetic config
    EVOL_SAVE_FILE = './results/lenet_GA_save.csv'
    EVOL_DATA = {
        'generation': [],
        'fitness': [],
        'chromosome': [],
        'accuracy': [],
        'accuracy_loss': [],
        'compression': [],
        'share_t': [],
        'train_t': [],
        'acc_t': []
    }
    EVOL_DATA_TYPES = {
        'generation' : 'uint8',
        'fitness': 'float32',
        'accuracy': 'float32',
        'accuracy_loss': 'float32',
        'compression': 'float32',
        'share_t': 'float32',
        'train_t': 'float32',
        'acc_t': 'float32'
    }

    # pso config
    PSO_PARTICLE_MAX_VELOCITY = [2.5 for _ in range(5)]
    PSO_SAVE_FILE = './results/lenet_PSO_save.csv'
    PSO_LIMIT_VELOCITY = True
    PSO_LIMIT_POSITION = True
    PSO_DATA = {
        'time': [],
        'fitness': [],
        'position': [],
        'representation': [],
        'velocity': [],
        'accuracy': [],
        'accuracy_loss': [],
        'compression': [],
        'share_t': [],
        'train_t': [],
        'acc_t': []
    }
    PSO_DATA_TYPES = {
        'time' : 'uint8',
        'fitness': 'float32',
        'accuracy': 'float32',
        'accuracy_loss': 'float32',
        'compression': 'float32',
        'share_t': 'float32',
        'train_t': 'float32',
        'acc_t': 'float32'
    }

    # random config
    RND_SAVE_FILE = './results/lenet_RND_save.csv'
    RND_DATA = {
        'fitness': [],
        'representation': [],
        'accuracy': [],
        'accuracy_loss': [],
        'compression': [],
        'share_t': [],
        'train_t': [],
        'acc_t': []
    }
    RND_DATA_TYPES = {
        'fitness': 'float32',
        'accuracy': 'float32',
        'accuracy_loss': 'float32',
        'compression': 'float32',
        'share_t': 'float32',
        'train_t': 'float32',
        'acc_t': 'float32'
    }

def fitness_fc(individual, model:nn.Module, train_settings:list, ws_controller:WeightShare, net_path:str) -> float:
    # reset the net
    get_trained(model, net_path, train_settings)
    ws_controller.reset()
    
    # get representation
    repres = individual.chromosome if hasattr(individual, 'chromosome') else individual.representation

    # share weigts by chromosome
    individual.data = ws_controller.share(repres, CompressConfig.SHARE_ORDER, CompressConfig.RETRAIN_AMOUNT, CompressConfig.PRECISION_REDUCTION)

    # compute fitness
    if individual.data['accuracy'] <= 0.95:
        return individual.data['accuracy']

    return 1 / math.sqrt(pow(1 - ((individual.data['accuracy'] - 0.9) * (1/0.1)), 2) + pow(1 - (individual.data['compression']/14), 2))