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
    RETRAIN_AMOUNT = None #[0, 0, 0, 0, 0]
    PRECISION_REDUCTION = None #['f4', 'f4', 'f4', 'f4', 'f4']
    CLUST_MOD_FOCUS = None #[0, 0, 0, 0, 0]
    CLUST_MOD_SPREAD = None #[0, 0, 0, 0, 0]
    DEVICE = 'cpu'
    OPTIM_TARGET = [1.0, 12.0]
    OPTIM_TARGET_LOCK = False
    OPTIM_TARGET_UPDATE_OFFSET = 1

    # genetic config
    EVOL_SAVE_FILE = './results/lenet_GA_save.csv'
    EVOL_DATA = {
        'generation': [],
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
        'accuracy': 'float32',
        'accuracy_loss': 'float32',
        'compression': 'float32',
        'share_t': 'float32',
        'train_t': 'float32',
        'acc_t': 'float32'
    }

    # pso config
    PSO_PARTICLE_MAX_VELOCITY = [4 for _ in range(5)]
    PSO_SAVE_FILE = './results/lenet_PSO_save.csv'
    PSO_LIMIT_VELOCITY = True
    PSO_LIMIT_POSITION = True
    PSO_INERTIA = 0.8
    PSO_DATA = {
        'time': [],
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
        'accuracy': 'float32',
        'accuracy_loss': 'float32',
        'compression': 'float32',
        'share_t': 'float32',
        'train_t': 'float32',
        'acc_t': 'float32'
    }

    # bh config
    BH_PARTICLE_MAX_VELOCITY = [4 for _ in range(5)]
    BH_RADIUS = 1
    BH_VEL_TRESH = 1
    BH_SAVE_FILE = './results/lenet_BH_save.csv'
    BH_LIMIT_VELOCITY = True
    BH_LIMIT_POSITION = True
    BH_INERTIA = 0.8
    BH_DATA = {
        'time': [],
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
    BH_DATA_TYPES = {
        'time' : 'uint8',
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
        'representation': [],
        'accuracy': [],
        'accuracy_loss': [],
        'compression': [],
        'share_t': [],
        'train_t': [],
        'acc_t': []
    }
    RND_DATA_TYPES = {
        'accuracy': 'float32',
        'accuracy_loss': 'float32',
        'compression': 'float32',
        'share_t': 'float32',
        'train_t': 'float32',
        'acc_t': 'float32'
    }

def fitness_vals_fc(individual, ws_controller:WeightShare):
    """Base values for computing fitness getter.

    Args:
        individual ([Individual, Particle]): is the infdividual or paticle to claclulate the values by.
        ws_controller (WeightShare): Is the WS controller.

    Returns:
        list: list of the base values (accuracy and compression)
    """

    # reset the net
    ws_controller.reset()

    # get representation
    repres = individual.chromosome if hasattr(individual, 'chromosome') else individual.representation
    
    # share weigts by particle
    if individual.data is None:
        individual.data = ws_controller.share(repres, CompressConfig.SHARE_ORDER, CompressConfig.RETRAIN_AMOUNT, 
        prec_reduct=CompressConfig.PRECISION_REDUCTION, mods_focus=CompressConfig.CLUST_MOD_FOCUS, 
        mods_spread=CompressConfig.CLUST_MOD_SPREAD)
    
    return [individual.data['accuracy'], individual.data['compression']]

def fit_from_vals(data:list[float], targ_vals:list[float]):
    """Computes fitness from the values that generates the function above.

    Args:
        data (list[float]): The values from the function above (accuracy and compression).
        targ_vals (list[float]): Target by fitness controller.

    Returns:
        float: The outut fitness.
    """

    # compute fitness
    if data['accuracy'] <= 0.95:
        return data['accuracy']

    return 1 / math.sqrt(pow(1 - ((data['accuracy'] - 0.9) * (1/0.1)), 2) + pow(1 - (data['compression']/targ_vals[1]), 2))
