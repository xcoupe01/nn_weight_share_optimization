#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: configuration of CNN compression 
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

import math
from data.utils.mnist_utils import *
from utils.weight_sharing import *

class CompressConfig:
    """Collection of compression setting used in the CLI compression.
    Look at the source code to learn abot them.
    """
    
    # shared config
    SAVE_EVERY = 1
    VERBOSE = True
    SHARE_ORDER = None #[0, 1, 2, 3, 4]
    RETRAIN_AMOUNT = None #[0, 0, 0, 0, 0]
    PRECISION_REDUCTION = 'f4'
    CLUST_MOD_FOCUS = None #[0, 0, 0, 0, 0]
    CLUST_MOD_SPREAD = None #[0, 0, 0, 0, 0]
    TOP_K_ACC = 1
    DEVICE = 'cpu'
    CLUST_ALG = 'minibatch-kmeans'
    TOP_REPR_SET_INDIV = True
    # optim target
    OPTIM_TARGET = [0.82, 4.0]
    OPTIM_TARGET_LOW_LIMIT = [0.82, 1.0]
    OPTIM_TARGET_LOCK = False
    OPTIM_TARGET_UPDATE_OFFSET = [0.001, 0.1]
    # net settings
    NET_TYPE = 'mobilenet_v2'
    # range optim
    RANGE_OPTIMIZATION = True
    RANGE_OPTIMIZATION_TRESHOLD = 0.825

    # genetic config
    EVOL_SAVE_FILE = './results/GA_save.csv'
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
    PSO_PARTICLE_MAX_VELOCITY = [4 for _ in range(53)]
    PSO_SAVE_FILE = './results/PSO_save.csv'
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
    BH_PARTICLE_MAX_VELOCITY = [4 for _ in range(53)]
    BH_RADIUS = None
    BH_VEL_TRESH = 0.25
    BH_REPR_RAD = True
    BH_SAVE_FILE = './results/BH_save.csv'
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
    RND_SAVE_FILE = './results/RND_save.csv'
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

    # total share config
    TOTAL_SAVE_FILE = f'./results/complete_{NET_TYPE}_share_{PRECISION_REDUCTION[0]}.csv'
    TOTAL_DATA = {
        'num_vals': [],
        'compression': [],
        'accuracy': [],
        'inertia': [],
    }
    TOTAL_DATA_TYPES = {
        'num_vals': 'uint8',
        'compression': 'float32',
        'accuracy': 'float32',
        'inertia': 'float32',
    }

def dump_comp_config() -> object:
    """Generates structure that can be used to create save file of the config, which 
    can be loaded later if needed.

    Returns:
        object: The save object.
    """
    return {
        'target':{
            'point': CompressConfig.OPTIM_TARGET,
            'lock': CompressConfig.OPTIM_TARGET_LOCK,
            'update offset': CompressConfig.OPTIM_TARGET_UPDATE_OFFSET,
            'update limit': CompressConfig.OPTIM_TARGET_LOW_LIMIT,
            'top indiv': CompressConfig.TOP_REPR_SET_INDIV,
        },
        'ws settings': {
            'share order': CompressConfig.SHARE_ORDER,
            'retrain': CompressConfig.RETRAIN_AMOUNT,
            'prec red': CompressConfig.PRECISION_REDUCTION,
            'top k acc': CompressConfig.TOP_K_ACC,
            'modulation': {
                'focus': CompressConfig.CLUST_MOD_FOCUS,
                'spread': CompressConfig.CLUST_MOD_SPREAD,
            },
            'clust alg': CompressConfig.CLUST_ALG,
        },
        'rnd': {},
        'ga': {},
        'pso': {
            'max vel': CompressConfig.PSO_PARTICLE_MAX_VELOCITY,
            'limit vel': CompressConfig.PSO_LIMIT_VELOCITY,
            'limit pos': CompressConfig.PSO_LIMIT_POSITION,
            'inertia': CompressConfig.PSO_INERTIA,
        },
        'bh':{
            'max vel': CompressConfig.BH_PARTICLE_MAX_VELOCITY,
            'limit vel': CompressConfig.BH_LIMIT_VELOCITY,
            'limit pos': CompressConfig.BH_LIMIT_POSITION,
            'inertia': CompressConfig.BH_INERTIA,
            'radius': CompressConfig.BH_RADIUS,
            'repr radius': CompressConfig.BH_REPR_RAD,
            'vel tresh': CompressConfig.BH_VEL_TRESH,
        },
        'net':{
            'type': CompressConfig.NET_TYPE
        },
        'compress space': {
            'optimized': CompressConfig.RANGE_OPTIMIZATION,
            'opt tresh': CompressConfig.RANGE_OPTIMIZATION_TRESHOLD,
        }
    }

def load_comp_config(json:object) -> None:
    """Loads config based on the given json object.

    Args:
        json (object): is the json object to be loaded.
    """
    # target load
    CompressConfig.OPTIM_TARGET = json['target']['point'] 
    CompressConfig.OPTIM_TARGET_LOCK = json['target']['lock']
    CompressConfig.OPTIM_TARGET_UPDATE_OFFSET = json['target']['update offset']
    CompressConfig.OPTIM_TARGET_LOW_LIMIT = json['target']['update limit']
    CompressConfig.TOP_REPR_SET_INDIV = json['target']['top indiv'] if 'top idniv' in json['target'].keys() else False
    #ws load
    CompressConfig.SHARE_ORDER = json['ws settings']['share order']
    CompressConfig.RETRAIN_AMOUNT = json['ws settings']['retrain']
    CompressConfig.PRECISION_REDUCTION = json['ws settings']['prec red'] if json['ws settings']['prec red'] is not None else 'f4' 
    CompressConfig.CLUST_MOD_FOCUS = json['ws settings']['modulation']['focus']
    CompressConfig.CLUST_MOD_SPREAD = json['ws settings']['modulation']['spread']
    CompressConfig.TOP_K_ACC = json['ws settings']['top k acc']
    CompressConfig.CLUST_ALG = json['ws settings']['clust alg']
    #rnd load
    #ga load
    #pso load
    CompressConfig.PSO_PARTICLE_MAX_VELOCITY = json['pso']['max vel']
    CompressConfig.PSO_LIMIT_VELOCITY = json['pso']['limit vel']
    CompressConfig.PSO_LIMIT_POSITION = json['pso']['limit pos']
    CompressConfig.PSO_INERTIA = json['pso']['inertia']
    #bh load
    CompressConfig.BH_PARTICLE_MAX_VELOCITY = json['bh']['max vel']
    CompressConfig.BH_LIMIT_VELOCITY = json['bh']['limit vel']
    CompressConfig.BH_LIMIT_POSITION = json['bh']['limit pos']
    CompressConfig.BH_INERTIA = json['bh']['inertia']
    CompressConfig.BH_RADIUS = json['bh']['radius']
    CompressConfig.BH_REPR_RAD = json['bh']['repr radius']
    CompressConfig.BH_VEL_TRESH = json['bh']['vel tresh']
    #net load
    CompressConfig.NET_TYPE = json['net']['type']
    CompressConfig.RANGE_OPTIMIZATION = json['compress space']['optimized']
    CompressConfig.RANGE_OPTIMIZATION_TRESHOLD = json['compress space']['opt tresh']

def set_save_files_path(folder:str):
    """Sets the filepaths to given folder for all compressions.

    Args:
        folder (str): is the path to the folder where the savefiles will be saved.
    """

    CompressConfig.EVOL_SAVE_FILE = os.path.join(folder, 'GA_save.csv')
    CompressConfig.PSO_SAVE_FILE = os.path.join(folder, 'PSO_save.csv')
    CompressConfig.BH_SAVE_FILE = os.path.join(folder, 'BH_save.csv')
    CompressConfig.RND_SAVE_FILE = os.path.join(folder, 'RND_save.csv')

def fitness_vals_fc(individual, ws_controller:WeightShare) -> list:
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

    prec_reduct_list = [CompressConfig.PRECISION_REDUCTION for _ in ws_controller.model_layers]
    
    # share weigts by particle
    if individual.data is None:
        individual.data = ws_controller.share(repres, CompressConfig.SHARE_ORDER, CompressConfig.RETRAIN_AMOUNT, 
        prec_reduct=prec_reduct_list, mods_focus=CompressConfig.CLUST_MOD_FOCUS, 
        mods_spread=CompressConfig.CLUST_MOD_SPREAD, clust_alg=CompressConfig.CLUST_ALG)
    
    return [individual.data['accuracy'], individual.data['compression']]

def fit_from_vals(data:list[float], targ_vals:list[float]) -> float:
    """Computes fitness from the values that generates the function above.

    Args:
        data (list[float]): The values from the function above (accuracy and compression).
        targ_vals (list[float]): Target by fitness controller.

    Returns:
        float: The outut fitness.
    """

    return 1 / math.sqrt(pow(1 - (data['accuracy']/targ_vals[0]), 2) + pow(1 - (data['compression']/targ_vals[1]), 2))
