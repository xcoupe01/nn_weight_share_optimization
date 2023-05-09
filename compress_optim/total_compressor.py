#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Implementation of total CNN compression 
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

import pandas as pd
import copy

from utils.weight_sharing import *
from .compressor_config import CompressConfig
from utils.weight_sharing import WeightShare

def compression_total(range:list, before_acc:float, ws_controller:WeightShare):
    """Shares the whole nets weights into given number of clusters - model wise weight sharing.
    This is computed for given list of numbers of clusters and the corresponding net performance
    is logged and stored.

    Args:
        range (list): Is the list of cluster numbers to be tried for model wise weight sharing.
        before_acc (float): Is the before share model accuracy.
        ws_controller (WeightShare): Is the weight sharing controler for the model.
    """

    df = copy.deepcopy(CompressConfig.TOTAL_DATA)

    # generating data
    for value in range:
        perf = ws_controller.share_total(value, prec_rtype=CompressConfig.PRECISION_REDUCTION[0],
            clust_alg=CompressConfig.CLUST_ALG)
        ws_controller.reset()
        df['num_vals'].append(value)
        df['compression'].append(perf['compression'])
        df['accuracy'].append(perf['accuracy'])
        df['inertia'].append(perf['inertia'])

    df = pd.DataFrame(df)
    df['accuracy_loss'] = before_acc - df['accuracy']
    df.to_csv(CompressConfig.TOTAL_SAVE_FILE)