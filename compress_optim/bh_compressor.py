#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Implementation of Blachole CNN compression 
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

import pandas as pd

from utils.pso import PSOController
from utils.weight_sharing import *
from utils.fitness_controller import FitnessController
from .compressor_config import CompressConfig

def logger_fc(pso_cont:PSOController, before_acc:float, save_data:pd.DataFrame=None) -> pd.DataFrame:
    """Logger function for Black Hole search

    Args:
        pso_cont (PSOController): Is the PSO controller which data are goind to be logged.
        before_acc (float): Before accuracy of the net to compute accuracy loss.
        save_data (pd.DataFrame, optional): Is the dataframe where the data is going to be saved. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with the saved data.
    """

    # skip if no place to log data to
    if save_data is None:
        return

    # create new dataframe
    new_data = copy.deepcopy(CompressConfig.BH_DATA)
    for particle in pso_cont.swarm:
        new_data['time'].append(pso_cont.time)
        new_data['position'].append(particle.position)
        new_data['representation'].append(particle.representation)
        new_data['velocity'].append(particle.velocity)
        new_data['accuracy'].append(particle.data['accuracy'])
        new_data['accuracy_loss'].append(before_acc - particle.data['accuracy'])
        new_data['compression'].append(particle.data['compression'])
        new_data['share_t'].append(particle.data['times']['share'])
        new_data['train_t'].append(particle.data['times']['train'])
        new_data['acc_t'].append(particle.data['times']['test'])

    # saving progress
    save_data = save_data.append(pd.DataFrame(new_data).astype(CompressConfig.BH_DATA_TYPES))
    if pso_cont.time % CompressConfig.SAVE_EVERY == CompressConfig.SAVE_EVERY - 1:
        save_data.reset_index(drop=True, inplace=True)
        save_data.to_csv(CompressConfig.BH_SAVE_FILE, index=False)

    return save_data

def compression_bh_optim(num_iterations:int, num_particles:int, ranges:list, before_loss:float, fit_controller:FitnessController) -> list:
    """Black Hole compression impelentation.

    Args:
        num_iterations (int): Number of iterations for the Black Hole optimization.
        num_particles (int): Number of particles in the Black Hole search.
        ranges (list): The particle representation ranges.
        before_loss (float): Before loss of the network to compute accuracy loss.
        fit_controller (FitnessController): Fitness controler for the optimization.

    Returns:
        list: The best found solution.
    """

    save_data = pd.read_csv(CompressConfig.BH_SAVE_FILE).astype(CompressConfig.BH_DATA_TYPES) if \
        os.path.exists(CompressConfig.BH_SAVE_FILE) else \
        pd.DataFrame(CompressConfig.BH_DATA).astype(CompressConfig.BH_DATA_TYPES)

    # initing pso
    lam_log = lambda pso_cont, save_data : logger_fc(pso_cont, before_loss, save_data)
    pso = PSOController(num_particles, ranges, CompressConfig.PSO_PARTICLE_MAX_VELOCITY, CompressConfig.PSO_INERTIA, fit_controller, 
        BH_radius=CompressConfig.BH_RADIUS, BH_repr_rad=CompressConfig.BH_REPR_RAD, BH_vel_tresh=CompressConfig.BH_VEL_TRESH)

    # loading if possible
    if save_data is not None and save_data.size != 0:
        pso.load_from_pd(save_data)
    elif CompressConfig.TOP_REPR_SET_INDIV:
        # setting top repr indiv if needed
        pso.swarm[0].set_pos([float(len(rng)) for rng in ranges])

    # compression
    return pso.run(num_iterations, lam_log, save_data, CompressConfig.PSO_LIMIT_POSITION, CompressConfig.PSO_LIMIT_VELOCITY ,verbose=CompressConfig.VERBOSE)
