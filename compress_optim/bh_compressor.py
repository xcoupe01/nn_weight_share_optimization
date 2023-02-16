import torch.nn as nn
import pandas as pd

from .compressor_config import CompressConfig, fitness_fc
from utils.weight_sharing import *
from utils.pso import PSOController

def logger_fc(pso_cont:PSOController, before_loss:float, save_data:pd.DataFrame=None) -> pd.DataFrame:

    # skip if no place to log data to
    if save_data is None:
        return

    # create new dataframe
    new_data = copy.deepcopy(CompressConfig.BH_DATA)
    for particle in pso_cont.swarm:
        new_data['time'].append(pso_cont.time)
        new_data['fitness'].append(particle.current_fit)
        new_data['position'].append(particle.position)
        new_data['representation'].append(particle.representation)
        new_data['velocity'].append(particle.velocity)
        new_data['accuracy'].append(particle.data['accuracy'])
        new_data['accuracy_loss'].append(before_loss - particle.data['accuracy'])
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

def compression_bh_optim(num_iterations:int, num_particles:int, ranges:list, before_loss:float, 
    model:nn.Module, train_settings:list, ws_controller:WeightShare, net_path:str) -> list:

    save_data = pd.read_csv(CompressConfig.BH_SAVE_FILE).astype(CompressConfig.BH_DATA_TYPES) if \
        os.path.exists(CompressConfig.BH_SAVE_FILE) else \
        pd.DataFrame(CompressConfig.BH_DATA).astype(CompressConfig.BH_DATA_TYPES)

    # initing pso
    lam_fit = lambda individual : fitness_fc(individual, model, train_settings, ws_controller, net_path)
    lam_log = lambda pso_cont, save_data : logger_fc(pso_cont, before_loss, save_data)
    pso = PSOController(num_particles, ranges, CompressConfig.BH_PARTICLE_MAX_VELOCITY, lam_fit, CompressConfig.BH_INERTIA, 
        BH_radius=CompressConfig.BH_RADIUS, BH_vel_tresh=CompressConfig.BH_VEL_TRESH)

    # loading if possible
    if save_data is not None and save_data.size != 0:
        pso.load_from_pd(save_data)

    # compression
    return pso.run(num_iterations, lam_log, save_data, CompressConfig.BH_LIMIT_POSITION, CompressConfig.BH_LIMIT_VELOCITY ,verbose=CompressConfig.VERBOSE)