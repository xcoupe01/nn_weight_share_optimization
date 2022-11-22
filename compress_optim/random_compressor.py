import torch.nn as nn
import pandas as pd

from utils.rnd import RandomController
from utils.weight_sharing import *
from .compressor_config import CompressConfig, fitness_fc

def logger_fc(rnd_controler:RandomController, before_loss:float, save_data:pd.DataFrame=None) -> pd.DataFrame:
    
    # skip if no place to log data to
    if save_data is None:
        return

    # create new dataframe
    new_data = copy.deepcopy(CompressConfig.RND_DATA)
    indiv = rnd_controler.current_indiv
    new_data['fitness'].append(indiv.fitness)
    new_data['representation'].append(indiv.representation)
    new_data['accuracy'].append(indiv.data['accuracy'])
    new_data['accuracy_loss'].append(before_loss - indiv.data['accuracy'])
    new_data['compression'].append(indiv.data['compression'])
    new_data['share_t'].append(indiv.data['times']['share'])
    new_data['train_t'].append(indiv.data['times']['train'])
    new_data['acc_t'].append(indiv.data['times']['test'])

    # saving progress
    save_data = save_data.append(pd.DataFrame(new_data).astype(CompressConfig.RND_DATA_TYPES))
    save_data.reset_index(drop=True, inplace=True)
    save_data.to_csv(CompressConfig.RND_SAVE_FILE, index=False)

    return save_data

def compression_random_optim(num_individuals:int, ranges:list, before_loss:float, 
    model:nn.Module, train_settings:list, ws_controller:WeightShare, net_path:str) -> list:

    save_data = pd.read_csv(CompressConfig.RND_SAVE_FILE).astype(CompressConfig.RND_DATA_TYPES) \
        if os.path.exists(CompressConfig.RND_SAVE_FILE) else \
        pd.DataFrame(CompressConfig.RND_DATA).astype(CompressConfig.RND_DATA_TYPES)

    # init random
    lam_fit = lambda individual : fitness_fc(individual, model, train_settings, ws_controller, net_path)
    lam_log = lambda rnd_cont, save_data : logger_fc(rnd_cont, before_loss, save_data)
    random = RandomController(ranges, lam_fit)

    # compression
    return random.run(num_individuals, lam_log, save_data, CompressConfig.VERBOSE)