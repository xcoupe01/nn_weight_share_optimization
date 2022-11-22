import torch.nn as nn
import pandas as pd

from .compressor_config import CompressConfig, fitness_fc
from utils.weight_sharing import *
from utils.genetic import GeneticController

def logger_fc(gen_cont:GeneticController, before_loss:float, save_data:pd.DataFrame=None) -> None:

    # skip if no place to log data to
    if save_data is None:
        return None

    # create new dataframe
    new_data = copy.deepcopy(CompressConfig.EVOL_DATA)
    for indiv in gen_cont.population:
        new_data['generation'].append(gen_cont.generation)
        new_data['fitness'].append(indiv.fitness)
        new_data['chromosome'].append(indiv.chromosome)
        new_data['accuracy'].append(indiv.data['accuracy'])
        new_data['accuracy_loss'].append(before_loss - indiv.data['accuracy'])
        new_data['compression'].append(indiv.data['compression'])
        new_data['share_t'].append(indiv.data['times']['share'])
        new_data['train_t'].append(indiv.data['times']['train'])
        new_data['acc_t'].append(indiv.data['times']['test'])

    # saving progress
    save_data = save_data.append(pd.DataFrame(new_data).astype(CompressConfig.EVOL_DATA_TYPES))
    if gen_cont.generation % CompressConfig.SAVE_EVERY == CompressConfig.SAVE_EVERY - 1:
        save_data.reset_index(drop=True, inplace=True)
        save_data.to_csv(CompressConfig.EVOL_SAVE_FILE, index=False)

    return save_data

def deal_elit(population:list) -> None:
    # zeroing the times
    for individual in population:
        if individual.data is not None:
            individual.data['times'] = {
                'share': 0,
                'train': 0,
                'test': 0
            }

def compression_genetic_optim(num_generation:int, num_population:int, ranges:list, before_loss:float, 
    model:nn.Module, train_settings:list, ws_controller:WeightShare, net_path:str) -> pd.DataFrame:

    # init data
    save_data = pd.read_csv(CompressConfig.EVOL_SAVE_FILE).astype(CompressConfig.EVOL_DATA_TYPES) \
        if os.path.exists(CompressConfig.EVOL_SAVE_FILE) else \
        pd.DataFrame(CompressConfig.EVOL_DATA).astype(CompressConfig.EVOL_DATA_TYPES)

    # init genetic controller
    lam_fit = lambda individual : fitness_fc(individual, model, train_settings, ws_controller, net_path)
    lam_log = lambda gen_cont, save_data : logger_fc(gen_cont, before_loss, save_data)
    genetic = GeneticController(ranges, num_population, lam_fit)

    # load if possible
    if save_data is not None and save_data.size != 0:
        genetic.load_from_pd(save_data)

    # compression
    return genetic.run(num_generation, lam_log, save_data, deal_elit, CompressConfig.VERBOSE)
