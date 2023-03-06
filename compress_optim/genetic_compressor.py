import pandas as pd

from utils.genetic import GeneticController
from utils.weight_sharing import *
from utils.fitness_controller import FitnessController
from .compressor_config import CompressConfig

def logger_fc(gen_cont:GeneticController, before_loss:float, save_data:pd.DataFrame=None) -> None:
    """Logger function for Genetic search

    Args:
        gen_cont (GeneticController): Is the Genetic controller which data are goind to be logged.
        before_loss (float): Before loss of the net to compute acculacy loss.
        save_data (pd.DataFrame, optional): Is the dataframe where the data is going to be saved. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with the saved data.
    """

    # skip if no place to log data to
    if save_data is None:
        return None

    # create new dataframe
    new_data = copy.deepcopy(CompressConfig.EVOL_DATA)
    for indiv in gen_cont.population:
        new_data['generation'].append(gen_cont.generation)
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

def compression_genetic_optim(num_generation:int, num_population:int, ranges:list, before_loss:float, fit_controller:FitnessController) -> pd.DataFrame:
    """Genetic compression impelentation.

    Args:
        num_generation (int): Number of generations for the Genetic optimization.
        num_population (int): Number of individuals in population in the Genetic search.
        ranges (list): The chromosome representation ranges.
        before_loss (float): Before loss of the network to compute accuracy loss.
        fit_controller (FitnessController): Fitness controler for the optimization.

    Returns:
        list: The best found solution.
    """

    # init data
    save_data = pd.read_csv(CompressConfig.EVOL_SAVE_FILE).astype(CompressConfig.EVOL_DATA_TYPES) \
        if os.path.exists(CompressConfig.EVOL_SAVE_FILE) else \
        pd.DataFrame(CompressConfig.EVOL_DATA).astype(CompressConfig.EVOL_DATA_TYPES)

    # init genetic controller
    lam_log = lambda gen_cont, save_data : logger_fc(gen_cont, before_loss, save_data)
    
    genetic = GeneticController(ranges, num_population, fit_controller)

    # load if possible
    if save_data is not None and save_data.size != 0:
        genetic.load_from_pd(save_data)

    # compression
    return genetic.run(num_generation, lam_log, save_data, deal_elit, CompressConfig.VERBOSE)
