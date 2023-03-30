import pandas as pd

from utils.rnd import RandomController
from utils.weight_sharing import *
from utils.fitness_controller import FitnessController
from .compressor_config import CompressConfig

def logger_fc(rnd_controler:RandomController, before_acc:float, save_data:pd.DataFrame=None) -> pd.DataFrame:
    """Logger function for PSO search

    Args:
        rnd_controller (RandomController): Is the Random Search controller which data are goind to be logged.
        before_acc (float): Before accuracy of the net to compute accuracy loss.
        save_data (pd.DataFrame, optional): Is the dataframe where the data is going to be saved. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with the saved data.
    """

    # skip if no place to log data to
    if save_data is None:
        return

    # create new dataframe
    new_data = copy.deepcopy(CompressConfig.RND_DATA)
    indiv = rnd_controler.current_indiv
    new_data['representation'].append(indiv.representation)
    new_data['accuracy'].append(indiv.data['accuracy'])
    new_data['accuracy_loss'].append(before_acc - indiv.data['accuracy'])
    new_data['compression'].append(indiv.data['compression'])
    new_data['share_t'].append(indiv.data['times']['share'])
    new_data['train_t'].append(indiv.data['times']['train'])
    new_data['acc_t'].append(indiv.data['times']['test'])

    # saving progress
    save_data = save_data.append(pd.DataFrame(new_data).astype(CompressConfig.RND_DATA_TYPES))
    save_data.reset_index(drop=True, inplace=True)
    save_data.to_csv(CompressConfig.RND_SAVE_FILE, index=False)

    return save_data

def compression_random_optim(num_individuals:int, ranges:list, before_loss:float, fit_controller:FitnessController) -> list:
    """Random Search compression impelentation.

    Args:
        num_individuals (int): Number of the tested individuals.
        ranges (list): The individual representation ranges.
        before_loss (float): Before loss of the network to compute accuracy loss.
        fit_controller (FitnessController): Fitness controler for the optimization.

    Returns:
        list: The best found solution.
    """

    save_data = pd.read_csv(CompressConfig.RND_SAVE_FILE).astype(CompressConfig.RND_DATA_TYPES) \
        if os.path.exists(CompressConfig.RND_SAVE_FILE) else \
        pd.DataFrame(CompressConfig.RND_DATA).astype(CompressConfig.RND_DATA_TYPES)

    # init random
    lam_log = lambda rnd_cont, save_data : logger_fc(rnd_cont, before_loss, save_data)
    
    random = RandomController(ranges, fit_controller)

    # load if possible
    if save_data is not None and save_data.size != 0:
        random.load_from_pd(save_data)

    # compression
    return random.run(num_individuals, lam_log, save_data, CompressConfig.VERBOSE)
