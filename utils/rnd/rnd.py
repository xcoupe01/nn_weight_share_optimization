import sys
sys.path.append('../code')
import random
import pandas as pd
import os
import copy
from utils.fitness_controller import FitnessController

class Individual:
    def __init__(self, possible_values:list) -> None:
        """Inits the individual for random search.

        Args:
            possible_values (list): Defines the possible position ranges. Each index of list corresponds to its dimension.
                Each item on index is expected to be range of possible values in given dimension.
        """
        self.representation = []
        self.fitness = None
        self.data = None

        for dimension in possible_values:
            self.representation.append(random.choice(dimension))

    def __repr__(self) -> str:
        """String representation of the Individual class.

        Returns:
            str: the string representation of the individual class.
        """
        return f'Individual ({self.representation}) fit:{self.fitness}'

    def compute_fitness(self, fit_func) -> float:
        """Computes the fintess value of this individual by a given fitness function.

        Args:
            fit_func (function): The which calculates the fitness based on the individuals data.

        Returns:
            float: The fitness value of the individual.
        """

        self.fitness = fit_func(self.data)
        return self.fitness

class RandomController:
    def __init__(self, individual_type:list, fitness_cont:FitnessController) -> None:
        """Inits the Random Search Controller.

        Args:
            individual_type (list): Defines the possible position ranges. Each index of list corresponds to its dimension.
                Each item on index is expected to be range of possible values in given dimension.
            fitness_cont (FitnessController): is a fitness controler object. 
        """
        self.individual_type = individual_type
        self.fitness_cont = fitness_cont
        self.iteration = 0
        self.current_indiv = None
        self.best_indiv = None

    def __repr__(self) -> str:
        """Defines the string representation of the RandomController class.

        Returns:
            str: the representation string.
        """
        return f'RandomController iter:{self.iteration} best:{self.best_indiv}'

    def load_from_pd(self, dataframe:pd.DataFrame, verbose:bool = False) -> None:
        """Loads the random controller state from a pandas dataframe.
        The dataframe must contain fitness and representation columns.

        Created for the pourpouses of WS optimalization!

        Args:
            dataframe (pd.DataFrame): is the dataframe to be loaded from.
            verbose (bool, optional): If true, prints out informations. Defaults to False.
        """
        self.iteration = len(dataframe.index) - 1
        self.best_indiv = Individual(self.individual_type)
        self.fitness_cont.fit_from_df(dataframe, verbose)
        best_row = dataframe.loc[dataframe['fitness'].idxmax()]
        self.best_indiv.fitness = best_row['fitness']
        self.best_indiv.representation = eval(best_row['representation'])
    
    def run(self, num_individuals:int, logger_fc=None, save_data:pd.DataFrame=None, verbose:bool=False) -> list:
        """Runs the random search in the goal to reach the num of individuals.
        All the generated solutions can be logged by the given logger function.

        Args:
            num_individuals (int): number of individuals to be evaluated
            logger_fc (function, optional): the logging function (as an attribute, this object is given). Defaults to None.
            save_data (pd.Dataframe, optional) :is an optional pandas datagrame to log data to. If defined,
                it is expected that logger_fc takes this dataframe and the function returns the updated one.
                Also if defined. the function returns the saves, otherwise the best found solution is returned.
            verbose (bool, optional): If true, information about the progress is printed. Defaults to False.

        Returns:
            list: representation of the best individual found. If save_file is defined, the save data are returned.
                or
            pd.Dataframe: save_data if this parameter is defined at the begining.
        """
        for iteration in range(1, num_individuals + 1):
            
            # create new individual
            self.current_indiv = Individual(self.individual_type)

            # compute fitness
            self.fitness_cont.compute_fit([self.current_indiv], verbose=verbose)

            if self.best_indiv is None or self.current_indiv.fitness > self.best_indiv.fitness:
                self.best_indiv = self.current_indiv

            # log
            if logger_fc is not None:
                if save_data is not None:
                    save_data = logger_fc(self, save_data)
                else:
                    logger_fc(self)

            self.iteration += 1

            if verbose:
                print(f'Individual {iteration}/{num_individuals} ({self.iteration}) evaluated {self.current_indiv}')

        if save_data is not None:
            return save_data
        
        return self.best_indiv.representation

# simple test run to optimize quadratic function
if __name__ == '__main__':

    # load save file if possible
    SAVE_FILE = './results/test/test_RND.csv'
    data_template = {'representation': [], 'fitness': []}
    data_df = pd.read_csv(SAVE_FILE) if os.path.isfile(SAVE_FILE) else pd.DataFrame(data_template)

    # logger function
    def logger_fc(idnividual):
        global data_df

        new_data = copy.deepcopy(data_template)
        new_data['representation'].append(idnividual.current_indiv.representation)
        new_data['fitness'].append(idnividual.current_indiv.fitness)

        data_df = data_df.append(pd.DataFrame(new_data), ignore_index=True)
    
    # init fit
    get_fit_vals = lambda p: [- pow(p.representation[0], 2), - pow(p.representation[1], 2)]
    def fit_from_vals(p, fv, mv): p.fitness = fv[0] + fv[1]
    fitness_cont = FitnessController([0, 0], get_fit_vals, fit_from_vals)

    # init controllers
    controler = RandomController([range(-100, 100), range(-100, 100)], fitness_cont)
    
    # load the controler with data - probably wont work :(
    if len(data_df.index) > 0:
        controler.load_from_pd(data_df)

    # run optimization
    print(controler.run(10, logger_fc, verbose=True))

    # save data
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    data_df.to_csv(SAVE_FILE, index=False)