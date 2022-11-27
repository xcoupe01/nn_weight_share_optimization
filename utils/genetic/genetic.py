import sys
sys.path.append('../code')
from utils.genetic.genetic_config import EvolConfig
import random
import pandas as pd
import copy
import os

class Individual:
    def __init__(self, possible_values:list, fitness_fc, chromosome:list=None):
        self.possible_values = possible_values
        self.chromosome = []
        self.fitness_fc = fitness_fc
        self.fitness = None
        self.chromosome = chromosome if chromosome is not None else []
        self.data = None # added for tmp info (for example to log cr and AL separetly, in fitness it will be combined)

        if chromosome == None:
            for list in self.possible_values:
                self.chromosome.append(random.choice(list))
            
    def __repr__(self) -> str:
        return f'(Individual: {self.chromosome}, fitness: {self.fitness})'

    def get_fitness(self) -> float:
        """Calculates individuals fitness by a given function

        Returns:
            float: the fitness value of the individual
        """
        if self.fitness is None or self.data is None:
            self.fitness = self.fitness_fc(self)
        return self.fitness

    def set_chromosome(self, chromosome:list) -> None:
        """Sets a chromosome of an individual, concurentrly erases all 
        chromosome based values in the object.

        Args:
            chromosome (list): the chromosome to be set.
        """
        self.chromosome = chromosome
        self.fitness = None
        self.data = None

    def mutate(self, p:float) -> None:
        """Performs mutation with a given probabilyty.
        The mutation algorithm is specified in the `genetic_config.py` file.

        Args:
            p (float): the mutation probability
        """
        EvolConfig.MUTATION_ALGORITHM(self, p)

class GeneticController:
    def __init__(self, individual_type:list, num_individuals:int, fitness_fc):
        self.individual_type = individual_type
        self.population = []
        self.generation = None
        self.jump_start = False

        self.generation = int(0)
        for _ in range(num_individuals):
            self.population.append(Individual(self.individual_type, fitness_fc))

    def __repr__(self) -> str:
        return f'Genetic Controller: generation: {self.generation}, num. population: {len(self.population)}'

    def compute_fitness(self) -> None:
        """Computes fitness of the whole population.
        """
        for indiv in self.population:
            indiv.get_fitness()

    def run(self, num_generations:int, logger_fc=None, save_data:pd.DataFrame=None ,deal_elit=None, verbose:bool=False) -> None:
        """Runs evolution algorithm as inited.

        Args:
            num_generations (int): Number of generations to be evaluated.
            logger_fc (function, optional): funtion that have full acces to controlers data and is meant to log data through
                the evolution - runned in every generation after the fitness scoring stage. If none, nothing happens. Defaults to None.
            save_data (pd.Dataframe, optional): Dataframe to save data to. If defined, it is expected that the logger_fc
                takes as parameter this dataframe and returns the updated one. Defaults to None.
            deal_elit (function, optional): describes what to do with elitist individuals. Defaults to None.
            verbose (bool, optional): If true, some info prints are displayed. Defaults to False.

        Returns:
            list: the best dound solution
                or
            pd.Dataframe: save_data if this parameter is defined at the begining.
        """

        best_chrom = None
        best_fit = None

        for generation in range(0 if self.jump_start else 1, num_generations + 1):
            
            if not self.jump_start:
                # score the population
                self.compute_fitness()
                self.population.sort(key = lambda x: x.fitness, reverse=True)

                # setting best position
                if best_fit is None or best_fit < self.population[0].fitness:
                    best_chrom = copy.deepcopy(self.population[0].chromosome)
                    best_fit = self.population[0].fitness

                # logging
                if logger_fc is not None:
                    if save_data is not None:
                        save_data = logger_fc(self, save_data)
                    else:
                        logger_fc(self)

            self.jump_start = False
            
            if verbose:
                print(f'Generation {generation}/{num_generations} ({self.generation + 1}) evaluated, best fitness {self.population[0].fitness}')

            self.generation += 1
            
            # setting pairs
            num_pairs = int((len(self.population) - EvolConfig.NUM_ELITISTS + 1 ) / 2)
            tmp_pairs = EvolConfig.PAIRING_ALGORITHM(self.population, num_pairs)
            new_chromosomes = []
            
            # breeding
            for pair in tmp_pairs:
                new_chromosomes.extend(EvolConfig.CROSSOVER_ALGORITHM(pair))

            # setting new population
            new_chromosomes = new_chromosomes[:len(self.population)]
            for i in range(EvolConfig.NUM_ELITISTS, len(self.population)):
                new_chromosome_i = i - EvolConfig.NUM_ELITISTS
                self.population[i].set_chromosome(new_chromosomes[new_chromosome_i])

            if deal_elit != None:
                deal_elit(self.population[:EvolConfig.NUM_ELITISTS])
            
            for i in range(0, EvolConfig.NUM_ELITISTS):
                pass

            #mutations
            for indiv in self.population[EvolConfig.NUM_ELITISTS:]:
                indiv.mutate(EvolConfig.MUTATION_PROBABILITY)
        
        if save_data is not None:
            return save_data
        
        return best_chrom

    def load_from_pd(self, df:pd.DataFrame) -> None:
        """Loads the Genetic controller state from a pandas dataframe.

        Args:
            df (pd.DataFrame): is the dataframe to be loaded from.
        """

        self.generation = df['generation'].max()
        self.jump_start = True

        df = df[df['generation'] == self.generation].reset_index()

        for i, row in df.iterrows():
            self.population[i].chromosome = eval(row['chromosome'])
            self.population[i].fitness = row['fitness']

# simple test run to optimize quadratic function
if __name__ == '__main__':

    SAVE_FILE = './results/test/test_GA.csv'

    data_template = {'generation': [], 'fitness': [], 'chromosome': []}
    data_df = pd.read_csv(SAVE_FILE) if os.path.isfile(SAVE_FILE) else pd.DataFrame(data_template)

    def logger_fc(controller):
        global data_df

        new_data = copy.deepcopy(data_template)
        for indiv in controller.population:

            new_data['generation'].append(controller.generation)
            new_data['fitness'].append(indiv.fitness)
            new_data['chromosome'].append(indiv.chromosome)

        data_df = data_df.append(pd.DataFrame(new_data), ignore_index=True)

    fitness_fc = lambda p: - pow(p.chromosome[0], 2) - pow(p.chromosome[1], 2)

    controler = GeneticController([range(-100, 100), range(-100, 100)], 5, fitness_fc)

    if len(data_df.index) > 0:
        controler.load_from_pd(data_df)

    print(controler.run(10, logger_fc, verbose=True))

    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    data_df.to_csv(SAVE_FILE, index=False)
