import random
from utils.genetic.genetic_config import EvolConfig
import pandas as pd

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

    def run_evolution(self, num_generations:int, logger_fc=None, deal_elit=None, verbose:bool=False) -> None:
        """Runs evolution algorithm as inited.

        Args:
            num_generations (int): Number of generations to be evaluated.
            logger_func (function, optionas): funtion that have full acces to controlers data and is meant to log data through
            the evolution - runned in every generation after the fitness scoring stage. If none, nothing happens. Defaults to None.
            deal_elit (funtion, optional): TODO - wite docstring
            verbose (bool, optional): If true, some info prints are displayed. Defaults to False.
        """

        for generation in range(0 if self.jump_start else 1, num_generations + 1):
            
            if not self.jump_start:
                # score the population
                self.compute_fitness()
                self.population.sort(key = lambda x: x.fitness, reverse=True)
                if logger_fc is not None:
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

    def load_from_pd(self, df:pd.DataFrame) -> None:

        self.generation = df['generation'].max()
        self.jump_start = True

        df = df[df['generation'] == self.generation].reset_index()

        for i, row in df.iterrows():
            self.population[i].chromosome = eval(row['chromosome'])
            self.population[i].fitness = row['fitness']
            
                

#------------------------------------ test temp ------------------------------------

def tmp_fit(x):
    return random.random()

def tmp_logger(x):
    print(x.population)

if __name__ == '__main__':
    settings = [[1, 40], [2, 40], [2, 40], [2, 40], [2, 40]]
    tmp = GeneticController(settings, 5, tmp_fit)

    tmp_cmp_chrom = tmp.population[0].chromosome
    #tmp_cmp_chrom = [1, 2, 3, 4, 5]

    tmp.run_evolution(3, tmp_logger, verbose=True)
    

    """print(tmp.data)
    print(tmp)

    print(tmp.data.dtypes)
    print(tmp_cmp_chrom)"""

    #print(tmp_cmp_chrom in tmp.data['chromosome'].values)
    #print(tmp_cmp_chrom, tmp.data['chromosome'].values)

    #print([1, 2] in [[1, 2], [1, 3]])

    #print(tmp.data.isin(str(tmp_cmp_chrom)))
    #print(tmp.data[tmp.data['chromosome'].apply(lambda x : x == tmp_cmp_chrom)].iloc[0]['fitness'])