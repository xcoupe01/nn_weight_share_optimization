import random

class Individual:
    def __init__(self, possible_values:list, fitness_fc) -> None:
        """Inits the individual for random search.

        Args:
            possible_values (list): Defines the possible position ranges. Each index of list corresponds to its dimension.
            Each item on index is expected to be range of possible values in given dimension.
            fitness_fc (function): Defines the fitness function of the individual. The function sould compute one float number, its
            input is the individual.
        """
        self.representation = []
        self.fitness_fc = fitness_fc
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

    def get_fitness(self) -> float:
        """Computes fitness based on the init.

        Returns:
            float: the fitness value of this individual.
        """
        if self.fitness is None:
            self.fitness = self.fitness_fc(self)
        return self.fitness

class RandomController:
    def __init__(self, individual_type:list, fitness_fc) -> None:
        """Inits the Random Search Controller.

        Args:
            individual_type (list): Defines the possible position ranges. Each index of list corresponds to its dimension.
            Each item on index is expected to be range of possible values in given dimension.
            fitness_fc (function): Defines the fitness function of the individual. The function sould compute one float number, its
            input is the individual.
        """
        self.individual_type = individual_type
        self.fitness_fc = fitness_fc
        self.iteration = 0
        self.current_indiv = None
        self.best_indiv = None

    def __repr__(self) -> str:
        """Defines the string representation of the RandomController class.

        Returns:
            str: the representation string.
        """
        return f'RandomController iter:{self.iteration} best:{self.best_indiv}'
    
    def run(self, num_individuals:int, logger_fc=None, verbose:bool=False) -> list:
        """Runs the random search in the goal to reach the num of individuals.
        All the generated solutions can be logged by the given logger function.

        Args:
            num_individuals (int): number of individuals to be evaluated
            logger_fc (function): the logging function (as an attribute, this object is given). Defaults to None.
            verbose (bool): If true, information about the progress is printed. Defaults to False.

        Returns:
            list: representation of the best individual found.
        """
        for iteration in range(1, num_individuals + 1):
            
            # create new individual
            self.current_indiv = Individual(self.individual_type, self.fitness_fc)

            # compute fitness
            fitness = self.current_indiv.get_fitness()

            if self.best_indiv is None or fitness > self.best_indiv.fitness:
                self.best_indiv = self.current_indiv

            # log
            if logger_fc is not None:
                logger_fc(self)

            self.iteration += 1

            if verbose:
                print(f'Individual {iteration}/{num_individuals} ({self.iteration}) evaluated {self.current_indiv}')

        return self.best_indiv.representation

#---------------------------------------------

def tmp_fit(x):
    return random.random()

def tmp_log(x):
    print(x, x.current_indiv)

if __name__ == '__main__':
    tmp = RandomController([range(20), range(20)], tmp_fit)

    print(tmp.run(10,tmp_log))