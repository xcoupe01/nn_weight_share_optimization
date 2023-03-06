import random

RUN_CELLING = 1000 # restricts while True runs 

class EvolConfig:

    PAIRING_ALGORITHM = lambda list_indivs, num_pairs : pair_rulette(list_indivs, num_pairs)    # Set pairing algorithm
    CROSSOVER_ALGORITHM = lambda pair : breed_single_point_crossover(pair)                      # Set crossover algorithm
    MUTATION_ALGORITHM = lambda indiv, p : mutation_prob_for_each(indiv, p)                     # Set mutation algorithm
    MUTATION_PROBABILITY = 0.2                                                                  # Set mutation probability (depends also on used alg)
    NUM_ELITISTS = 1                                                                            # Set the number of elitist chromosomes
    PAIRING_ALGORITHM_STR = 'roulette'                                                          # pairing algorithm text indication for experiment settings dump
    CROSSOVER_ALGORITHM_STR = 'single point'                                                    # crossover algorithm text indication for experiment settings dump
    MUTATION_ALGORITHM_STR = 'prob for each'                                                    # mutation algorithm text indication for experiment settings dump

# -------------- breeding functions --------------

def pair_rulette(individuals:list, num_pairs:int) -> list:
    """Performs a rullete pairing of individuals. The result is 
    a list of pairings to be crosbreeded.
    The rulette choses form the population by their fitness value.
    Each individual has a chance to be chosen as the ration of 
    its fitness value and the total fitness value.

    Args:
        individuals (list): the list of the whole population
        num_pairs (int): number of pairs wantet to be generated to breed.

    Raises:
        Exception: When the RN generator failes and the algorithm cannot create proper
        pairs, an exception is raised.

    Returns:
        list: list of pairing of individuals ready for crossbreeding.
    """

    fit_sum = sum(i.fitness for i in individuals)
    pairing = []

    # creating pairing - assuming pair makes two new individuals
    for _ in range(num_pairs):
        parents = []

        for _ in range(2):
            run = 0
            while True:
                
                # searching for parent - rulette
                fitness_index = random.random() * fit_sum
                in_index = 0
                run += 1

                if fit_sum > 0:
                    while fitness_index > individuals[in_index].fitness:
                        fitness_index -= individuals[in_index].fitness
                        in_index += 1
                else:
                    while fitness_index < individuals[in_index].fitness:
                        fitness_index -= individuals[in_index].fitness
                        in_index += 1  

                # adding parent
                if individuals[in_index] not in parents:
                    parents.append(individuals[in_index])
                    break
                
                # to not get stuck on the while loop
                if run >= RUN_CELLING:
                    raise Exception('pair_rulette error - too many runs')

        pairing.append(parents)
    
    return pairing

# -------------- pairing functions --------------

def breed_single_point_crossover(pair:list) -> list:
    """Breeding algorithm - single point crossover.
    One point in the pairings chromosomes are chosen and
    the parts after the point the chromosome values
    swap between the pairing. The point is chosen randomly.

    Args:
        pair (list): a pair of Individual objects to be breeded.

    Returns:
        list: list of new chromosomes.
    """
    # cannot be 0 or len(chromosome) - no crossover happens when those indexes
    breed_index = random.randint(1, len(pair[0].chromosome)-1)
    
    # creating new chromosomes
    new_chromosome1 = pair[0].chromosome[:breed_index] + pair[1].chromosome[breed_index:]
    new_chromosome2 = pair[1].chromosome[:breed_index] + pair[0].chromosome[breed_index:]

    # creating new individuals
    return [new_chromosome1, new_chromosome2]

def breed_uniform(pair:list) -> list:
    """Breeding algorthm - uniform crossover.
    List is being generated where item on each index
    tells if the child should have corresponding chromosome
    list item from parent 0 or parent 1. The other child
    is the oposite. The breed list is chosen randomly.

    Args:
        pair (list): a pair of Individual objects to be breeded

    Raises:
        Exception: when the function fails to find the breed list.

    Returns:
        list: list of new chromosomes
    """
    
    # breed split list
    breed_list = random.choices([True, False], k=len(pair[0].chromosome))
    
    # ensuring optimal breed split list
    run = 0
    while all(breed_list) or not any(breed_list):
        run += 1
        breed_list = random.choices([True, False], k=len(pair[0].chromosome))
        if run >= RUN_CELLING:
            raise Exception('breed_uniform error - cannot create breed list')

    # creating new chromosomes
    new_chromosome1 = [pair[0].chromosome[x] if breed_list[x] else pair[1].chromosome[x] for x in range(len(pair[0].chromosome))]
    new_chromosome2 = [pair[1].chromosome[x] if breed_list[x] else pair[0].chromosome[x] for x in range(len(pair[0].chromosome))]

    # creating new individuals
    return [new_chromosome1, new_chromosome2]

# -------------- mutation functions --------------

def mutation_prob_for_each(individual, p:float) -> None:
    """Mutation algorithm. Mutates the given individuals
    chromosome in the ranges which are set to this individual.
    Each gen in the chromosome has a `p` chance to be mutated.

    Args:
        individual (Individual): an Individual type object.
        p (float): chance of mutation for every gen in chromosome.
    """

    # for every gen in chromosome, there is a p chance of mutation
    for i in range(len(individual.chromosome)):
        if random.random() <= p:
            individual.chromosome[i] = random.choice(individual.possible_values[i])

# -------------- config dump --------------

def dump_ga_config() -> object:
    """Creates object structure containing all the settings of the evolution config.
    The functions are represented with corresponding string name.

    Returns:
        object: the save structure of the settings.
    """

    return {
        'pairing': EvolConfig.PAIRING_ALGORITHM_STR,
        'crossover': EvolConfig.CROSSOVER_ALGORITHM_STR,
        'mutation': EvolConfig.MUTATION_ALGORITHM_STR,
        'mut prob': EvolConfig.MUTATION_PROBABILITY,
        'num elit': EvolConfig.NUM_ELITISTS,
    }

def load_ga_config(json:object) -> None:
    """Loads the config from given json. It does not load the corresponding functions.

    Args:
        json (object): Is the object to be loaded from.
    """
    EvolConfig.MUTATION_PROBABILITY = json['mut prob']
    EvolConfig.NUM_ELITISTS = json['num elit']
