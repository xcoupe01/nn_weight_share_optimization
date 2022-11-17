import random

RUN_CELLING = 1000 # restricts while True runs 

class EvolConfig:

    PAIRING_ALGORITHM = lambda list_indivs, num_pairs : pair_rulette(list_indivs, num_pairs)    # Set pairing algorithm
    CROSSOVER_ALGORITHM = lambda pair : breed_single_point_crossover(pair)                      # Set crossover algorithm
    MUTATION_ALGORITHM = lambda indiv, p : mutation_prob_for_each(indiv, p)                     # Set mutation algorithm
    MUTATION_PROBABILITY = 0.4                                                                  # Set mutation probability (depends also on used alg)
    NUM_ELITISTS = 1                                                                            # Set the number of elitist chromosomes

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

                while fitness_index > individuals[in_index].fitness:
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

def breed_single_point_crossover(pair:list) -> None:
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

    # chreating new individuals
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

