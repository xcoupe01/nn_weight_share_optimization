import random
import copy
import pandas as pd
import os
import numpy as np
from math import sqrt
from utils.fitness_controller import FitnessController

class Particle :
    def __init__(self, possible_values:list, max_velocity:list, inertia_c:float, cognitive_c: float, 
        social_c:float, position:list=None, velocity:list = None) -> None:
        """Inits particle object in order to do PSO optimization.

        Args:
            possible_values (list): Defines the possible position values. Each index of list corresponds to its dimension.
                Each item on index is a list of possible values in this dimension.
            max_velocity (list): Defines the maximum velocity for the particle. Each index of list corresponds to its dimension.
                Each item on index is expected to be a float value.
            inertia_c (float): is the particles inertia coeficient.
            cognitive_c (float): is the particles cognitive coeficient.
            social_c (float): is the particles social coeficient.
            position (list, optional): preset position (can be set only when velocity is specified). Defaults to None.
            velocity (list, optional): preset velocity (can be set only when position is specified). Defaults to None.

        Raises:
            Exception: Particle init error - particle setting do not match
            Exception: Particle init error - loaded data do not match
            Exception: Particle init error - cannot load only part of the data
        """

        if len(possible_values) != len(max_velocity):
            raise Exception('Particle init error - particle setting do not match')

        # setting atributes
        self.max_velocity = max_velocity
        self.possible_values = possible_values
        self.position_ranges = [[0, len(x)] for x in possible_values]
        self.position = []
        self.velocity = []
        self.fitness = None
        self.my_best:Particle = None
        self.inertia_c = inertia_c
        self.cognitive_c = cognitive_c
        self.social_c = social_c
        self.data = None
        self.save_only = False  # when True, the my best is unactive

        # loading position if defined
        if position is not None and velocity is not None:
            if len(position) != len(self.position_ranges) or len(velocity) != len(max_velocity):
                raise Exception('Particle init error - loaded data do not match')
            self.set_pos(position)
            self.velocity = velocity
            return

        if position is not None or velocity is not None:
            raise Exception('Particle init error - cannot load only part of the data')

        # generating position and velocity if not specified and setting representation
        self.rand_pos()

    def __repr__(self) -> str:
        """Defines the string representation of the particle object.

        Returns:
            str: Particle string with its information about current position, velocity and fitness.
        """

        return f'Particle position:{self.position} velocity:{self.velocity} fitness:{self.fitness}'
    
    def rand_pos(self) -> None:
        """Sets random position and velociti of the particle.
        """

        self.position = []
        self.velocity = []
        self.my_best = None

        # generating position and velocity
        for i in range(len(self.position_ranges)):
            self.position.append(random.uniform(*self.position_ranges[i]))
            self.velocity.append(random.uniform(-self.max_velocity[i], self.max_velocity[i]))

        # setting curent representation
        self.representation = []
        for i in range(len(self.possible_values)):
            index_position = min(int(self.position[i]), len(self.possible_values[i]) - 1)
            index_position = max(index_position, 0)
            self.representation.append(self.possible_values[i][index_position])

    def set_pos(self, position:list) -> None:
        """Sets the position of the particle and calculates its corresponding representation.

        Args:
            position (list): the particle position to be set.
        """

        # saving position
        self.position = position

        # setting curent representation
        self.representation = []
        for i in range(len(self.possible_values)):
            index_position = min(int(self.position[i]), len(self.possible_values[i]) - 1)
            index_position = max(index_position, 0)
            self.representation.append(self.possible_values[i][index_position])

    def move(self, swarm_best_pos:list, limit_position:bool=True, limit_velocity:bool=True) -> None:
        """Moves the particle accordigly to the PSO algorithm by following expression:
        - first the new velocity is calculated as:
            v(t+1) = in*v(t) + cp*rp*(pmb - p(t)) + cg*rg*(pb - p(t))

        - then the new position is calculated as:
            p(t+1) = p(t) + v(t+1)

        where:
            - v   - the velocity
            - t   - time of the current structure is set to (t+1 is the new position)
            - in  - is the inertia
            - cp  - is the cognitive coeficient
            - rp  - is the cognitive multiplier
            - pmb - is the particles best position achieved yet
            - p   - is the current position 
            - cg  - is the social coeficient
            - rg  - is the social multiplier
            - pb  - is the swarm best position

        the move can be aslo limited by the parameters to inited values

        Args:
            swarm_best_pos (list): is the pb of the move (swarm best position). 
            limit_position (bool, optional): tells if the position should be limited . Defaults to True.
            limit_velocity (bool, optional): tells if the velocity should be limied. Defaults to True.
        """

        # my best position yet check
        if self.my_best is None:
            my_best_pos = self.position
        else:
            my_best_pos = self.my_best.position

        # compute new velocity
        for i in range(len(self.velocity)):
            self.velocity[i] = self.inertia_c * self.velocity[i] + \
                self.cognitive_c * random.uniform(0, 1) * (my_best_pos[i] - self.position[i]) + \
                self.social_c * random.uniform(0, 1) * (swarm_best_pos[i] - self.position[i])
            # velocity limitation
            if limit_velocity and self.velocity[i] > self.max_velocity[i]:
                self.velocity[i] = self.max_velocity[i]
            if limit_velocity and self.velocity[i] < - self.max_velocity[i]:
                self.velocity[i] = -self.max_velocity[i]

        # compute new position
        for i in range(len(self.position)):
            self.position[i] = self.position[i] + self.velocity[i]
            # position limitation
            if limit_position:
                self.position[i] = max(self.position[i], self.position_ranges[i][0])
                self.position[i] = min(self.position[i], self.position_ranges[i][1])
        
        # compute new representation
        self.representation = []
        for i in range(len(self.possible_values)):
            index_position = min(int(self.position[i]), len(self.possible_values[i]) - 1)
            self.representation.append(self.possible_values[i][index_position])
        
        # erase data
        self.data = None

    def compute_fitness(self, fit_func) -> float:
        """Updates the particles fitness and its best position fitness by
        given fitness function.

        If save_only is set to true, the my_best position is not updated.

        Args:
            fit_func (function): if the function to compute the fitness from with the particles data.

        Returns:
            float: The fitness value of this particle.
        """
       
        # get my fitness
        self.fitness = fit_func(self.data)

        # update my best fitness 
        if self.my_best is not None:
            self.my_best.fitness = fit_func(self.my_best.data)

        # update my best
        if not self.save_only and (self.my_best is None or self.my_best.fitness < self.fitness):
            self.my_best = copy.deepcopy(self)
            self.my_best.my_best = None

        return self.fitness

class PSOController:
    def __init__(self, num_particles:int, particle_range:list, particle_max_velocity:list,
        inertia_c:float, fitness_cont:FitnessController, cognitive_c:float=2.05, social_c:float=2.05, 
        BH_radius:float = None, BH_vel_tresh:float = None, BH_repr_rad:bool = False) -> None:
        """Inits the PSO controller with its swarm. Also can be upgraded to Black Hole algorithm
        by specifying the radius and velocity theshold.

        Args:
            num_particles (int): is the number of the particlen in the PSO swarm.
            particle_range (list): particle ranges as describes in init Particle object.
            particle_max_velocity (list): particle max velocity values as described in init Particle object.
            inertia_c (float): inertia coeficient of the algorithm.
            fitness_cont (FitnessController): is a fitness controler object. 
            cognitive_c (float, optional): cognitive coeficient of the particles. Defaults to 2.05.
            social_c (float, optional): social coeficient of the particles. Defaults to 2.05.
            BH_radius (float, optional): Part of Black Hole algorithm upgrade. Defines the radius,
                where the paricles are absorbed by the best position.
            BH_vel_tresh (float, optional):Part of Black Hole algorithm upgrade. Defines the velocity theshold.
                When some particle is slover than this speed, its absorbed by the best position.
            BH_repr_rad (float, optional): If True, BH_radius is not taken into account and instead if 
                the representation is the same as best position, the particle is absorbed. Defaults to False.
        """

        # set attributes
        self.inertia_c = inertia_c
        self.cognitive_c = cognitive_c
        self.social_c = social_c
        self.swarm:list[Particle] = []
        self.time = 0
        self.jump_start = False
        self.particle_range = particle_range
        self.BH_radius = BH_radius
        self.BH_repr_rad = BH_repr_rad
        self.BH_vel_tresh = BH_vel_tresh
        self.fitness_controller = fitness_cont
        self.max_vel_vect = sqrt(np.sum(np.power(np.array(particle_max_velocity), 2)))

        # init swarm
        for _ in range(num_particles):
            self.swarm.append(Particle(particle_range, particle_max_velocity, inertia_c, cognitive_c, social_c))

    def __repr__(self) -> str:
        """Defines the string representation of the PSO controller object.

        Returns:
            str: the string repsresentation of the POS controller.
        """

        return f'PSO Controller: time: {self.time}, num particles {len(self.swarm)}'

    def load_from_pd(self, dataframe:pd.DataFrame, verbose:bool = False) -> None:
        """Loads the PSO controller state from a pandas dataframe.
        It assumes the dataframe contains the dataframe is saved in a way, that there
        are at least position, velocity, compression rate and accuracy attributes for each time step
        of the algorithm. Also that the dataframe is sorted by the time and
        each particle have the same index in the one time period.

        Created for the pourpouses of WS optimalization!

        Args:
            dataframe (pd.DataFrame): is the dataframe to be loaded from.
            verbose (bool, optional): If true, prints out informations. Defaults to False.
        """
        
        # setting the controller
        self.time = dataframe.time.max()
        self.jump_start = True

        self.fitness_controller.fit_from_df(dataframe, verbose=verbose)

        # setting the swarm
        for i in range(len(self.swarm)):
            # skipping to over the same number of rows to get the particle evolution
            # dataframe through time
            ptd = dataframe.iloc[i::(len(self.swarm)), :]
            best_fit = ptd[ptd.fitness == ptd.fitness.max()].reset_index()
            current_fit = ptd[ptd.time == ptd.time.max()].reset_index()

            # setting particle attribudes by the file
            self.swarm[i].position = eval(current_fit.iloc[0]['position'])
            self.swarm[i].velocity = eval(current_fit.iloc[0]['velocity'])
            self.swarm[i].fitness = current_fit.iloc[0]['fitness']

            particle_best = Particle(
                self.particle_range, 
                self.swarm[i].max_velocity, 
                self.inertia_c, 
                self.cognitive_c, 
                self.social_c, 
                eval(best_fit.iloc[0]['position']) if type(best_fit.iloc[0]['position']) is str else best_fit.iloc[0]['position'],
                eval(best_fit.iloc[0]['velocity']) if type(best_fit.iloc[0]['velocity']) is str else best_fit.iloc[0]['velocity'])
            particle_best.data = {
                'accuracy': best_fit.iloc[0]['accuracy'], 
                'compression': best_fit.iloc[0]['compression'],
            }
            particle_best.fitness = best_fit.iloc[0]['fitness']
            self.swarm[i].my_best = particle_best

            self.swarm[i].representation = eval(current_fit.iloc[0]['representation'])

    def run(self, time:int, logger_fc=None, save_data:pd.DataFrame=None, limit_position:bool=True, limit_velocity:bool=True, verbose:bool=False) -> list:
        """Executes the whole particle swarm optimization for a given number of time iterations.

        Args:
            time (int): is the number of iterations over the whole swarm.
            logger_fc (function, optional): is an optional logger function that can acces the whole object
                after each iteration and is meant to log data (if save_data given its also its input). Defaults to None.
            save_data (pd.Dataframe, optional): is an optional pandas datagrame to log data to. If defined,
                it is expected that logger_fc takes this dataframe and the function returns the updated one.
                Also if defined. the function returns the saves, otherwise the best found solution is returned.
            limit_position (bool, optional): defines if the position is limited during the run. Defaults to True.
            limit_velocity (bool, optional): defines if the velocity is limited during the run. Defaults to True.
            verbose (bool, optional): enables progression prints. Defaults to False.

        Returns:
            list: the best found position of the whole swarm. When save_data is defined, the data are returned instead.
                or
            pd.Dataframe: save_data if this parameter is defined at the begining
        """

        # initing swarm fitness
        if not self.jump_start:
            self.fitness_controller.compute_fit(self.swarm, verbose=verbose)
            # logging
            if logger_fc is not None:
                if save_data is not None:
                    save_data = logger_fc(self, save_data)
                else:
                    logger_fc(self)

        # initing best swarm position and fitness
        best_particle:Particle = max(self.swarm, key=lambda p: p.my_best.fitness).my_best
        # set save only to not compute my_best of the best particle
        best_particle.save_only = True
        
        start_from = 1 if self.jump_start else 2
        if verbose: 
            print(f'Time {start_from-1}/{time} ({self.time}) best fitness {best_particle.fitness}')

        # optimization loop
        for t in range(start_from, time+1):
            for particle in self.swarm:
                particle.move(best_particle.position, limit_position, limit_velocity)
            
            # ensure fitness update of the possibly detached best_particle
            self.fitness_controller.compute_fit([*self.swarm, best_particle], verbose=verbose)

            # this is here because of BH update - can happen that the particle with the
            # best my_best is reseted, but this link should be intact if it is still the 
            # best found swarm position
            best_particle_candidat = max(self.swarm, key=lambda p: p.my_best.fitness).my_best
            if best_particle_candidat.fitness > best_particle.fitness:
                best_particle = best_particle_candidat
                # set save only to not compute my_best of the best particle
                best_particle.save_only = True

            self.time += 1
            if verbose:
                print(f'Time {t}/{time} ({self.time}) best fitness {best_particle.fitness}')

            # logging
            if logger_fc is not None:
                if save_data is not None:
                    save_data = logger_fc(self, save_data)
                else:
                    logger_fc(self)

            # blackhole algorithm upgrade
            if (self.BH_radius is not None or self.BH_repr_rad) and self.BH_vel_tresh is not None:
                reseted_particles_count = 0
                for particle in self.swarm:
                    if particle == best_particle:
                        continue
                    
                    # position radius path
                    if self.BH_radius is not None and\
                        np.linalg.norm(np.array(particle.position) - np.array(best_particle.position)) <= self.BH_radius and \
                        np.linalg.norm(particle.velocity) <= (self.BH_vel_tresh * self.max_vel_vect):
                        particle.rand_pos()
                        reseted_particles_count += 1

                    # representation raduis path
                    elif self.BH_repr_rad is not False and\
                        particle.representation == best_particle.representation and\
                        np.linalg.norm(particle.velocity) <= (self.BH_vel_tresh * self.max_vel_vect):
                        particle.rand_pos()
                        reseted_particles_count += 1
                
                if verbose and reseted_particles_count > 0:
                    print(f'BH {reseted_particles_count} {"particles" if reseted_particles_count > 1 else "particle"} reset')

        # returning the best found representation
        best_representation = []

        for i in range(len(self.particle_range)):
            index_position = min(int(best_particle.position[i]), len(self.particle_range[i]) - 1)
            best_representation.append(self.particle_range[i][index_position])

        if save_data is not None:
            return save_data

        return best_representation

# simple test run to optimize quadratic function
if __name__ == '__main__':

    # load save file if possible
    SAVE_FILE = './results/test/test_PSO.csv'
    data_template = {'position': [], 'velocity': [], 'representation': [], 'fitness': [], 'time': []}
    data_df = pd.read_csv(SAVE_FILE) if os.path.isfile(SAVE_FILE) else pd.DataFrame(data_template)

    # logger function
    def logger_fc(controller):
        global data_df

        new_data = copy.deepcopy(data_template)
        for particle in controller.swarm:
            new_data['position'].append(particle.position)
            new_data['velocity'].append(particle.velocity)
            new_data['representation'].append(particle.representation)
            new_data['fitness'].append(particle.current_fit)
            new_data['time'].append(controller.time)

        data_df = data_df.append(pd.DataFrame(new_data), ignore_index=True)

    # init fit
    get_fit_vals = lambda p: [- pow(p.representation[0], 2), - pow(p.representation[1], 2)]
    def fit_from_vals(p, fv, mv): p.fitness = fv[0] + fv[1]
    fitness_cont = FitnessController([0, 0], get_fit_vals, fit_from_vals)
    
    # init controllers
    controler = PSOController(10, [range(-100, 100), range(-100, 100)], [1, 1], inertia_c=0.8, fitness_cont=fitness_cont, BH_radius=1, BH_vel_tresh=1)

    # load the controler with data - probably wont work :(
    if len(data_df.index) > 0:
        controler.load_from_pd(data_df)

    # run optimization
    print(controler.run(20, logger_fc, verbose=True))

    # save data
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    data_df.to_csv(SAVE_FILE, index=False)
