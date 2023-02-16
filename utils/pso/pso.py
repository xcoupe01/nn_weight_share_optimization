import random
import copy
import pandas as pd
import os
import numpy as np
from math import sqrt

class Particle :
    def __init__(self, possible_values:list, max_velocity:list, fitness_fc, 
        inertia_c:float, cognitive_c: float, social_c:float, position:list=None, velocity:list = None) -> None:
        """Inits particle object in order to do PSO optimization.

        Args:
            position_ranges (list): Defines the possible position ranges. Each index of list corresponds to its dimension.
            Each item on index is expected to be a tuple in format (minimal value, maxima value).
            max_velocity (list): Defines the maximum velocity for the particle. Each index of list corresponds to its dimension.
            Each item on index is expected to be a float value.
            fitness_fc (function): Defines the fitness function of the particle. The function sould compute one float number, its
            input is the particle position.
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
        self.current_fit = None
        self.my_best_pos = None
        self.my_best_fit = None
        self.fitness_fc = fitness_fc
        self.inertia_c = inertia_c
        self.cognitive_c = cognitive_c
        self.social_c = social_c
        self.data = None

        # loading position if defined
        if position is not None and velocity is not None:
            if len(position) != len(self.position_ranges) or len(velocity) != len(max_velocity):
                raise Exception('Particle init error - loaded data do not match')
            self.position = position
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

        return f'Particle position:{self.position} velocity:{self.velocity} fitness:{self.current_fit}'
    
    def rand_pos(self) -> None:
        """Sets random position and velociti of the particle.
        """

        self.position = []
        self.velocity = []

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

    def compute_fitness(self) -> float:
        """Computes the particles fitness based on the set fitness function.

        Returns:
            float: current fitness of the particle
        """

        self.current_fit = self.fitness_fc(self)
        if self.my_best_pos is None or self.current_fit > self.my_best_fit:
            self.my_best_fit = self.current_fit
            self.my_best_pos = copy.deepcopy(self.position)
        return self.current_fit

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

        # compute new velocity
        for i in range(len(self.velocity)):
            self.velocity[i] = self.inertia_c * self.velocity[i] + \
                self.cognitive_c * random.uniform(0, 1) * (self.my_best_pos[i] - self.position[i]) + \
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

class PSOController:
    def __init__(self, num_particles:int, particle_range:list, particle_max_velocity:list,
        particle_fitness, inertia_c:float, cognitive_c:float=2.05, social_c:float=2.05, 
        BH_radius:float = None, BH_vel_tresh:float = None) -> None:
        """Inits the PSO controller with its swarm.

        Args:
            num_particles (int): is the number of the particlen in the PSO swarm.
            particle_range (list): particle ranges as describes in init Particle object.
            particle_max_velocity (list): particle max velocity values as described in init Particle object.
            particle_fitness (function): particle fitness function.
            inertia_c (float): inertia coeficient of the algorithm.
            cognitive_c (float, optional): cognitive coeficient of the particles. Defaults to 2.05.
            social_c (float, optional): social coeficient of the particles. Defaults to 2.05.
        """

        # set attributes
        self.inertia_c = inertia_c
        self.cognitive_c = cognitive_c
        self.social_c = social_c
        self.swarm:list[Particle] = []
        self.time = 0
        self.jump_start = False
        self.partice_range = particle_range
        self.BH_radius = BH_radius
        self.BH_vel_tresh = BH_vel_tresh

        # init swarm
        for _ in range(num_particles):
            self.swarm.append(
                Particle(particle_range, particle_max_velocity, 
                    particle_fitness, inertia_c, cognitive_c, social_c)
                )

    def __repr__(self) -> str:
        """Defines the string representation of the PSO controller object.

        Returns:
            str: the string repsresentation of the POS controller.
        """

        return f'PSO Controller: time: {self.time}, num particles {len(self.swarm)}'

    def load_from_pd(self, dataframe:pd.DataFrame) -> None:
        """Loads the PSO controller state from a pandas dataframe.
        It assumes the dataframe contains the dataframe is saved in a way, that there
        are at least position, velocity and fitness attributes for each time step
        of the algorithm. Also that the dataframe is sorted by the time and
        each particle have the same index in the one time period.

        Args:
            dataframe (pd.DataFrame): is the dataframe to be loaded from.
        """
        
        # setting the controller
        self.time = dataframe.time.max()
        self.jump_start = True

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
            self.swarm[i].current_fit = current_fit.iloc[0]['fitness']
            self.swarm[i].my_best_pos = eval(best_fit.iloc[0]['position'])
            self.swarm[i].my_best_fit = best_fit.iloc[0]['fitness']
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
            for particle in self.swarm:
                particle.compute_fitness()
            # logging
            if logger_fc is not None:
                if save_data is not None:
                    save_data = logger_fc(self, save_data)
                else:
                    logger_fc(self)

        # initing best swarm position and fitness
        best_particle = max(self.swarm, key=lambda p: p.my_best_fit)
        best_position = copy.deepcopy(best_particle.my_best_pos)
        best_fitness = best_particle.my_best_fit
        
        start_from = 1 if self.jump_start else 2
        if verbose: 
            print(f'Time {start_from-1}/{time} ({self.time}) best fitness {best_fitness}')

        # optimization loop
        for t in range(start_from, time+1):
            for particle in self.swarm:
                particle.move(best_position, limit_position, limit_velocity)
                particle.compute_fitness()
            best_particle = max(self.swarm, key=lambda p: p.my_best_fit)
            best_position = copy.deepcopy(best_particle.my_best_pos)
            best_fitness = best_particle.my_best_fit

            self.time += 1
            if verbose:
                print(f'Time {t}/{time} ({self.time}) best fitness {best_fitness}')

            # logging
            if logger_fc is not None:
                if save_data is not None:
                    save_data = logger_fc(self, save_data)
                else:
                    logger_fc(self)

            # blackhole algorithm upgrade
            if self.BH_radius is not None and self.BH_vel_tresh is not None:
                for particle in self.swarm:
                    if particle == best_particle:
                        continue
                    if sqrt(np.sum(np.power((np.array(particle.position) - np.array(best_position)), 2))) <= self.BH_radius and \
                        np.sum(np.power(np.array(particle.velocity), 2)) <= self.BH_vel_tresh:
                        particle.rand_pos()

        # returning the best found representation
        best_representation = []

        for i in range(len(self.partice_range)):
            index_position = min(int(best_position[i]), len(self.partice_range[i]) - 1)
            best_representation.append(self.partice_range[i][index_position])

        if save_data is not None:
            return save_data

        return best_representation

# simple test run to optimize quadratic function
if __name__ == '__main__':

    SAVE_FILE = './results/test/test_PSO.csv'

    data_template = {'position': [], 'velocity': [], 'representation': [], 'fitness': [], 'time': []}
    data_df = pd.read_csv(SAVE_FILE) if os.path.isfile(SAVE_FILE) else pd.DataFrame(data_template)

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

    fitness_fc = lambda p: - pow(p.representation[0], 2) - pow(p.representation[1], 2)

    controler = PSOController(10, [range(-100, 100), range(-100, 100)], [1, 1], fitness_fc ,inertia_c=0.8, BH_radius=1, BH_vel_tresh=1)

    if len(data_df.index) > 0:
        controler.load_from_pd(data_df)

    print(controler.run(20, logger_fc, verbose=True))

    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    data_df.to_csv(SAVE_FILE, index=False)
