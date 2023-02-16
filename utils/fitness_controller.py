import numpy as np
import pandas as pd

class FitnessController:
    def __init__(self, base_targs:list[float], get_fit_vals, fit_from_vals, fitness_targ_update_fc = None, target_max_offset:float = 0, lock:bool = False):
        """Inits the fitness controller.

        TODO: short description, how it works.

        Args:
            base_targs (list[float]): Is the base target of the fitness calculation.
            get_fit_vals (function): Function, that takes the representation and gets the needed values to compute fitness.
            fit_from_vals (function): Function, that takes the values from previous function and computes the fitness.
            fitness_targ_update_fc (function, optional): Function that is triggered, when the target changes. Defaults to None.
            target_max_offset (float, optional): New target offset. Defaults to 0.
            lock (bool, optional): If true, the target does not move. Defaults to False.
        """

        self.targ = np.array(base_targs)
        self.get_fit_vals = get_fit_vals
        self.fit_from_vals = fit_from_vals
        self.fitness_targ_update_fc = fitness_targ_update_fc
        self.target_offset = target_max_offset
        self.lock = lock

    def update_targs(self, potential_targs:list[float], verbose:bool = False) -> bool:
        
        if self.lock:
            return False

        change = False

        for i in [i for i, x in enumerate(potential_targs >= self.targ) if x]:
            self.targ[i] = potential_targs[i] + self.target_offset
            change = True

        if verbose and change: print(f'Fitness target update to {self.targ}')

        return change
        
    def compute_fit(self, individuals:list, verbose:bool = False) -> None:

        fit_vals = []

        for individual in individuals:
            fit_vals.append(self.get_fit_vals(individual))

        max_vals = np.array(fit_vals).max(axis=0)

        if self.update_targs(max_vals, verbose) and self.fitness_targ_update_fc is not None:
            for individual in individuals:
                self.fitness_targ_update_fc(individual, self.targ)

        for i, individual in enumerate(individuals):
            self.fit_from_vals(fit_vals[i], self.targ, individual=individual)

    def fit_from_df(self, df:pd.DataFrame, verbose:bool = False):

        max_vals = [df['accuracy'].max(), df['compression'].max()]

        self.update_targs(max_vals, verbose)

        df['fitness'] = 0
        for i, row in df.iterrows():
            df['fitness'][i] = self.fit_from_vals([row['accuracy'], row['compression']], self.targ)
        