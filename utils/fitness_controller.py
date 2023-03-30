import numpy as np
import pandas as pd

class FitnessController:
    def __init__(self, base_targs:list[float], get_fit_vals, fit_from_vals, fitness_targ_update_fc = None, 
        target_update_offset:list[float] = None, target_limit:list[float] = None ,lock:bool = False):
        """Inits the fitness controller.

        The fitness controler is desined to work with unreachable best possible target in some spectrum.
        The basic target is set, then when the controller sees that better value then the target was found, it updates the
        target to push the optimalization even further to get better solutions.

        For now its designed to operate for `accuracy` and `compression` used in the WS. Curently only `compression`
        target value is being moved.

        The controller works in a following way:
        1) it computes all the values need for the fitness value (in the WS its `accuracy` and `compression`) by
            fit_from_vals function in arguments.
        2) Then it takes the maximum of each metric and compares them to its target. If any value is greater than the target,
            The target is shifted in this dimentsion to this value + target_max_offset.
        3) Lastly all the fitness values are calculated by fit_from_vals.

        Args:
            base_targs (list[float]): Is the base target of the fitness calculation.
            get_fit_vals (function): Function, that takes the representation and gets the needed values to compute fitness.
            fit_from_vals (function): Function, that takes the values from previous function and computes the fitness.
            fitness_targ_update_fc (function, optional): Function that is triggered, when the target changes. Defaults to None.
            target_update_offset (float, optional): New target offset. Defaults to None (all offsets are zero).
            target_limit (list[floar], optional): Defines the lower bound in each dimenstion for point
                to be considered as possible target (ie.: when you have possible target that have 0 acc and 32 compression
                and you dont want it to be proceeded and taken as target for the compression, u set the acc limit to be 0.95
                which means this point would not pass as target because 0 < 0.95). Defaults to None (No limits).
            lock (bool, optional): If true, the target does not move. Defaults to False.
        """

        if target_update_offset is None:
            target_update_offset = [0 for _ in base_targs]

        self.targ = np.array(base_targs)
        self.get_fit_vals = get_fit_vals
        self.fit_from_vals = fit_from_vals
        self.fitness_targ_update_fc = fitness_targ_update_fc
        self.target_offset = target_update_offset
        self.lock = lock
        self.target_limit = target_limit

    def update_targs(self, potential_targs:list[float], verbose:bool = False) -> bool:
        """Updates the target values to potential target values if the potentila target value is greater.

        Args:
            potential_targs (list[float]): Is the potential new target.
            verbose (bool, optional): If True, the information is outputed to the console. Defaults to False.

        Returns:
            bool: If the change happened.
        """
        
        # exit if locked
        if self.lock:
            return False

        change = False

        if np.array(potential_targs).ndim == 1:
            potential_targs = [potential_targs]

        # limiting points
        if self.target_limit is not None:
            for i, limit in enumerate(self.target_limit):
                potential_targs = [x for x in potential_targs if x[i] > limit]
        
        if len(potential_targs) == 0:
            return False

        potential_targs = np.array(potential_targs).max(axis=0)

        # change the target
        for i in [i for i, x in enumerate(potential_targs >= self.targ) if x]:
            self.targ[i] = potential_targs[i] + self.target_offset[i]
            change = True

        if verbose and change: print(f'Fitness target update to {self.targ}')

        return change
        
    def compute_fit(self, individuals:list, verbose:bool = False) -> None:
        """Provides the complete fitness calculation for group of representations.
        It firstly computes all the components of the fitness value, than fixes the target,
        and lastly computes all the fitness values.

        Args:
            individuals (list): List of all the representation objects.
            verbose (bool, optional): If True, information is printed when target is changed. Defaults to False.
        """

        fit_vals = []

        # calculating the fitness data
        for individual in individuals:
            fit_vals.append(self.get_fit_vals(individual))

        # updating the target
        if self.update_targs(fit_vals, verbose) and self.fitness_targ_update_fc is not None:
            for individual in individuals:
                self.fitness_targ_update_fc(individual, self.targ)

        # computing the fitness function
        fit_lam = lambda data: self.fit_from_vals(data, self.targ)
        for individual in individuals:
            individual.compute_fitness(fit_lam)

    def fit_from_df(self, df:pd.DataFrame, verbose:bool = False):
        """Computes fitness column to pandas dataframe containing the optimization
        process.

        Args:
            df (pd.DataFrame): Is the pandas dataframe.
            verbose (bool, optional): If true, the target updates are printed out in console. Defaults to False.
        """

        # update the target
        all_vals = np.transpose(np.array([df['accuracy'], df['compression']]))
        self.update_targs(all_vals, verbose)

        # generate fitness column
        df['fitness'] = 0.0
        for i, row in df.iterrows():
            df['fitness'][i] = self.fit_from_vals(row, self.targ)

    def update_targ_by_dfs(self, dfs:list[pd.DataFrame], verbose:bool):
        
        for df in dfs:
            self.fit_from_df(df, verbose=verbose)