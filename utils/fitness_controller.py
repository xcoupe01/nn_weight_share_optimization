import numpy as np
import pandas as pd

class FitnessController:
    def __init__(self, base_targs:list[float], get_fit_vals, fit_from_vals, fitness_targ_update_fc = None, target_max_offset:float = 0, lock:bool = False):
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

        # change the target
        for i in [i for i, x in enumerate(potential_targs >= self.targ) if x]:
            self.targ[i] = potential_targs[i] + self.target_offset
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
        max_vals = np.array(fit_vals).max(axis=0)
        if self.update_targs(max_vals, verbose) and self.fitness_targ_update_fc is not None:
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
        max_vals = [df['accuracy'].max(), df['compression'].max()]
        self.update_targs(max_vals, verbose)

        # generate fitness column
        df['fitness'] = 0
        for i, row in df.iterrows():
            df['fitness'][i] = self.fit_from_vals(row, self.targ)
        