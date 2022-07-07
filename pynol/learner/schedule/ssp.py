import copy

import numpy as np
from pynol.learner.base import Base


class SSP:
    """The simplest class to initialize base-learners, which accepts the base
    instances as the input directly.

    Args:
        bases (list): List of base-learners.
    """
    def __init__(self, bases: list = None):
        self.bases = bases

    def __add__(self, ssp):
        new_ssp = copy.deepcopy(self)
        new_ssp.bases = self.bases + ssp.bases if self.bases is not None else ssp.bases
        return new_ssp

    def __len__(self):
        return len(self.bases)


class StepSizeFreeSSP(SSP):
    """The class to initialize step size free base-learners.

    Args:
        base_class (Base): Base class to schedule.
        num_bases (int): Number of base-learners.
        **kwargs_base (dict): Parameters of base-learners.
    """

    def __init__(self, base_class: Base, num_bases: int, **kwargs_base: dict):
        bases = [base_class(**kwargs_base) for _ in range(num_bases)]
        super().__init__(bases)


class DiscreteSSP(SSP):
    """The most commonly used SSP for dynamic algorithms, which construct a step
    size pool at first, and then initialize multiple base-learners, each employs
    a specific step size.

    Args:
        base_class (Base): Base class to initialize.
        min_step_size (float): Minimal value of the possible range of the
            optimal step size.
        max_step_size (float): Maximal value of the possible range of the
            optimal step size.
        grid (int): Grid size to discretize the possible range of the optimal
            step size.
        **kwargs (dict): Parameter of the base-learners.
    """

    def __init__(self,
                 base_class: Base,
                 min_step_size: float,
                 max_step_size: float,
                 grid: int = 2,
                 **kwargs_base):
        self.step_pool = self.discretize(min_step_size, max_step_size, grid)
        bases = [
            base_class(step_size=self.step_pool[i], **kwargs_base)
            for i in range(len(self.step_pool))
        ]
        super().__init__(bases)

    @staticmethod
    def discretize(min_step_size: float,
                   max_step_size: float,
                   grid: float = 2.):
        """Discretize the possible range of the optimal step size exponentially

        Args:
            min_step_size (float): Minimal value of the possible range of the
                optimal step size.
            max_step_size (float): Maximal value of the possible range of the
                optimal step size.
            grid (int): Grid size to discretize the possible range of the optimal
                step size.
        """
        step_pool = [min_step_size]
        while (min_step_size <= max_step_size):
            min_step_size *= grid
            step_pool.append(min_step_size)
        return np.array(step_pool)
