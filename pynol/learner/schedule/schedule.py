from copy import deepcopy
from typing import Optional, Union

import numpy as np
from pynol.environment.environment import Environment
from pynol.learner.schedule.cover import (Cover, FullCover, PCover)
from pynol.learner.schedule.ssp import SSP


class Schedule:
    """The class to schedule base-learners with problem-independent cover.

    Args:
        ssp (SSP): A ssp instance containing a bunch of initialized
            base-learners.
        cover (Cover, optional): A cover instance deciding the
            active state of base-learners.

    """

    def __init__(self, ssp: SSP, cover: Optional[Cover] = None):
        self.bases = ssp.bases
        self.cover = cover if cover is not None else FullCover(len(self.bases))
        self._t = 0
        self._optimism = None
        self.active_state = np.ones(len(self.bases))
        self.active_index = np.where(self.active_state > 0)[0]

    def opt_by_optimism(self, optimism: np.ndarray):
        """Optimize by the optimism for all base-learners.

        Args:
            optimism (numpy.ndarray): External optimism for all alive base-learners.
        """
        for idx in self.active_index:
            self.bases[idx].opt_by_optimism(optimism)

    def opt_by_gradient(self, env: Environment):
        """Optimize by the gradient for all base-learners.

        Args:
            env (Environment): Environment at current round.

        Returns:
            tuple: tuple contains:
                loss (float): the origin loss of all alive base-learners. \n
                surrogate_loss (float): the surrogate loss of all alive base-learners.
        """
        loss = np.zeros(len(self.active_index))
        surrogate_loss = np.zeros_like(loss)
        for i, idx in enumerate(self.active_index):
            _, loss[i], surrogate_loss[i] = self.bases[idx].opt_by_gradient(
                env)
        return loss, surrogate_loss

    @property
    def t(self):
        """Set the number of current round, get active state from ``cover`` and
        reinitialize the base-learners whose active state is 2.
        """
        return self._t

    @t.setter
    def t(self, t):
        self._t = t
        self.cover.t = t
        self.active_state = self.cover.active_state
        self.active_index = np.where(self.active_state > 0)[0]
        self.reinit_bases()

    @property
    def x_active_bases(self):
        """Get the decisions of all alive base-learners.

        Returns:
            numpy.ndarray: Decisions of all alive base learners.
        """
        return np.array([self.bases[i].x for i in self.active_index])

    @property
    def optimism(self):
        """Get the optimisms of all alive base-learners.

        Returns:
            numpy.ndarray: Optimisms of all alive base learners.
        """
        self._optimism = np.zeros_like(self.x_active_bases)
        for i, idx in enumerate(self.active_index):
            self._optimism[i] = self.bases[idx].optimism
        return self._optimism

    def reinit_bases(self):
        """Reinitialize the base-learners whose active state is 2."""
        reinit_idx = np.where(self.active_state == 2)[0]
        for idx in reinit_idx:
            self.bases[idx].reinit()


class PSchedule(Schedule):
    """The class to schedule base-learners with problem-dependent cover.

    Args:
        ssp (SSP): A ssp instance containing a bunch of initialized
            base-learners.
        cover (PCover, optional): A cover instance deciding the
            active state of base-learners.

    """

    def __init__(self, ssp: SSP, cover: Optional[PCover] = None):
        super().__init__(ssp, cover)
        self._instance = deepcopy(self.bases[0])

    def opt_by_optimism(self, optimism: np.ndarray):
        """Optimize by the optimism for all base-learners.

        Args:
            optimism (numpy.ndarray): External optimism for all alive base-learners.
        """
        for idx in self.active_index:
            self.bases[idx].opt_by_optimism(optimism)
        self._instance.opt_by_optimism(optimism)

    def opt_by_gradient(self, env):
        """Optimize by the gradient for all base-learners.

        Args:
            env (Environment): Environment at current round.

        Returns:
            tuple: tuple contains:
                loss (float): the origin loss of all alive base-learners. \n
                surrogate_loss (float): the surrogate loss of all alive base-learners.
        """
        loss = np.zeros(len(self.active_index))
        surrogate_loss = np.zeros_like(loss)
        for i, idx in enumerate(self.active_index):
            _, loss[i], surrogate_loss[i] = self.bases[idx].opt_by_gradient(
                env)
        _, instance_loss, _ = self._instance.opt_by_gradient(env)
        reinit_instance = self.cover.set_instance_loss(instance_loss)
        if reinit_instance:
            self._instance.reinit()
        return loss, surrogate_loss

    @property
    def t(self):
        """Set the number of current round, get active state from ``cover`` and
        reinitialize the base-learners whose active state is 2.
        """
        return self._t

    @t.setter
    def t(self, t):
        self._t = t
        self.cover.t = t
        self.active_state = self.cover.active_state
        self.active_index = np.where(self.active_state > 0)[0]
        self.reinit_bases()

    @property
    def x_active_bases(self):
        """Get the decisions of all alive base-learners.

        Returns:
            numpy.ndarray: Decisions of all alive base learners.
        """
        return np.array([self.bases[i].x for i in self.active_index])

    @property
    def optimism(self):
        """Get the optimisms of all alive base-learners.

        Returns:
            numpy.ndarray: Optimisms of all alive base learners.
        """
        self._optimism = np.zeros_like(self.x_active_bases)
        for i, idx in enumerate(self.active_index):
            self._optimism[i] = self.bases[idx].optimism
        return self._optimism

    def reinit_bases(self):
        """Reinitialize the base-learners whose active state is 2."""
        reinit_idx = np.where(self.active_state == 2)[0]
        for idx in reinit_idx:
            self.bases[idx].reinit()
