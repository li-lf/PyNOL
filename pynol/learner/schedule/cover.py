from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Cover(ABC):
    """The abstract class for problem-independent cover.

    Args:
        active_state (numpy.ndarray): Initial active state.
        alive_time_threshold (int, optional): Minimal interval length for cover.
            All intervals whose length are less than ``alive_time_threshold``
            will not be activated.
    """

    def __init__(self,
                 active_state: np.ndarray,
                 alive_time_threshold: Optional[int] = None):
        self._t = 0
        self._active_state = active_state
        self._alive_time_threshold = alive_time_threshold

    @property
    def t(self):
        """Set the number of current round and compute the active state at
        current round.
        """

        return self._t

    @t.setter
    @abstractmethod
    def t(self):
        pass

    @property
    def active_state(self):
        """Get the active state of base-learners:

        - 0: sleep
        - 1: active at the current round and the previous round.
        - 2: active at the current round and sleep at the previous round.
        """

        if self._alive_time_threshold is None:
            return self._active_state
        else:
            return self.check_threshold()

    def check_threshold(self):
        """Check the interval length. All intervals whose length are less than
        ``alive_time_threshold`` will not be activated.
        """
        threshold_idx = int(np.ceil(np.log2(self._alive_time_threshold)))
        active_state = self._active_state.copy()
        active_state[:threshold_idx] = 0
        if not np.any(active_state[threshold_idx:] > 0):
            active_state[threshold_idx] = 2 if self.t == 0 else 1
        return active_state


class FullCover(Cover):
    """The cover that sets all base-learners alive all the time, which is used
    for dynamic algorithms.

    Args:
        N (int): Number of base-learners.
    """

    def __init__(self, N: int):
        super().__init__(np.ones(N), None)

    @Cover.t.setter
    def t(self, t):
        self._t = t


class GC(Cover):
    """Geometric cover is the classical interval partition method, which
    discretize the whole horizon into sub-intervals with exponential length.
    Below is an illustration.

    .. image:: ../_static/figures/GC.png

    Args:
        N (int): Number of base-learners. :math:`N=5` for the above figure.
        alive_time_threshold (int, optional): Minimal interval length for cover.
            All intervals whose length are less than ``alive_time_threshold``
            will not be activated.

    Below we give an example of ``GC`` with ``alive_time_threshold=4``:

    .. image:: ../_static/figures/threshold.png

    The base-learners whose alive time is less than 4 will not be activated,
    shown as gray blocks. Since there must be at least one base-learner, we
    initialize base-learner with ``Alive Time = 4`` at ``t=1``.
    """

    def __init__(self, N: int, alive_time_threshold: Optional[int] = None):
        super().__init__(np.zeros(N), alive_time_threshold)
        self._I_k_endtime = np.zeros(N)
        self._interval_length = np.exp2(np.arange(N))

    @Cover.t.setter
    def t(self, t):
        self._t = t
        # find active bases
        self._active_state[np.where(self._I_k_endtime > 0)] = 1

        # 2^(next_k - 1) <= t <= 2 ^ next_k
        next_k = int(np.ceil(np.log2(self._t + 1)))
        if self._t + 1 == 2**next_k:
            # the first end time is (i + 1) * 2^ next_k - 1, i = 1
            self._I_k_endtime[next_k] = (1 + 1) * 2**next_k - 1
            self._active_state[next_k] = 2

        # reinit bases
        to_init_bases = (self._I_k_endtime > 0) & (
            self._I_k_endtime < self._t + 1)
        self._active_state[to_init_bases] = 2

        # update_stepsize[i] = 2^i if entry i needs to reinit or update_stepsize[i] = 0
        update_stepsize = np.where(to_init_bases, self._interval_length, 0)
        self._I_k_endtime[to_init_bases] += update_stepsize[to_init_bases]

        if self._alive_time_threshold is not None:
            self.check_threshold()


class CGC(GC):
    """CGC stands for compact geometric cover which removes some redundant
    intervals from geometric cover while keeping the desired properties. Below is
    an illustration.

    .. image:: ../_static/figures/CGC.png

    Args:
        N (int): Number of base-learners. :math:`N=5` for the above figure.
        alive_time_threshold (int, optional): Minimal interval length for cover.
            All intervals whose length are less than ``alive_time_threshold``
            will not be activated.

    """

    def __init__(self, N: int, alive_time_threshold: Optional[int] = None):
        super().__init__(N, alive_time_threshold)

    @GC.t.setter
    def t(self, t):
        super(CGC, CGC).t.__set__(self, t)
        # inactive some bases
        inactive_bases = (self._I_k_endtime > 0) & ((
            (self._I_k_endtime + 1) / self._interval_length) % 2 != 0)
        self._active_state[inactive_bases] = 0


class PCover(ABC):
    """The abstract class for problem-dependent cover.

    Args:
        active_state (numpy.ndarray): Initial active state.
        loss_threshold (int, optional): A new interval is activated only when
            the cumulative loss of the additional benchmark base-learner is larger
            than the ``loss_threshold``.
    """

    def __init__(self, active_state: np.ndarray, loss_threshold: float = 2):
        self._t = 0
        self._active_state = active_state
        self._loss_threshold = loss_threshold

    @property
    def t(self):
        """Set the number of current round and compute the active state at
        current round.
        """
        return self._t

    @t.setter
    @abstractmethod
    def t(self):
        pass

    @property
    def active_state(self):
        """Get the active state of base-learners:

        - 0: sleep
        - 1: active at the current round and the previous round.
        - 2: active at the current round and sleep at the previous round.
        """
        return self._active_state

    @abstractmethod
    def set_instance_loss(self, loss: float):
        """Set the loss of the additional benchmark base-learner.

        Args:
            loss (float): Loss of the additional benchmark base-learner.

        Return:
            bool: Whether to activate a new interval.

        """
        raise NotImplementedError()


class PGC(PCover):
    """PGC stands for problem-dependent geometric cover, a
    problem-dependent version of geometric cover. Compared with geometric cover,
    it can achieve more adaptive results with proper base and meta-algorithms.
    Below is an illustration.

    .. image:: ../_static/figures/PGC.png

    Args:
        active_state (numpy.ndarray): Initial active state.
        loss_threshold (int, optional): A new interval is activated only when
            the cumulative loss of the additional benchmark base-learner is larger
            than the ``loss_threshold``.

    """

    def __init__(self, N: int, loss_threshold: float = 2):
        super().__init__(np.zeros(N), loss_threshold)
        self._cum_instance_loss = 0
        self._t_loss = 0
        self._interval_length = np.exp2(np.arange(N))
        self._I_k_restart_marker = np.exp2(np.arange(N))
        self._marker = 1

    def set_instance_loss(self, loss: float):
        self._t_loss = loss
        return True if loss + self._cum_instance_loss > self._loss_threshold else False

    def _get_t_marker(self):
        self._cum_instance_loss += self._t_loss
        if self._cum_instance_loss > self._loss_threshold:
            self._marker += 1
            self._cum_instance_loss = 0
        return self._marker

    @PCover.t.setter
    def t(self, t):
        self._t = t
        marker = self._get_t_marker()
        self._active_state[(marker >= self._interval_length)] = 1
        to_reinit = (self._I_k_restart_marker == marker)
        self._active_state[to_reinit] = 2
        self._I_k_restart_marker[to_reinit] += self._interval_length[to_reinit]


class PCGC(PGC):
    """PCGC stands for problem-dependent compact geometric cover
    which removes some redundant intervals from problem-dependent geometric
    cover while keeping the desired properties. Below is an illustration.

    .. image:: ../_static/figures/PCGC.png

    Args:
        active_state (numpy.ndarray): Initial active state.
        loss_threshold (int, optional): A new interval is activated only when
            the cumulative loss of the additional benchmark base-learner is larger
            than the ``loss_threshold``.

    """

    def __init__(self, N: int, loss_threshold: float = 2):
        super().__init__(N, loss_threshold)

    @PGC.t.setter
    def t(self, t):
        super(PCGC, PCGC).t.__set__(self, t)
        to_inactive = ((self._I_k_restart_marker / self._interval_length) % 2
                       != 0)
        self._active_state[to_inactive] = 0
