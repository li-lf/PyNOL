from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain
from pynol.learner.base import SOGD
from pynol.learner.meta import AdaNormalHedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.cover import CGC, PCGC
from pynol.learner.schedule.schedule import PSchedule, Schedule
from pynol.learner.schedule.ssp import StepSizeFreeSSP


class SACS(Model):
    """Implementation of Adaptive Regret of Convex and Smooth Functions.

    ``SACS`` exploits the smoothness condition to achieve small-loss strongly
    adaptive regret of order :math:`\mathcal{O}(\sqrt{L_r^s \log s \cdot
    \log(s-r)})`, where :math:`L_r^s = \min_{x\in \mathcal{X}}\sum_{t=r}^s
    f_t(x)`. This bound could be much smaller than
    :math:`\mathcal{O}(\sqrt{(s-r) \log s})` when :math:`L_s^r` is small.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        alive_time_threshold (int, optional): Minimal alive time for base-learners.
            All base-learners whose alive time are less than ``alive_time_threshold``
            will not be activated.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

    References:
        http://proceedings.mlr.press/v97/zhang19j/zhang19j.pdf

    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 alive_time_threshold: Optional[int] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        N = int(np.ceil(np.log2(T + 1)))
        ssp = StepSizeFreeSSP(
            SOGD, num_bases=N, domain=domain, prior=prior, seed=seed)
        cover = CGC(N, alive_time_threshold)
        meta = AdaNormalHedge(N=N)
        schedule = Schedule(ssp, cover)
        super().__init__(schedule, meta)


class PSACS(Model):
    """Implementation of Adaptive Regret of Convex and Smooth Functions.

    ``PSACS`` further improves the performance by
    problem-dependent intervals and obtains strongly adaptive regret of order
    :math:`\mathcal{O}(\sqrt{L_r^s \log L_1^s \cdot L_r^s})`, where :math:`L_r^s
    = \min_{x\in \mathcal{X}}\sum_{t=r}^s f_t(x)`.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        loss_threshold (float, optional): A new interval is activated only when
            the cumulative loss of the additional benchmark base-learner is larger
            than the ``loss_threshold``.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 loss_threshold: float = 1.,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        N = int(np.ceil(np.log2(T + 1)))
        ssp = StepSizeFreeSSP(
            SOGD, num_bases=N, domain=domain, prior=prior, seed=seed)
        meta = AdaNormalHedge(N=N)
        cover = PCGC(N, loss_threshold)
        schedule = PSchedule(ssp, cover)
        super().__init__(schedule, meta)
