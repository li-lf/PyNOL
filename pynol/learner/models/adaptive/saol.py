from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain
from pynol.learner.base import SOGD
from pynol.learner.meta import Prod
from pynol.learner.models.model import Model
from pynol.learner.schedule.cover import GC
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import StepSizeFreeSSP


class SAOL(Model):
    """Implementation of Strongly Adaptive Online Learning.

    ``SAOL`` is the online algorithm aiming to optimize strongly adaptive
    regret. It can leverage small-regret online algorithm and enjoy the strongly
    adaptive regret in order of :math:`R(|I|)+\mathcal{O}(\sqrt{|I|}\log T)` on
    each interval :math:`I \in [T]`, which :math:`|I|` is the length of the
    interval :math:`I`.

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
        http://proceedings.mlr.press/v37/daniely15.html
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 alive_time_threshold: Optional[int] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        N = int(np.ceil(np.log2(T)))
        ssp = StepSizeFreeSSP(
            SOGD, num_bases=N, domain=domain, prior=prior, seed=seed)
        cover = GC(N, alive_time_threshold)
        schedule = Schedule(ssp, cover)
        lr = np.minimum(1 / 2, 1 / np.sqrt(np.exp2(np.arange(N))))
        meta = Prod(N=N, lr=lr[None, :])
        super().__init__(schedule, meta)
