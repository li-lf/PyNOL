from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex
from pynol.learner.base import OGD
from pynol.learner.meta import Hedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.surrogate_base import LinearSurrogateBase
from pynol.learner.specification.surrogate_meta import SurrogateMetaFromBase


class Ader(Model):
    """Implementation of Adaptive Online Learning in Dynamic Environments.

    Ader is an online algorithm designed for optimizing dynamic regret for
    general convex online functions, which is shown to enjoy
    :math:`\mathcal{O}(\sqrt{T(1+P_T)})` dynamic regret.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        G (float): Upper bound of gradient.
        surrogate (bool): Whether to use surrogate loss.
        min_step_size (float): Minimal step size for the base-learners. It is
            set as the theory suggests by default.
        max_step_size (float): Maximal step size for the base-learners. It is
            set as the theory suggests by default.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

    References:
        https://proceedings.neurips.cc/paper/2018/file/10a5ab2db37feedfdeaab192ead4ac0e-Paper.pdf
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float,
                 surrogate: bool = True,
                 min_step_size: Optional[float] = None,
                 max_step_size: Optional[float] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D = 2 * domain.R
        if min_step_size is None:
            min_step_size = D / G * (7 / (2 * T))**0.5
        if max_step_size is None:
            max_step_size = D / G * (7 / (2 * T) + 2)**0.5
        ssp = DiscreteSSP(
            OGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = np.array([1 / (G * D * (t + 1)**0.5) for t in range(T)])
        meta = Hedge(prob=Simplex(len(ssp)).init_x(prior='nonuniform'), lr=lr)
        surrogate_base = LinearSurrogateBase() if surrogate is True else None
        surrogate_meta = SurrogateMetaFromBase() if surrogate is True else None
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta)
