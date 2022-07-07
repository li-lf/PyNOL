from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex
from pynol.learner.base import OGD
from pynol.learner.meta import Hedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.surrogate_base import InnerSurrogateBase
from pynol.learner.specification.surrogate_meta import \
    InnerSwitchingSurrogateMeta


class Scream(Model):
    """Implementation of Non-stationary Online Learning with Memory and
    Non-stochastic Control.

    ``Scream`` is designed the optimize the dynamic regret for online convex
    optimization with switching cost. In the technical aspect, it neatly solves
    the challenge of controlling the switching cost in the meta-base two-layer
    structure while enjoying the desired dynamic regret bound. It is shown to
    enjoy :math:`\mathcal{O}(\sqrt{T(1+P_T)})` dynamic regret with switching
    cost.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        G (float): Upper bound of gradient.
        penalty (float): Penalty coefficient of switching cost.
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
        https://proceedings.mlr.press/v151/zhao22a/zhao22a.pdf
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float,
                 penalty: float,
                 min_step_size: Optional[float] = None,
                 max_step_size: Optional[float] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D = 2 * domain.R
        if min_step_size is None:
            min_step_size = (D**2 / (G * (penalty + G) * T))**0.5
        if max_step_size is None:
            max_step_size = ((D**2 * (1 + 2 * T)) / (G *
                                                     (penalty + G) * T))**0.5
        ssp = DiscreteSSP(
            OGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = np.array([
            (2 / ((2 * penalty + G) * (penalty + G) * D**2 * (t + 1)))**0.5
            for t in range(T)
        ])
        meta = Hedge(prob=Simplex(len(ssp)).init_x(prior='nonuniform'), lr=lr)
        surrogate_base = InnerSurrogateBase()
        surrogate_meta = InnerSwitchingSurrogateMeta(
            penalty=penalty, norm=2, order=1)
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta)
