from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain
from pynol.learner.base import OptimisticOGD
from pynol.learner.meta import OptimisticHedge, OptimisticLR
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.optimism_base import LastGradOptimismBase
from pynol.learner.specification.optimism_meta import \
    InnerSwitchingOptimismMeta
from pynol.learner.specification.surrogate_base import InnerSurrogateBase
from pynol.learner.specification.surrogate_meta import \
    InnerSwitchingSurrogateMeta


class SwordPP(Model):
    """Implementation of Adaptivity and Non-stationarity: Problem-dependent
    Dynamic Regret for Online Convex Optimization.

    ``Swordpp`` is an improved version of ``Sword``, who reduces the gradient
    query complexity of each round from :math:`\mathcal{O}(\log T)` to :math:`1`
    and achieves the best-of-both-worlds dynamic regret bounds by a single
    algorithm.

    References:
        https://arxiv.org/abs/2112.14368
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float,
                 L_smooth: float,
                 min_step_size: Optional[float] = None,
                 max_step_size: Optional[float] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D = 2 * domain.R
        if min_step_size is None:
            min_step_size = (D**2 / (G**2 * T))**0.5
        if max_step_size is None:
            max_step_size = 1 / (8 * L_smooth)
        ssp = DiscreteSSP(
            OptimisticOGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = OptimisticLR(upper_bound=1 / (8 * D**2 * L_smooth))
        meta = OptimisticHedge(prob=np.ones(len(ssp)) / len(ssp), lr=lr)
        surrogate_base = InnerSurrogateBase()
        optimism_base = LastGradOptimismBase()
        penalty = 2 * L_smooth
        surrogate_meta = InnerSwitchingSurrogateMeta(
            penalty=penalty, norm=2, order=2)
        optimism_meta = InnerSwitchingOptimismMeta(
            penalty=penalty, norm=2, order=2)
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta,
            optimism_base=optimism_base,
            optimism_meta=optimism_meta)
