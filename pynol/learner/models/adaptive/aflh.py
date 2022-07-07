from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain
from pynol.learner.base import SOGD
from pynol.learner.meta import AFLHMeta
from pynol.learner.models.model import Model
from pynol.learner.schedule.cover import CGC
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import StepSizeFreeSSP
from pynol.learner.specification.surrogate_base import LinearSurrogateBase
from pynol.learner.specification.surrogate_meta import SurrogateMetaFromBase


class AFLH(Model):
    """Implementation of Adaptive Algorithms for Online Decision Problems and
    Minimizing Adaptive Regret with One Gradient per Iteration.

    ``AFLH`` is an online algorithm optimizing the weak adaptive regret. With
    its neat meta-based design, for convex functions AFLH achieves regret of
    order :math:`R(T)+\mathcal{O}(\sqrt{T}\log T)` on each interval :math:`I\in[T]`,
    which :math:`R(T)` is the regret of any black-box base algorithm and
    :math:`\mathcal{O}(\sqrt{T}\log T)` is the regret overhead incurred by the
    meta-algorithm.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        lr (float, numpy.ndarray, optional): Learning rate for meta-algorithm.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

    References:
        https://dominoweb.draco.res.ibm.com/reports/rj10418.pdf \n
        https://www.ijcai.org/proceedings/2018/0383.pdf
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 lr: Optional[Union[float, np.ndarray]] = None,
                 surrogate: bool = True,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        N = int(np.ceil(np.log2(T + 1)))
        lr = lr if lr is not None else T**(-0.5)
        ssp = StepSizeFreeSSP(
            SOGD, num_bases=N, domain=domain, prior=prior, seed=seed)
        cover = CGC(N)
        meta = AFLHMeta(N=N, lr=lr)
        schedule = Schedule(ssp, cover)
        if surrogate is True:
            surrogate_base = LinearSurrogateBase()
            surrogate_meta = SurrogateMetaFromBase()
        else:
            surrogate_base, surrogate_meta = None, None
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta)
