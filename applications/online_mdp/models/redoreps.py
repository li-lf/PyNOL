from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Simplex
from pynol.learner.base import Hedge as BaseHedge
from pynol.learner.meta import Hedge as MetaHedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.surrogate_base import InnerSurrogateBase
from pynol.learner.specification.surrogate_meta import \
    InnerSwitchingSurrogateMeta


class REDOREPS(Model):

    def __init__(self,
                 domain,
                 T,
                 penalty,
                 min_step_size=None,
                 max_step_size=None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        if min_step_size is None:
            min_step_size = T**(-0.5)
        if max_step_size is None:
            max_step_size = 1.
        ssp = DiscreteSSP(
            BaseHedge,
            min_step_size,
            max_step_size,
            domain=domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = np.array([(t + 1)**(-0.5) for t in range(T)])
        meta = MetaHedge(Simplex(len(ssp)).init_x('uniform'), lr=lr)
        surrogate_base = InnerSurrogateBase()
        surrogate_meta = InnerSwitchingSurrogateMeta(
            penalty=penalty, norm=1, order=1)
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta)
