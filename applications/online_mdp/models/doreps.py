from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Simplex
from pynol.learner.base import Hedge as BaseHedge
from pynol.learner.base import OptimisticHedge as BaseOptimisticHedge
from pynol.learner.meta import Hedge as MetaHedge
from pynol.learner.meta import OptimisticHedge as MetaOptimisticHedge
from pynol.learner.meta import OptimisticLR
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.optimism_base import (
    EnvironmentalOptimismBase, LastGradOptimismBase)
from pynol.learner.specification.optimism_meta import InnerOptimismMeta
from pynol.learner.specification.surrogate_base import InnerSurrogateBase


class DOREPS(Model):

    def __init__(self,
                 domain,
                 K,
                 min_step_size=None,
                 max_step_size=None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        if min_step_size is None:
            min_step_size = K**(-0.5)
        if max_step_size is None:
            max_step_size = 1.
        ssp = DiscreteSSP(
            BaseHedge,
            min_step_size,
            max_step_size,
            domain=domain,
            grid=2,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = K**(-0.5)
        surrogate_base = InnerSurrogateBase()
        meta = MetaHedge(Simplex(len(ssp)).init_x(prior='uniform'), lr=lr)
        super().__init__(schedule, meta, surrogate_base=surrogate_base)


class OptimisticDOREPS(Model):

    def __init__(self,
                 domain,
                 K,
                 min_step_size=None,
                 max_step_size=None,
                 optimism_type='external',
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        if min_step_size is None:
            min_step_size = K**(-0.5)
        if max_step_size is None:
            max_step_size = K**0.5
        ssp = DiscreteSSP(
            BaseOptimisticHedge,
            min_step_size,
            max_step_size,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = OptimisticLR(scale=1, norm=np.inf)
        meta = MetaOptimisticHedge(
            Simplex(len(ssp)).init_x(prior='uniform'), lr=lr)
        surrogate_base = InnerSurrogateBase()
        if optimism_type is None:
            optimism_base = None
        elif optimism_type == 'external':
            optimism_base = EnvironmentalOptimismBase()
        elif optimism_type == 'last_grad':
            optimism_base = LastGradOptimismBase()
        else:
            raise TypeError(f'{optimism_type} is not defined.')
        optimism_meta = None if optimism_type is None else InnerOptimismMeta()
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            optimism_base=optimism_base,
            optimism_meta=optimism_meta)
