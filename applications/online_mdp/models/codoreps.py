from copy import deepcopy
from functools import reduce
from typing import Optional, Union

import numpy as np
from pynol.learner.base import Hedge, OptimisticHedge
from pynol.learner.meta import MSMWC
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import SSP, DiscreteSSP
from pynol.learner.specification.optimism_base import (
    EnvironmentalOptimismBase, LastGradOptimismBase)
from pynol.learner.specification.optimism_meta import InnerOptimismMeta
from pynol.learner.specification.surrogate_base import InnerSurrogateBase


class AdaOREPS(Model):

    def __init__(self,
                 domain,
                 K,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        horizon_pool = DiscreteSSP.discretize(domain.diameter, K)
        bases = []
        for horizon in horizon_pool:
            domain_base = deepcopy(domain)
            domain_base.horizon = horizon
            bases.append(
                Hedge(
                    domain=domain_base,
                    step_size=(horizon / domain.diameter / K)**0.5,
                    correct=True,
                    prior=prior,
                    seed=seed))
        ssp = SSP(bases=bases)
        schedule = Schedule(ssp)
        lr = 1 / np.sqrt(horizon_pool * domain.diameter * K)
        lr = lr * (1 / max(lr))
        meta = MSMWC(prob=np.ones(len(lr)) / len(lr), lr=lr[None, :])
        surrogate_base = InnerSurrogateBase()
        super().__init__(schedule, meta, surrogate_base=surrogate_base)


class OptimisticCDOREPS(Model):

    def __init__(self,
                 domain,
                 K,
                 optimism_type='external',
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        horizon_pool = DiscreteSSP.discretize(domain.diameter, K)
        ssp = SSP()
        lr = []
        for horizon in horizon_pool:
            domain_base = deepcopy(domain)
            domain_base.horizon = horizon
            new_ssp = DiscreteSSP(
                OptimisticHedge,
                min_step_size=(horizon / domain.diameter / K)**0.5,
                max_step_size=1,
                grid=2,
                domain=domain_base,
                optimism_type=optimism_type,
                correct=True,
                prior=prior,
                seed=seed)
            lr.append(new_ssp.step_pool / horizon)
            ssp += new_ssp
        lr = reduce(np.append, lr)
        lr = lr * (1 / max(lr))
        schedule = Schedule(ssp)
        meta = MSMWC(prob=np.ones(len(lr)) / len(lr), lr=lr[None, :])
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


class CDOREPS(OptimisticCDOREPS):

    def __init__(self,
                 domain,
                 K,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, K, None, prior, seed)
