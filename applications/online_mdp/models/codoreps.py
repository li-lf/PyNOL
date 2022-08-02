from copy import deepcopy
from functools import reduce
from typing import Optional, Union

import numpy as np
from pynol.learner.base import Hedge, OptimisticHedge
from pynol.learner.meta import MSMWC
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import SSP, DiscreteSSP
from pynol.learner.specification.optimism_base import EnvironmentalOptimismBase
from pynol.learner.specification.optimism_meta import InnerOptimismMeta
from pynol.learner.specification.surrogate_base import InnerSurrogateBase


class AdaOREPS(Model):

    def __init__(self,
                 domain,
                 K,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        horizon_pool = DiscreteSSP.discretize(domain.diameter, K, grid=8)
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
        lr = 2 / np.sqrt(horizon_pool * domain.diameter * K)
        meta = MSMWC(prob=lr**2 / np.sum(lr**2), lr=lr[None, :])
        surrogate_base = InnerSurrogateBase()
        super().__init__(schedule, meta, surrogate_base=surrogate_base)


class CDOREPS(Model):

    def __init__(self,
                 domain,
                 K,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):

        horizon_pool = DiscreteSSP.discretize(domain.diameter, K, grid=8)
        ssp = SSP()
        lr = []
        for horizon in horizon_pool:
            domain_base = deepcopy(domain)
            domain_base.horizon = horizon
            new_ssp = DiscreteSSP(
                Hedge,
                min_step_size=(horizon / domain.diameter / K)**0.5,
                max_step_size=1,
                grid=2,
                domain=domain_base,
                correct=True,
                prior=prior,
                seed=seed)
            lr.append(2 * new_ssp.step_pool / horizon)
            ssp += new_ssp
        lr = reduce(np.append, lr)
        schedule = Schedule(ssp)
        meta = MSMWC(prob=lr**2 / np.sum(lr**2), lr=lr[None, :])
        surrogate_base = InnerSurrogateBase()
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base)


class OptimisticCDOREPS(Model):

    def __init__(self,
                 domain,
                 K,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        horizon_pool = DiscreteSSP.discretize(domain.diameter, K, grid=8)
        ssp = SSP()
        lr = []
        for horizon in horizon_pool:
            domain_base = deepcopy(domain)
            domain_base.horizon = horizon
            new_ssp_1 = DiscreteSSP(
                OptimisticHedge,
                min_step_size=(horizon / domain.diameter / K)**0.5,
                max_step_size=1,
                grid=2,
                domain=domain_base,
                optimism_type=None,
                correct=True,
                prior=prior,
                seed=seed)
            new_ssp_2 = DiscreteSSP(
                OptimisticHedge,
                min_step_size=(horizon / domain.diameter / K)**0.5,
                max_step_size=1,
                grid=2,
                domain=domain_base,
                optimism_type='external',
                correct=True,
                prior=prior,
                seed=seed)
            lr.append(2 * new_ssp_1.step_pool / horizon)
            lr.append(2 * new_ssp_2.step_pool / horizon)
            ssp += new_ssp_1
            ssp += new_ssp_2
        lr = reduce(np.append, lr)
        schedule = Schedule(ssp)
        meta = MSMWC(prob=lr**2 / np.sum(lr**2), lr=lr[None, :])
        surrogate_base = InnerSurrogateBase()
        optimism_base = EnvironmentalOptimismBase()
        optimism_meta = InnerOptimismMeta()
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            optimism_base=optimism_base,
            optimism_meta=optimism_meta)
