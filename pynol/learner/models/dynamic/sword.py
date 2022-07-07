from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain
from pynol.learner.base import OEGD, OGD
from pynol.learner.meta import OptimisticHedge, OptimisticLR
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.optimism_meta import (
    SwordBestOptimismMeta, SwordVariationOptimismMeta)
from pynol.learner.specification.surrogate_meta import InnerSurrogateMeta


class SwordVariation(Model):
    """``SwordVariation`` enjoys a gradient-variation dynamic regret bound of
    :math:`\mathcal{O}(\sqrt{(1 + P_T + V_T)(1 + P_T)})`, where
    :math:`V_{T}=\sum_{t=2}^{T} \sup_{\mathbf{x} \in \mathcal{X}}\left\|\\nabla
    f_{t-1}(\mathbf{x})-\\nabla f_{t}(\mathbf{x})\\right\|_{2}^{2}` is the
    gradient variation of online functions.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        G (float): Upper bound of gradient.
        L_smooth (float): Smooth constant of online loss functions
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
        https://proceedings.neurips.cc/paper/2020/file/939314105ce8701e67489642ef4d49e8-Paper.pdf

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
            min_step_size = (D**2 / (8 * G**2 * T))**0.5
        if max_step_size is None:
            max_step_size = 1 / (4 * L_smooth)
        ssp = DiscreteSSP(
            OEGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = OptimisticLR(upper_bound=1 / (8 * D**2 * L_smooth))
        meta = OptimisticHedge(prob=np.ones(len(ssp)) / len(ssp), lr=lr)
        surrogate_meta = InnerSurrogateMeta()
        optimism_meta = SwordVariationOptimismMeta()
        super().__init__(
            schedule,
            meta,
            surrogate_meta=surrogate_meta,
            optimism_meta=optimism_meta)


class SwordSmallLoss(Model):
    """``SwordSmallLoss`` enjoys a small-loss dynamic regret bound of
    :math:`\mathcal{O}(\sqrt{(1 + P_T + F_T)(1 + P_T)})`, where
    :math:`F_T =\sum_{t=1}^T f_t(u_t)` is  the cumulative loss of the comparator
    sequence.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        G (float): Upper bound of gradient.
        L_smooth (float): Smooth constant of online loss functions
        min_step_size (float): Minimal step size for the base-learners. It is
            set as the theory suggests by default.
        max_step_size (float): Maximal step size for the base-learners. It is
            set as the theory suggests by default.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

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
            min_step_size = (D / (16 * L_smooth * G * T))**0.5
        if max_step_size is None:
            max_step_size = 1 / (4 * L_smooth)
        ssp = DiscreteSSP(
            OGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = OptimisticLR(upper_bound=1 / (8 * D**2 * L_smooth))
        meta = OptimisticHedge(prob=np.ones(len(ssp)) / len(ssp), lr=lr)
        surrogate_meta = InnerSurrogateMeta()
        super().__init__(schedule, meta, surrogate_meta=surrogate_meta)


class SwordBest(Model):
    """``SwordBest`` enjoys a best-of-both-worlds dynamic regret bound of
    :math:`\mathcal{O}(\sqrt{(1 + P_T + \min\{V_T, F_T\})(1 + P_T)})`, where
    :math:`V_{T}=\sum_{t=2}^{T} \sup_{\mathbf{x} \in \mathcal{X}}\left\|\\nabla
    f_{t-1}(\mathbf{x})-\\nabla f_{t}(\mathbf{x})\\right\|_{2}^{2}` is the
    gradient variation of online functions, :math:`F_T =\sum_{t=1}^T f_t(u_t)`
    is  the cumulative loss of the comparator sequence.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        G (float): Upper bound of gradient.
        L_smooth (float): Smooth constant of online loss functions
        min_step_size (float): Minimal step size for the base-learners. It is
            set as the theory suggests by default.
        max_step_size (float): Maximal step size for the base-learners. It is
            set as the theory suggests by default.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

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
        min_step_size_OEGD = (
            D**2 /
            (8 * G**2 * T))**0.5 if min_step_size is None else min_step_size
        max_step_size_OEGD = 1 / (
            4 * L_smooth) if max_step_size is None else max_step_size
        ssp_OEGD = DiscreteSSP(
            OEGD,
            min_step_size_OEGD,
            max_step_size_OEGD,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)

        min_step_size_OGD = (D / (16 * L_smooth * G * T)
                             )**0.5 if min_step_size is None else min_step_size
        max_step_size_OGD = 1 / (
            4 * L_smooth) if max_step_size is None else max_step_size
        ssp_OGD = DiscreteSSP(
            OGD,
            min_step_size_OGD,
            max_step_size_OGD,
            grid=2,
            domain=domain,
            prior=prior,
            seed=seed)

        ssp = ssp_OEGD + ssp_OGD
        schedule = Schedule(ssp)
        lr = OptimisticLR(upper_bound=1 / (8 * D**2 * L_smooth))
        meta = OptimisticHedge(prob=np.ones(len(ssp)) / len(ssp), lr=lr)
        surrogate_meta = InnerSurrogateMeta()
        optimism_meta = SwordBestOptimismMeta()
        super().__init__(
            schedule,
            meta,
            surrogate_meta=surrogate_meta,
            optimism_meta=optimism_meta)
