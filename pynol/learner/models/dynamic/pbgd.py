from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex
from pynol.learner.base import OGD
from pynol.learner.meta import Hedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.perturbation import (OnePointPerturbation,
                                                      TwoPointPerturbation)
from pynol.learner.specification.surrogate_base import LinearSurrogateBase
from pynol.learner.specification.surrogate_meta import SurrogateMetaFromBase


class PBGDOnePoint(Model):
    """Implementation of Bandit Convex Optimization in Non-stationary
    Environments.

    ``PBGDOnePoint`` stands for Parameter-free Bandit Gradient Descent with
    One-Point feedback, which is a parameter-free algorithm for optimizing
    dynamic regret for general convex function with one-point feedback. Is is
    shown to enjoy :math:`\mathcal{O}(T^{\\frac{3}{4}}(1+P_T)^{\\frac{1}{2}})`
    dynamic regret.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        C (float): Upper bound of loss value.
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
        https://www.jmlr.org/papers/volume22/20-763/20-763.pdf
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 C: float,
                 min_step_size: Optional[float] = None,
                 max_step_size: Optional[float] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D, R, r = 2 * domain.R, domain.R, domain.r
        scale_perturb = min(r / 2, domain.dimension**0.5 * R * T**(-0.25))
        shrink = scale_perturb / r
        G_estimate = domain.dimension * C / scale_perturb
        if min_step_size is None:
            min_step_size = R * (7 * R**2)**0.5 / (
                domain.dimension**0.5 * C * T**0.75)
        if max_step_size is None:
            max_step_size = R * (7 * R**2 + 2 * R**2 * T)**0.5 / (
                domain.dimension**0.5 * C * T**0.75)
        ssp = DiscreteSSP(
            OGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=(1 - shrink) * domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = np.array(
            [2 / ((t + 1) * G_estimate**2 * D**2)**0.5 for t in range(T)])
        meta = Hedge(prob=Simplex(len(ssp)).init_x(prior='nonuniform'), lr=lr)
        perturbation = OnePointPerturbation(domain, scale_perturb)
        surrogate_base = LinearSurrogateBase()
        surrogate_meta = SurrogateMetaFromBase()
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta,
            perturbation=perturbation)


class PBGDTwoPoint(Model):
    """Implementation of Bandit Convex Optimization in Non-stationary
    Environments.

    ``PBGDTwoPoint`` stands for Parameter-free Bandit Gradient Descent with
    Two-Point feedback, which is a parameter-free algorithm for optimizing
    dynamic regret for general convex function with two-point feedback. Is is
    shown to enjoy :math:`\mathcal{O}(\sqrt{T(1+P_T)})` dynamic regret.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        L_lipschitz (float): Lipschitz constant for loss functions.
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
                 L_lipschitz: float,
                 min_step_size: Optional[float] = None,
                 max_step_size: Optional[float] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D, R, r = 2 * domain.R, domain.R, domain.r
        scale_perturb = min(r / 2, domain.dimension * R / T**0.5)
        shrink = scale_perturb / r
        G_estimate = L_lipschitz * domain.dimension
        if min_step_size is None:
            min_step_size = (
                (7 * R**2) /
                (2 * L_lipschitz**2 * domain.dimension**2 * T))**0.5
        if max_step_size is None:
            max_step_size = (
                (7 * R**2 + 2 * R**2 * T) /
                (2 * L_lipschitz**2 * domain.dimension**2 * T))**0.5
        ssp = DiscreteSSP(
            OGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=(1 - shrink) * domain,
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        lr = np.array(
            [2 / ((t + 1) * G_estimate**2 * D**2)**0.5 for t in range(T)])
        meta = Hedge(prob=Simplex(len(ssp)).init_x(prior='nonuniform'), lr=lr)
        perturbation = TwoPointPerturbation(domain, scale_perturb)
        surrogate_base = LinearSurrogateBase()
        surrogate_meta = SurrogateMetaFromBase()
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta,
            perturbation=perturbation)
