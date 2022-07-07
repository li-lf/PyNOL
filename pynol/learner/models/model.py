from typing import Optional
import numpy as np
from pynol.environment.environment import Environment
from pynol.learner.meta import Meta
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.specification.optimism_base import OptimismBase
from pynol.learner.specification.optimism_meta import OptimismMeta
from pynol.learner.specification.perturbation import Perturbation
from pynol.learner.specification.surrogate_base import SurrogateBase
from pynol.learner.specification.surrogate_meta import SurrogateMeta


class Model:
    """Combines several main components into the final algorithm.

    Args:
        schedule (Schedule): Schedule method, refer to
            ``pynol.schedule.schedule``.
        meta (Meta): Meta algorithm, refer to ``pynol.meta``.
        surrogate_base (SurrogateBase): Surrogate loss class for base-learners,
            refer to ``pynol.specification.surrogate_base``.
        optimism_base (optimismBase): Optimism class for base-learners,
            refer to ``pynol.specification.optimism_base``.
        surrogate_meta (SurrogateMeta): Surrogate loss class for meta-algorithm,
            refer to ``pynol.specification.surrogate_meta``.
        optimism_base (OptimismBase): Optimism class for meta-algorithm,
            refer to ``pynol.specification.optimism_meta``.
        perturbation (Perturbation): Perturbation class used in bandit setting.
    """

    def __init__(self,
                 schedule: Schedule,
                 meta: Meta,
                 surrogate_base: SurrogateBase = None,
                 optimism_base: OptimismBase = None,
                 surrogate_meta: SurrogateMeta = None,
                 optimism_meta: OptimismMeta = None,
                 perturbation: Perturbation = None) -> None:
        self.schedule = schedule
        self.meta = meta
        self.surrogate_base = surrogate_base
        self.optimism_base = optimism_base
        self.surrogate_meta = surrogate_meta
        self.optimism_meta = optimism_meta
        self.perturbation = perturbation
        self.t = 0
        self.internal_optimism_base = None
        self.internal_optimism_meta = None

    def opt(self, env: Environment):
        """The optimization process of the base algorithm.

        All algorithms are divided into two parts:
        :meth:`~pynol.learner.models.Model.opt_by_optimism` at the beginning of
        current round and :meth:`~pynol.learner.models.Model.opt_by_gradient`
        at the end of current round.

        Args:
            env (Environment): Environment at the current round.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): Decision at the current round. \n
                loss (float): Origin loss at the current round. \n
                surrogate_loss (float): the surrogate loss at the current round.
        """
        self.opt_by_optimism(env.optimism)
        return self.opt_by_gradient(env)

    def opt_by_optimism(self, optimism: Optional[np.ndarray]):
        """Optimize by the optimism.

        Args:
            optimism (numpy.ndarray, optional): External optimism at the beginning of the
                current round.

        Returns:
            None
        """
        self.optimism_env = optimism
        variables = vars(self)
        self.schedule.t = self.t
        self.meta.active_state = self.schedule.active_state
        if self.optimism_base is not None and self.optimism_base.is_external:
            optimism_base = self.optimism_base.compute_optimism_base(variables)
        else:
            optimism_base = self.internal_optimism_base
        self.schedule.opt_by_optimism(optimism_base)
        if self.optimism_meta is not None and self.optimism_meta.is_external:
            optimism_meta = self.optimism_meta.compute_optimism_meta(variables)
        else:
            optimism_meta = self.internal_optimism_meta
        self.meta.opt_by_optimism(optimism_meta)

    def opt_by_gradient(self, env: Environment):
        """Optimize by the true gradient.

        Args:
            env (Environment): Environment at the current round.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): the decision at the current round. \n
                loss (float): the origin loss at the current round. \n
                surrogate_loss (float): the surrogate loss at the current round.
        """
        self.env = env
        variables = vars(self)
        self.x_bases = self.schedule.x_active_bases
        self.x = np.dot(self.meta.prob, self.x_bases)

        if env.full_info:
            loss, surrogate_loss = env.get_loss(self.x)
            self.grad = env.get_grad(self.x)
        else:
            self.perturbation.perturb_x(self.x)
            loss, surrogate_loss = self.perturbation.compute_loss(env)
            self.grad = self.perturbation.construct_grad()

        # update bases
        base_env = Environment(func=env.func, grad_func=env.grad_func)
        if self.surrogate_base is not None:
            base_env.surrogate_func, base_env.surrogate_grad = self.surrogate_base.compute_surrogate_base(
                variables)

        self.loss_bases, self.surrogate_loss_bases = self.schedule.opt_by_gradient(
            base_env)

        # compute surrogate loss of meta #
        if self.surrogate_meta is not None:
            self.loss_bases = self.surrogate_meta.compute_surrogate_meta(
                variables)

        # update meta
        self.meta.opt_by_gradient(self.loss_bases, loss)

        # compute internal optimism of bases
        self.compute_internal_optimism(variables)

        self.t += 1
        return self.x, loss, surrogate_loss

    def compute_internal_optimism(self, variables):
        """Compute the internal optimism.

        Args:
            variables (dict): Intermediate variables at the current round.
        """
        if self.optimism_base is not None and self.optimism_base.is_external is False:
            self.internal_optimism_base = self.optimism_base.compute_optimism_base(
                variables)
        if self.optimism_meta is not None and self.optimism_meta.is_external is False:
            self.internal_optimism_meta = self.optimism_meta.compute_optimism_meta(
                variables)
