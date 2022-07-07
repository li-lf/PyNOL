from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain
from pynol.environment.environment import Environment


class Base(ABC):
    """An abstract class for base algorithms.

    Args:
        domain (Domain): Feasible set for the base algorithm.
            step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round $t$.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, np.ndarray],
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        self.domain = domain
        self.step_size = step_size
        self.prior = prior
        self.seed = seed
        self.x = self.domain.init_x(prior, seed)
        self.t = 0

    def opt(self, env: Environment):
        """The optimization process of the base algorithm.

        All base algorithms are divided into two parts:
        :meth:`~pynol.learner.base.Base.opt_by_optimism` at the beginning of
        current round and :meth:`~pynol.learner.base.Base.opt_by_gradient` at the
        end of current round.

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

    def get_step_size(self):
        """Get the step size at each round.

        Returns:
            float: Step size at the current round.
        """
        return self.step_size[self.t] if hasattr(self.step_size,
                                                 '__len__') else self.step_size

    def opt_by_optimism(self, optimism: Optional[np.ndarray] = None):
        """Optimize by the optimism.

        Args:
            optimism (numpy.ndarray, optional): External optimism at the beginning of the
                current round.

        Returns:
            None
        """
        pass

    @abstractmethod
    def opt_by_gradient(self, env: Environment):
        """Optimize by the true gradient.

        All base-algorithms are required to override this method to implement
        their own optimization process.

        Args:
            env (Environment): Environment at the current round.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): the decision at the current round. \n
                loss (float): the origin loss at the current round. \n
                surrogate_loss (float): the surrogate loss at the current round.
        """
        pass

    def reinit(self):
        """Reset the base algorithm to the initial state, which is used for
        adaptive algorithms.
        """
        self.__init__(self.domain, self.step_size, self.prior, self.seed)


class OGD(Base):
    """Implementation of Online Gradient Descent.

    ``OGD`` stands for Online Gradient Descent, the most popular algorithm for
    online learning. `OGD` updates the decision :math:`x_{t+1}` by

    .. math::

        x_{t+1} = \Pi_{\mathcal{X}} [x_t - \eta_t \\nabla f_t(x_t)]

    where :math:`\eta_t > 0` is the step size at round `t`, and
    :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the projection onto the nearest
    point in :math:`\mathcal{X}`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round $t$.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, np.ndarray],
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, prior, seed)

    def opt_by_gradient(self, env: Environment):
        x = self.x
        loss, surrogate_loss = env.get_loss(x)
        grad = env.get_grad(x)
        step_size = self.get_step_size()
        self.x = x - step_size * grad
        self.x = self.domain.project(self.x)
        return x, loss, surrogate_loss


class BGDOnePoint(Base):
    """Implementation of Bandit Convex Optimization with one-point feedback.

    ``BGDOnePoint`` stands for Bandit Convex Optimization with one-point
    feedback, in which at each round the learner submits one decision and then
    can only observe the function value of this point. ``BGDOnePoint`` updates
    the decision :math:`x_{t+1}` by

    .. math::

        x_{t+1} = \Pi_{\mathcal{X}}[x_t - \eta_t \\tilde{g}_t] \mbox{ with }
        \\tilde{g}_t = \\frac{d}{\delta}f_t(x_t + \delta s_t) \cdot s_t,

    where :math:`\eta_t > 0` is the step size at round `t`, :math:`d` is the
    dimension, :math:`\delta` is the scale of the perturbation, :math:`s_t` is
    the unit vector selected uniformly at random and
    :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the projection onto the nearest
    point in :math:`\mathcal{X}`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round $t$.
        scale_perturb (float): Scale of perturbation :math:`\delta`.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.

    References:
        https://arxiv.org/abs/cs/0408007
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, np.ndarray],
                 scale_perturb: float,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, prior, seed)
        self.scale_perturb = scale_perturb
        self.shrink = self.scale_perturb / self.domain.r

    def opt_by_gradient(self, env: Environment):
        """Optimize by the estimated gradient.

        Args:
            env (Environment): Environment at the current round.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): Decision at the current round. \n
                loss (float): the origin loss at the current round. \n
                surrogate_loss (float): Surrogate loss at the current round.
        """
        unit_vec = self.domain.unit_vec()
        x1 = self.x + self.scale_perturb * unit_vec
        loss, surrogate_loss = env.get_loss(x1)
        grad = self.get_grad(self.domain.dimension, loss, unit_vec)
        step_size = self.get_step_size()
        self.x = self.x - step_size * grad
        self.x = ((1 - self.shrink) * self.domain).project(self.x)
        self.t += 1
        return x1, loss, surrogate_loss

    def get_grad(self, dimension, loss, unit_vec) -> np.ndarray:
        """Estimate the gradient by :math:`\\tilde{g}_t =
        \\frac{d}{\delta}f_t(x_t + \delta s_t) \cdot s_t`.

        Args:
            dimension (int): Dimension of the decision.
            loss (float): Loss of the submitted decision.
            unit_vec (numpy.ndarray): Unit vector used to perturb the decision.

        Returns:
            numpy.ndarray: the estimated gradient at the current round.
        """
        grad = dimension / self.scale_perturb * loss * unit_vec
        return grad

    def reinit(self):
        self.__init__(self.domain, self.step_size, self.scale_perturb,
                      self.prior, self.seed)


class BGDTwoPoint(Base):
    """Implementation of Bandit Convex Optimization with Two-point Feedback.

    ``BGDTwoPoint`` stands for Bandit Convex Optimization with two-point
    feedback, in which at each round the learner submits two decisions and then
    can only observe the function values of these two points. ``BGDTwoPoint``
    updates the decision :math:`x_{t+1}` by

    .. math::

        x_{t+1} = \Pi_{\mathcal{X}} [x_t - \eta_t \\tilde{g}_t] \mbox{ with }
        \\tilde{g}_t = \\frac{d}{2\delta}(f_t(x_t + \delta s_t) - f_t(x_t -
        \delta s_t))\cdot s_t,

    where :math:`\eta_t > 0` is the step size at round `t`, :math:`d` is the
    dimension, :math:`\delta` is the scale of the perturbation, :math:`s_t` is
    the unit vector selected uniformly at random, and
    :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the projection onto the nearest
    point in :math:`\mathcal{X}`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round $t$.
        scale_perturb (float): Scale of perturbation :math:`\delta`.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.


    References:
        https://alekhagarwal.net/bandits-colt.pdf
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, list],
                 scale_perturb: float,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, prior, seed)
        self.scale_perturb = scale_perturb
        self.shrink = self.scale_perturb / self.domain.r

    def opt_by_gradient(self, env: Environment):
        """Optimize by the estimated gradient.

        Args:
            env (Environment): Environment at the current round.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): Average decision at the current round. \n
                loss (float): Average origin loss at the current round. \n
                surrogate_loss (float): Average surrogate loss at the current round.
        """
        x = self.x
        unit_vec = self.domain.unit_vec()
        x1 = self.x + self.scale_perturb * unit_vec
        x2 = self.x - self.scale_perturb * unit_vec
        loss1, surrogate_loss1 = env.get_loss(x1)
        loss2, surrogate_loss2 = env.get_loss(x2)
        loss = (loss1 + loss2) / 2
        if surrogate_loss1 is not None and surrogate_loss2 is not None:
            surrogate_loss = (surrogate_loss1 + surrogate_loss2) / 2
        else:
            surrogate_loss = None
        grad = self.get_grad(self.domain.dimension, loss1, loss2, unit_vec)
        step_size = self.get_step_size()
        self.x = self.x - step_size * grad
        self.x = ((1 - self.shrink) * self.domain).project(self.x)
        self.t += 1
        return x, loss, surrogate_loss

    def get_grad(self, dimension, loss1, loss2, unit_vec):
        """Estimate the gradient by :math:`\\tilde{g}_t =
        \\frac{d}{2\delta}(f_t(x_t + \delta s_t) - f_t(x_t - \delta s_t))\cdot
        s_t`.

        Args:
            dimension (int): Dimension of the decision.
            loss1 (float): Loss of the first submitted decision.
            loss2 (float): Loss of the second submitted decision.
            unit_vec (numpy.ndarray): Unit vector used to perturb the decision.

        Returns:
            numpy.ndarray: Estimated gradient at the current round.
        """
        self.grad = dimension / (2 * self.scale_perturb) * (loss1 -
                                                            loss2) * unit_vec
        return self.grad

    def reinit(self):
        self.__init__(self.domain, self.step_size, self.scale_perturb,
                      self.prior, self.seed)


class SOGD(Base):
    """Implementation of scale-free online learning.

    ``SOGD`` stands for Scale-free Online Gradient Descent, an online convex
    optimization algorithm that achieves parameter-free and scale-free
    simultaneously, which is useful in deriving small-loss bound for smooth
    functions. ``SOGD`` updates the decision :math:`x_{t+1}` by

    .. math::

        x_{t+1} = \Pi_{\mathcal{X}}[x_t - \\frac{1}{\sqrt{\delta +
        \sum_{s=1}^{t-1} \\nabla f_s(x_s)}}  \\nabla f_t(x_t)]

    where :math:`\delta` is a small constant to avoid divide by :math:`0`, and
    :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the projection onto the nearest
    point in :math:`\mathcal{X}`. We set :math:`\delta=1` in our implementation.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.

    References:
        https://arxiv.org/abs/1601.01974
    """

    def __init__(self,
                 domain: Domain,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, None, prior, seed)
        self.grad_cum = 1

    def opt_by_gradient(self, env: Environment):
        x = self.x
        loss, surrogate_loss = env.get_loss(x)
        grad = env.get_grad(x)
        self.grad_cum += np.dot(grad, grad)
        step_size = 1 / np.sqrt(self.grad_cum)
        self.x = self.x - step_size * grad
        self.x = self.domain.project(self.x)
        self.t += 1
        return x, loss, surrogate_loss

    def reinit(self):
        self.__init__(self.domain, self.prior, self.seed)


class OEGD(Base):
    """Implementation of online extra-gradient descent.

    ``OEGD`` stands for Online Extra-Gradient Descent. It is shown to enjoy
    gradient-variation regret guarantee, which would be smaller when the
    environments evolve gradually. ``OEGD`` updates decision :math:`x_t` by

    .. math::

        x_t = \Pi_{\mathcal{X}}[\hat{x}_t - \eta_t \\nabla f_{t-1}(\hat{x}_t)], \n
        \hat{x}_{t+1} = \Pi_{\mathcal{X}}[\hat{x}_t - \eta_t \\nabla f_t(x_t)],

    where :math:`\eta_t > 0` is is the step size at round :math:`t`, and
    :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the projection onto the nearest
    point in :math:`\mathcal{X}`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round $t$.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.

    References:
        https://proceedings.mlr.press/v23/chiang12.html
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, np.ndarray],
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, prior, seed)
        self.middle_x = self.x
        self.optimism = np.zeros_like(self.x)

    def opt_by_gradient(self, env: Environment):
        loss, surrogate_loss = env.get_loss(self.x)
        grad = env.get_grad(self.x)
        step_size = self.get_step_size()
        self.middle_x = self.middle_x - step_size * grad
        self.middle_x = self.domain.project(self.middle_x)
        self.optimism = env.get_grad(self.middle_x)
        self.t += 1
        return self.x, loss, surrogate_loss

    def opt_by_optimism(self, optimism: Optional[np.ndarray] = None):
        step_size = self.get_step_size()
        self.x = self.middle_x - step_size * self.optimism
        self.x = self.domain.project(self.x)


class OptimisticBase(Base):
    """An abstract class for optimistic-type base algorithms.

    The optimistic-type algorithms are very general and useful, and they can
    achieve a tighter regret guarantee with benign “predictable sequences”.
    There are mainly two types of optimistic algorithms: Optimistic Online
    Mirror Descent (Optimistic OMD) and Optimistic Follow The Regularized Leader
    (Optimistic FTRL). The general update rule of ``Optimistic OMD`` is as
    follows,

    .. math::
        x_t = {\\arg\min}_{x \in \mathcal{X}} \ \eta_t \langle m_t, x\\rangle + D_R(x,
        \hat{x}_t) \n
        \hat{x}_{t+1} = {\\arg\min}_{x \in \mathcal{X}} \ \eta_t \langle \\nabla
        f_t(x_t), x\\rangle + D_R(x, \hat{x}_t),

    and the general update rule of ``Optimistic FTRL`` is as follows,

    .. math::
        x_t = {\\arg\min}_{x \in \mathcal{X}} \ \eta_t \langle \sum_{s=1}^{t-1}
        \\nabla f_s(x_s) + m_t, x\\rangle + R(x),

    where :math:`\eta_t` is the step size at round :math:`t`, :math:`m_t` is the
    optimism at the beginning of round :math:`t`, serving as a guess of the true
    gradient :math:`\\nabla f_t(x_t)` at round :math:`t`, :math:`R` is the
    regularizer and :math:`D_R(\cdot, \cdot)` is the Bregman divergence with
    respect to the regularizer :math:`R`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round :math:`t`.
        optimism_type (str, optional): Type of optimism used for algorithm.
            Valid actions include ``external``, ``last_grad``, ``middle_grad`` and
            ``None``. if ``optimism_type='external'``, the algorithm will accept the
            external specified :math:`m_t` by the environment at each round; if
            ``optimism_type='last_grad'``, the optimism is set as :math:`m_t =
            \\nabla f_{t-1}(x_{t-1})`; if ``optimism_type='middle_grad'``, the
            optimism is set as :math:`m_t = \\nabla f_{t-1}(\hat{x}_t)`, and if
            ``optimism_type=None``, the optimism is set as :math:`m_t = 0`.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.

    References:
        https://proceedings.mlr.press/v23/chiang12.html
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, list],
                 optimism_type: str = 'external',
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, prior, seed)
        self.optimism_type = optimism_type
        self.middle_x = self.x
        self.optimism = np.zeros_like(self.x)

    def compute_internal_optimism(self, env: Environment):
        """Compute the internal optimism.

        Args:
            env (Environment): Environment at the current round.
        """
        if self.optimism_type is None or self.optimism_type == 'external':
            pass
        elif self.optimism_type == 'last_grad':
            self.optimism = self.grad
        elif self.optimism_type == 'middle_grad':
            self.optimism = env.get_grad(self.middle_x)
        else:
            raise TypeError(f'{self.optimism_type} is not defined')

    def reinit(self):
        self.__init__(self.domain, self.step_size, self.optimism_type,
                      self.prior, self.seed)


class OptimisticOGD(OptimisticBase):
    """Implementation of Optimistic Online Gradient Descent.

    ``OptimisticOGD`` is an online convex optimization algorithm, which is a
    specialization of ``Optimistic OMD`` with the Euclidean distance as the
    regularizer. ``OptimisticOGD`` updates the decision :math:`x_{t}` by

    .. math::

        x_t = \Pi_{\mathcal{X}}[\hat{x}_t - \eta_t m_t] \n
        \hat{x}_{t+1} = \Pi_{\mathcal{X}}[\hat{x}_t - \eta_t \\nabla f_t(x_t)],

    where :math:`\eta_t > 0` is the step size at round :math:`t`, :math:`m_t` is
    the optimism at the beginning of round :math:`t`, and
    :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the projection onto the nearest
    point in :math:`\mathcal{X}`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round :math:`t`.
        optimism_type (str, optional): Type of optimism used for algorithm.
            Valid actions include ``external``, ``last_grad``, ``middle_grad`` and
            ``None``. if ``optimism_type='external'``, the algorithm will accept the
            external specified :math:`m_t` by the environment at each round; if
            ``optimism_type='last_grad'``, the optimism is set as :math:`m_t =
            \\nabla f_{t-1}(x_{t-1})`; if ``optimism_type='middle_grad'``, the
            optimism is set as :math:`m_t = \\nabla f_{t-1}(\hat{x}_t)`, and if
            ``optimism_type=None``, the optimism is set as :math:`m_t = 0`.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.

    .. note::
        1. ``OGD`` is a special case of ``OptimisticOGD`` with ``optimism_type=None``.
        2. ``OEGD`` is a special case of ``OptimisticOGD`` with ``optimism_type='middle_grad``.

    References:
        https://proceedings.mlr.press/v30/Rakhlin13.html
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, list],
                 optimism_type: str = 'external',
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, optimism_type, prior, seed)

    def opt_by_optimism(self, optimism: Optional[np.ndarray] = None):
        if self.optimism_type is None:
            self.optimism = np.zeros_like(self.middle_x)
        elif self.optimism_type == 'external':
            self.optimism = optimism if optimism is not None else np.zeros_like(
                self.middle_x)
        else:
            pass
        step_size = self.get_step_size()
        self.x = self.middle_x - step_size * self.optimism
        self.x = self.domain.project(self.x)

    def opt_by_gradient(self, env: Environment):
        loss, surrogate_loss = env.get_loss(self.x)
        self.grad = env.get_grad(self.x)
        step_size = self.get_step_size()
        self.middle_x = self.middle_x - step_size * self.grad
        self.middle_x = self.domain.project(self.middle_x)
        self.compute_internal_optimism(env)
        self.t += 1
        return self.x, loss, surrogate_loss


class OptimisticHedge(OptimisticBase):
    """Implementation of Optimistic Hedge.

    ``Hedge`` is the most popular algorithm for tracking the best expert
    problem. ``Optimistic Hedge`` is an enhanced version of ``Hedge`` which can
    further incorporate the knowledge of predictable sequences. There are two
    versions of ``Optimistic Hedge``: the greedy version and the lazy version.
    The greedy version is a special case of ``Optimistic OMD`` with the negative
    entropy as the regularizer and the lazy version is a special case of
    ``Optimistic FTRL``. The greedy version ``Optimistic Hedge`` updates the
    decision :math:`x_t` by

    .. math::

        {x}_t =  \Pi_{\mathcal{X}}[\hat{x}_{t} \exp(-\eta_t m_{t})] \n
        \hat{x}_{t+1} = \Pi_{\mathcal{X}}[\hat{x}_t\exp(-\eta_t (\\nabla
        f_t({x}_t) +a_t))],

    and the lazy version ``Optimistic Hedge`` updates the decision :math:`x_t` by

    .. math::

        x_t = \Pi_{\mathcal{X}}[x_1 \exp(-\eta_t (\sum_{s=1}^{t-1} (\\nabla
        f_s(x_s) +a_s) + m_t))],

    where :math:`\eta_t > 0` is the step size at round :math:`t`, :math:`m_t` is
    the optimism at the beginning of round :math:`t`, :math:`a_t` is the
    correction term and :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the projection
    onto the nearest point in :math:`\mathcal{X}`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): Step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round :math:`t`.
        optimism_type (str, optional): Type of optimism used for algorithm.
            Valid actions include ``external``, ``last_grad``, ``middle_grad`` and
            ``None``. if ``optimism_type='external'``, the algorithm will accept the
            external specified :math:`m_t` by the environment at each round; if
            ``optimism_type='last_grad'``, the optimism is set as :math:`m_t =
            \\nabla f_{t-1}(x_{t-1})`; if ``optimism_type='middle_grad'``, the
            optimism is set as :math:`m_t = \\nabla f_{t-1}(\hat{x}_t)`, and if
            ``optimism_type=None``, the optimism is set as :math:`m_t = 0`.
        is_lazy (bool): Type of the update version: lazy or greedy. The
            default is False.
        correct (bool): Whether to use correction term. :math:`a_t` is
            set as :math:`a_t = \eta_t (\\nabla f_t(x_t) - m_t)^2` if ``correct=True`` and
            :math:`a_t=0` otherwise. The default is False.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.

    References:
        http://proceedings.mlr.press/v30/Rakhlin13.pdf

    .. note::
        1. Greedy version ``Optimistic Hedge`` is a special case of ``OptimisticOMD``
        with the negative entropy as the regularizer.

        2. Lazy version ``Optimistic Hedge`` is a special case of ``OptimisticFTRL``
        with the negative entropy as the regularizer.
    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, list],
                 optimism_type: str = 'external',
                 is_lazy: bool = False,
                 correct: bool = False,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, optimism_type, prior, seed)
        self.is_lazy = is_lazy
        if self.is_lazy:
            self.cum_loss = np.zeros_like(self.x)
        self.correct = correct

    def opt_by_optimism(self, optimism: Optional[np.ndarray] = None):
        if self.optimism_type == 'external' and optimism is not None:
            self.optimism = optimism
        else:
            pass
        step_size = self.get_step_size()
        self.x = self.middle_x * np.exp(-step_size * self.optimism)
        self.x = self.domain.project(self.x)

    def opt_by_gradient(self, env: Environment):
        loss, surrogate_loss = env.get_loss(self.x)
        self.grad = env.get_grad(self.x)
        step_size = self.get_step_size()
        if self.correct and self.optimism is not None:
            correction = step_size * (self.grad - self.optimism)**2
        elif self.correct:
            correction = step_size * (self.grad)**2
        else:
            correction = np.zeros_like(self.grad)
        if self.is_lazy:
            self.cum_loss += self.grad
            self.middle_x = self.init_x * np.exp(-step_size *
                                                 (self.cum_loss + correction))
        else:
            self.middle_x = self.middle_x * np.exp(-step_size *
                                                   (self.grad + correction))
        self.middle_x = self.domain.project(self.middle_x)
        self.compute_internal_optimism(env)
        self.t += 1
        return self.x, loss, surrogate_loss

    def reinit(self):
        self.__init__(self.domain, self.step_size, self.optimism_type,
                      self.is_lazy, self.correct, self.prior, self.seed)


class Hedge(OptimisticHedge):
    """Implementation of Hedge.

    ``Hedge`` is the most popular algorithm for tracking the best expert
    problem. There are two versions of ``Hedge``: the greedy version and the
    lazy version. The greedy version is a special case of ``Optimistic OMD``
    with the negative entropy as the regularizer and ``optimism_type=None`` and
    the lazy version is a special case of ``Optimistic FTRL`` with the negative
    entropy as the regularizer and ``optimism_type=None``. The greedy version
    ``Hedge`` updates the decision :math:`x_{t+1}` by

    .. math::

        x_{t+1} = \Pi_{\mathcal{X}}[x_t\exp(-\eta_t (\\nabla f_t({x}_t) +a_t))],

    and the lazy version ``Hedge`` updates the decision :math:`x_{t+1}` by

    .. math::

        x_{t+1} = \Pi_{\mathcal{X}}[x_1 \exp(-\eta_t (\sum_{s=1}^{t} (\\nabla
        f_s(x_s) +a_s)))],

    where :math:`\eta_t > 0` is the step size, :math:`a_t` is the correction
    term at round :math:`t`, and :math:`\Pi_{\mathcal{X}}[\cdot]` denotes the
    projection onto the nearest point in :math:`\mathcal{X}`.

    Args:
        domain (Domain): Feasible set for the base algorithm.
        step_size (float, numpy.ndarray): the step size :math:`\eta` for the
            base algorithm. Valid types include ``float`` and ``numpy.ndarray``. If
            the type of the step size :math:`\eta` is `float`, the algorithm will
            use the fixed step size all the time, otherwise, the algorithm will use
            the step size :math:`\eta_t` at round :math:`t`.
        is_lazy (bool): Type of the update version: lazy or greedy. The
            default is False.
        correct (bool): Whether to use correction term. :math:`a_t` is
            set as :math:`a_t = \\nabla f_t(x_t)^2` if ``correct=True`` and
            :math:`a_t=0` otherwise. The default is False.
        prior (str, numpy.ndarray, optional): The initial decision of the
            algorithm is set as ``domain.init_x(prior, seed)``.
        seed (int, optional): The initial decision of the algorithm is set as
            ``domain.init_x(prior, seed)``.

    """

    def __init__(self,
                 domain: Domain,
                 step_size: Union[float, list],
                 is_lazy: bool = False,
                 correct: bool = False,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        super().__init__(domain, step_size, None, is_lazy, correct, prior,
                         seed)
