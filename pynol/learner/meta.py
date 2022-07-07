from abc import ABC, abstractmethod
from typing import Optional, Union

import cvxpy as cp
import numpy as np


class OptimisticLR:
    """An self-confident tuning method for optimistic algorithms.

    The update rule of the optimistic learning rate is

    .. math::

        \\varepsilon_t = \\frac{\\alpha}{\sqrt{\sum_{s=1}^{t-1}\lVert \ell_t - M_t \\rVert_p^q}},

    where :math:`\\alpha` is the scale parameter, :math:`p` is the norm order
    and :math`q` is the order.

    Args:
        scale (float): Scale parameter.
        norm (int, inf): Order of norm :math:`p`.
        order (int): Order of the norm value :math:`q`.
        upper_bound (float): Upper bound of learning rate.
    """

    def __init__(self,
                 scale: float = 1.,
                 norm: int = np.inf,
                 order: int = 2,
                 upper_bound: float = 1.):
        self.scale = scale
        self.norm = norm
        self.order = order
        self.upper_bound = upper_bound
        self.cum_var = 1
        self.lr = upper_bound

    def update_lr(self, optimism, loss):
        """Update learning rate by ``optimism`` and ``loss`` of the current round."""
        self.cum_var += np.linalg.norm(
            loss - optimism, ord=self.norm)**self.order
        self.lr = min(self.upper_bound, self.scale * self.cum_var**(-0.5))
        return self.lr


class Meta(ABC):
    """An abstract class for meta-algorithms.

    Args:
        prob (numpy.ndarray): Initial probability over the base-learners.
        lr (float, numpy.ndarray, OptimisticLR): Learning rate for
            meta-algorithm.
    """

    def __init__(self, prob: np.ndarray, lr: Union[float, np.ndarray,
                                                   OptimisticLR]):
        self._prob = prob
        self._init_prob = self._prob.copy()
        self.lr = lr
        self._active_state = np.ones(len(prob))
        self._active_index = np.where(self._active_state > 0)[0]
        self.t = 0

    def opt(self,
            loss_bases: np.ndarray,
            loss_meta: Optional[float] = None,
            optimism: Optional[np.ndarray] = None):
        """The optimization process of the meta-algorithm.

        All base algorithms are divided into two parts:
        :meth:`~pynol.learner.meta.Meta.opt_by_optimism` at the beginning of
        current round and :meth:`~pynol.learner.meta.Meta.opt_by_gradient` at the
        end of current round.

        Args:
            loss_bases (numpy.ndarray): Losses of all alive base-learners.
            loss_meta (float, optional): Loss of the combined decision.
            optimism (numpy.ndarray): Optimism at the beginning of the
                current round, serving as a guess of ``loss_bases``.

        Returns:
            numpy.ndarray: Probability over the alive base-learners.
        """
        self.opt_by_optimism(optimism)
        return self.opt_by_gradient(loss_bases, loss_meta)

    def opt_by_optimism(self, optimism: Optional[np.ndarray]):
        """Optimize by the optimism.

        Args:
            optimism (numpy.ndarray, optional): the optimism at the beginning of the
                current round.

        Returns:
            None
        """
        pass

    @abstractmethod
    def opt_by_gradient(
        self,
        loss_bases: np.ndarray,
        loss_meta: Optional[float] = None,
    ):
        """Optimize by the true gradient (loss).

        All base-algorithms are required to override this method to implement
        their own optimization process.

        Args:
            loss_bases (numpy.ndarray): Losses of all alive base-learners.
            loss_meta (float): Loss of the combined decision.

        Returns:
            numpy.ndarray: Probability over the alive base-learners.
        """
        pass

    def get_lr(self):
        """Compute the learning rate for meta-algorithms.

        If the type of the learning rate :math:`\\varepsilon` is ``float``, this
        method will return the constant learning rate :math:`\\varepsilon` all the
        time; if the type of :math:`\\varepsilon` is ``numpy.ndarray`` with
        shape ``[T, ]``, this method will return a scalar
        :math:`\\varepsilon[t]` at round :math:`t`; if the type of
        :math:`\\varepsilon` is ``numpy.ndarray`` with shape ``[T, N]`` where
        :math:`N` is the dimension, this method will return a vector
        :math:`\\varepsilon[t]` at time :math:`t`; if the type of
        :math:`\\varepsilon` is ``numpy.ndarray`` with shape ``[1, N]``, this
        method will return a vector :math:`\\varepsilon` all the time; if the
        type of :math:`\\varepsilon` is
        :meth:`~pynol.learner.meta.Meta.OptimisticLR`, the learning rate
        :math:`\\varepsilon_t` is computed by
        :meth:`~pynol.learner.meta.Meta.OptimisticLR.compute_lr()` at round
        :math:`t`.
        """
        if isinstance(self.lr, OptimisticLR):
            return self.lr.lr
        elif isinstance(self.lr, np.ndarray):
            if self.lr.ndim == 1:
                return self.lr[self.t]
            elif self.lr.ndim == 2:
                if self.lr.shape[0] == 1:
                    return self.lr[0, self._active_index]
                elif self.lr.shape[1] == 1:
                    return self.lr[self.t, 0]
                else:
                    assert self.lr.shape[1] == len(self.active_state)
                    return self.lr[self.t, self._active_index]
        else:
            return self.lr

    @property
    def active_state(self):
        """Get the active state of base-learners:

        - 0: sleep
        - 1: active at the current round and the previous round.
        - 2: active at the current round and sleep at the previous round.
        """
        return self._active_state

    @active_state.setter
    def active_state(self, active_state):
        self._active_state = active_state
        self._active_index = np.where(active_state > 0)[0]

    @property
    def prob(self):
        """Get the current probability over the alive base-learners."""
        return self._prob[self._active_index]

    @prob.setter
    def prob(self, prob):
        self._prob[self._active_index] = prob

    @property
    def init_prob(self):
        """Get the initial probability over the current alive base-learners."""
        return self._init_prob[self._active_index]


class OptimisticMeta(Meta):
    """An abstract class for optimistic-type base algorithms.

    The optimistic-type algorithms are very general and useful, and they can
    achieve a tighter regret guarantee with benign “predictable sequences”.
    There are mainly two types of optimistic algorithms: Optimistic Online
    Mirror Descent (Optimistic OMD) and Optimistic Follow The Regularized Leader
    (Optimistic FTRL). The general update rule of ``Optimistic OMD`` is as
    follows,

    .. math::
        p_t = {\\arg\min}_{p \in \Delta_N} \ \\varepsilon_t \langle M_t, p\\rangle + D_R(p,
        \hat{p}_t) \n
        \hat{p}_{t+1} = {\\arg\min}_{x \in \Delta_N} \ \\varepsilon_t \langle
        \ell_t, p\\rangle + D_R(p, \hat{p}_t),

    and the general update rule of ``Optimistic FTRL`` is as follows,

    .. math::
        p_t = {\\arg\min}_{x \in \Delta_N} \ \\varepsilon_t \langle \sum_{s=1}^{t-1}
        \ell_s + M_t, x\\rangle + R(x),

    where :math:`\\varepsilon_t` is the step size at round :math:`t`, :math:`M_t` is the
    optimism at the beginning of round :math:`t`, serving as a guess of the true
    gradient :math:`\ell_t` at round :math:`t`, :math:`R` is the
    regularizer and :math:`D_R(\cdot, \cdot)` is the Bregman divergence with
    respect to the regularizer :math:`R`.

    Args:
        prob (numpy.ndarray): The initial probability over the base-learners.
        lr (float, numpy.ndarray, OptimisticLR): The learning rate for
            meta-algorithm.
        optimism_type (str, optional): the type of optimism used for algorithm.
            Valid actions include ``external``, ``last_loss`` and
            ``None``. if ``optimism_type='external'``, the algorithm will accept the
            external optimism :math:`M_t` at each round; if
            ``optimism_type='last_loss'``, the optimism is set as :math:`M_t =
            \ell_{t-1}`; and if ``optimism_type=None``, the optimism is set as :math:`M_t = 0`.

    References:
        https://proceedings.mlr.press/v23/chiang12.html
    """

    def __init__(self,
                 prob: np.ndarray,
                 lr: Union[float, np.ndarray, OptimisticLR],
                 optimism_type: Optional[str] = 'external'):
        super().__init__(prob, lr)
        self.optimism_type = optimism_type
        self._middle_prob = self._prob.copy()
        self._optimism = np.zeros_like(self._prob)

    def compute_internal_optimism(self, loss_bases: np.ndarray):
        """Compute the internal optimism for the next round.

        If ``optimism_type='last_loss'``, the interval optimism is set as
        :math:`M_t = \ell_{t-1}`.

        Args:
            loss_bases (numpy.ndarray): Losses of base-learners at the current round.
        """
        if self.optimism_type is None or self.optimism_type == 'external':
            pass
        elif self.optimism_type == 'last_loss':
            self.optimism = loss_bases
        else:
            raise TypeError(f'{self.optimism_type} is not defined')

    @property
    def optimism(self):
        """Get the current optimism of the alive base-learners."""
        return self._optimism[self._active_index]

    @optimism.setter
    def optimism(self, optimism: Optional[np.ndarray]):
        self._optimism[self._active_index] = optimism

    @property
    def middle_prob(self):
        """Get the current intermediate probability over the alive base-learners."""
        return self._middle_prob[self._active_index]

    @middle_prob.setter
    def middle_prob(self, middle_prob):
        self._middle_prob[self._active_index] = middle_prob


class OptimisticHedge(OptimisticMeta):
    """Implementation of Optimistic Hedge.

    ``Hedge`` is the most popular algorithm for tracking the best expert
    problem. ``Optimistic Hedge`` is an enhanced version of ``Hedge`` which can
    further incorporate the knowledge of predictable sequences. There are two
    versions of ``Optimistic Hedge``: the greedy version and the lazy version.
    The greedy version is a special case of ``Optimistic OMD`` with the negative
    entropy as the regularizer and the lazy version is a special case of
    ``Optimistic FTRL``. The greedy version ``Optimistic Hedge`` updates the
    decision :math:`p_t` by

    .. math::

        {p}_{t, i} = \\frac{\hat{p}_{t,i}\exp(-\\varepsilon_t
        M_{t,i})}{\sum_{j=1}^N \hat{p}_{t,j} \exp(-\\varepsilon_t M_{t,j})},
        \\forall i \in [N] \n
        \hat{p}_{t+1, i} = \\frac{\hat{p}_{t,i}\exp(-\\varepsilon_t (\ell_{t,i} +
        b_{t,i}))}{\sum_{j=1}^N \hat{p}_{t,j} \exp(-\\varepsilon_t (\ell_{t,j}+
        b_{t,j}))}, \\forall i \in [N],

    and the lazy version ``Optimistic Hedge`` updates the decision :math:`x_t` by

    .. math::

        p_{t, i} \propto p_{1,i} \exp \Big(-\\varepsilon_t (\sum_{s=1}^{t-1} (\ell_{s,i}
        + b_{s,i})+ m_{t,i})\Big), \\forall i \in [N],

    where :math:`\\varepsilon_t > 0` is the learning rate at round :math:`t`,
    :math:`M_t` is the optimism at the beginning of round :math:`t`, :math:`b_t`
    is the correction term.

    Args:
        prob (numpy.ndarray): Initial probability over the base-learners.
        lr (float, numpy.ndarray, OptimisticLR): The learning rate for
            meta-algorithm.
        optimism_type (str, optional): Type of optimism used for algorithm.
            Valid actions include ``external``, ``last_loss`` and
            ``None``. if ``optimism_type='external'``, the algorithm will accept the
            external optimism :math:`M_t` at each round; if
            ``optimism_type='last_loss'``, the optimism is set as :math:`M_t =
            \ell_{t-1}`; and if ``optimism_type=None``, the optimism is set as
            :math:`M_t = 0`.
        is_lazy (bool): Type of the update version: lazy or greedy. The
            default is False.
        correct (bool): Whether to use correction term. :math:`b_t` is
            set as :math:`b_t = \\varepsilon_t (\ell_t-M_t)^2` if
            ``correct=True`` and :math:`b_t=0` otherwise. The default is False.

    .. note::
        1. Greedy version ``Optimistic Hedge`` is a special case of ``OptimisticOMD``
        with the negative entropy as the regularizer.

        2. Lazy version ``Optimistic Hedge`` is a special case of ``OptimisticFTRL``
        with the negative entropy as the regularizer.
    """

    def __init__(self,
                 prob: np.ndarray,
                 lr: Union[float, np.ndarray, OptimisticLR],
                 optimism_type: Optional[str] = 'external',
                 is_lazy: bool = False,
                 correct: bool = False):
        super().__init__(prob, lr, optimism_type)
        self.is_lazy = is_lazy
        if self.is_lazy:
            self._cum_loss = np.zeros_like(self._prob)
        self.correct = correct

    def opt_by_optimism(self, optimism: Optional[np.ndarray]):
        if self.optimism_type == 'external' and optimism is not None:
            self.optimism = optimism
        else:
            pass
        lr = self.get_lr()
        exp_optimism = np.exp(-lr * self.optimism)
        self.prob = self.middle_prob * exp_optimism / np.dot(
            self.middle_prob, exp_optimism)

    def opt_by_gradient(self,
                        loss_bases: np.ndarray,
                        loss_meta: Optional[float] = None):
        lr = self.get_lr()
        if self.correct:
            correction = lr * (loss_bases - self.optimism)**2
        else:
            correction = np.zeros_like(loss_bases)
        if self.is_lazy:
            self.cum_loss += loss_bases
            exp_loss = np.exp(-lr * (self.cum_loss + correction))
            self.middle_prob = self.init_prob * exp_loss / np.dot(
                self.init_prob, exp_loss)
        else:
            exp_loss = np.exp(-lr * (loss_bases + correction))
            self.middle_prob = self.middle_prob * exp_loss / np.dot(
                self.middle_prob, exp_loss)
        # update learning rate
        if isinstance(self.lr, OptimisticLR):
            self.lr.update_lr(self.optimism, loss_bases)
        # update by internal optimism
        self.compute_internal_optimism(loss_bases)
        self.t += 1
        return self.prob

    @property
    def cum_loss(self):
        """Get the cumulative loss of the alive base-learners."""
        return self._cum_loss[self._active_index]

    @cum_loss.setter
    def cum_loss(self, cum_loss: np.ndarray):
        self._cum_loss[self._active_index] = cum_loss


class Hedge(OptimisticHedge):
    """Implementation of Hedge.

    ``Hedge`` is the most popular algorithm for tracking the best expert
    problem. There are two versions of ``Hedge``: the greedy version and the
    lazy version. The greedy version is a special case of ``Optimistic OMD``
    with the negative entropy as the regularizer and ``optimism_type=None`` and
    the lazy version is a special case of ``Optimistic FTRL`` with the negative
    entropy as the regularizer and ``optimism_type=None``. The greedy version
    ``Hedge`` updates the decision :math:`p_{t+1}` by

    .. math::

        p_{t+1, i} \propto p_{t,i} \exp \Big(-\\varepsilon_t (\ell_{t,i}
        + b_{t,i}) \Big), \\forall i \in [N],

    and the lazy version ``Hedge`` updates the decision :math:`p_{t+1}` by

    .. math::

        p_{t+1, i} \propto p_{1,i} \exp \Big(-\\varepsilon_t \sum_{s=1}^{t} (\ell_{s,i}
        + b_{s,i})\Big), \\forall i \in [N],

    where :math:`\\varepsilon_t > 0` is the step size, :math:`b_t` is the correction
    term at round :math:`t`.

    Args:
        prob (numpy.ndarray): Initial probability over the base-learners.
        lr (float, numpy.ndarray, OptimisticLR): The learning rate for
            meta-algorithm.
        is_lazy (bool): Type of the update version: lazy or greedy. The
            default is False.
        correct (bool): Whether to use correction term. :math:`b_t` is
            set as :math:`b_t = \\varepsilon_t \ell_t^2` if ``correct=True`` and
            :math:`b_t=0` otherwise. The default is False.

    References:
        http://proceedings.mlr.press/v30/Rakhlin13.pdf

    .. note::
        1. ``Hedge`` is a special case of ``OptimisticHedge`` with ``optimism_type=None``.
    """

    def __init__(self,
                 prob: np.ndarray,
                 lr: Union[float, np.ndarray, OptimisticLR],
                 is_lazy: bool = False,
                 correct: bool = False):
        super().__init__(prob, lr, None, is_lazy, correct)


class MSMWC(OptimisticMeta):
    """Implementation of Impossible Tuning Made Possible: A New Expert Algorithm and
    Its Applications.

    ``MSMWC`` stands for Multi-scale Multiplicative-weight with Correction, a
    multi-scale expert-tracking algorithms that achieves second-order small-loss
    regret simultaneously for all expert. It can further exploit the knowledge
    of optimism to achieve tighter bound. The update rule of ``MSMWC`` is as
    follows,

    .. math::
        p_t = {\\arg\min}_{p \in \Delta_N} \ \\varepsilon_t \langle p,  M_t\\rangle
        + D_R(p, \hat{p}_t) \n
        \hat{p}_{t+1} = {\\arg\min}_{p \in \Delta_N} \ \\varepsilon_t \langle p,
        \ell_t + b_t\\rangle + D_R(p, \hat{p}_t),

    where :math:`\\varepsilon_t > 0` is the step size, :math:`b_t =
    \\varepsilon_t (\ell_t - M_t)^2` is the correction term at round :math:`t`,
    :math:`R` is the weighted regularizer :math:`R(p) = \sum_{i=1}^N
    \\frac{1}{\\varepsilon_{t,i}}p_i \ln p_i` and :math:`D_R(\cdot, \cdot)` is
    the Bregman divergence with respect to the regularizer :math:`R`.

    Args:
        prob (numpy.ndarray): Initial probability over the base-learners.
        lr (float, numpy.ndarray, OptimisticLR): The learning rate for
            meta-algorithm.
        optimism_type (str, optional): Type of optimism used for algorithm.
            Valid actions include ``external``, ``last_loss`` and
            ``None``. if ``optimism_type='external'``, the algorithm will accept the
            external optimism :math:`M_t` at each round; if
            ``optimism_type='last_loss'``, the optimism is set as :math:`M_t =
            \ell_{t-1}`; and if ``optimism_type=None``, the optimism is set as
            :math:`M_t = 0`.

    References:
        https://arxiv.org/abs/2102.01046
    """

    def __init__(self,
                 prob: np.ndarray,
                 lr: Union[float, np.ndarray, OptimisticLR],
                 optimism_type: Optional[str] = 'external'):
        super().__init__(prob, lr, optimism_type)
        self.epsilon = 1e-3

    def opt_by_optimism(self, optimism: Optional[np.ndarray]):
        if self.optimism_type == 'external' and optimism is not None:
            self.optimism = optimism
        else:
            pass
        lr = self.get_lr()
        self.prob = self.middle_prob * np.exp(-lr * self.optimism)
        self.prob = self.project(self.prob, lr)

    def opt_by_gradient(self,
                        loss_bases: np.ndarray,
                        loss_meta: Optional[float] = None):
        lr = self.get_lr()
        correction = lr * (loss_bases - self.optimism)**2
        self.middle_prob = self.middle_prob * np.exp(-lr *
                                                     (loss_bases + correction))
        self.middle_prob = self.project(self.middle_prob, lr)
        # update by internal optimism
        self.compute_internal_optimism(loss_bases)
        self.t += 1
        return self.prob

    def project(self, prob: np.ndarray, lr: np.ndarray):
        """Project :math:`p'` back to the simplex by

        .. math::
            p = {\\arg\min}_{\Delta_N} \ D_R(p, p')

        where :math:`R` is the weighted regularizer :math:`R(p) = \sum_{i=1}^N
        \\frac{1}{\\varepsilon_{i}}p_i \ln p_i` and :math:`D_R(\cdot, \cdot)` is
        the Bregman divergence with respect to the regularizer :math:`R`.

        Args:
            prob (numpy.ndarray): Probability to be projected.
            lr (numpy.ndarray): Learning rate of the weighted regularizer :math:`R`.
        """
        x = cp.Variable(len(lr))
        prob = np.maximum(self.epsilon, prob)
        obj = cp.Minimize(cp.sum(cp.multiply(1 / lr, cp.kl_div(x, prob))))
        constr = [x >= self.epsilon, cp.sum(x) == 1]
        problem = cp.Problem(obj, constr)
        try:
            problem.solve()
        except Exception:
            problem.solve(solver='SCS', max_iters=200)
        if not problem.status.startswith('optimal'):
            raise RuntimeError('Optimal solution is not found.')
        return x.value


class Prod(Meta):
    """Implementation of Improved Second-Order Bounds for Prediction with Expert
    Advice and Strongly Adaptive Online Learning.

    ``Prod`` can be view as a first-order approximation to weighted majority
    majority, which is proved to enjoy second-order bounds. The update rule is
    as follows:

    .. math::

        p_{t+1,i} = (1+\\varepsilon_{t,i}) p_{t,i} / W_{t+1},

    where :math:`\\varepsilon_t > 0` is the learning rate and :math:`W_{t+1}` is
    the normalization constant.

    Args:
        N (int): Number of the total base-learners.
        lr (float, numpy.ndarray): Learning rate of meta-algorithm.

    References:
        https://link.springer.com/content/pdf/10.1007/s10994-006-5001-7.pdf \n
        http://proceedings.mlr.press/v37/daniely15.pdf
    """

    def __init__(self, N: int, lr: Union[float, np.ndarray]):
        super().__init__(np.ones(N), lr)
        self._w = np.zeros(N)

    def opt_by_gradient(self, loss_bases, loss_meta):
        self.w = self.w * (1 + self.get_lr() * (loss_meta - loss_bases))
        self.t += 1

    @Meta.active_state.setter
    def active_state(self, active_state):
        super(Prod, Prod).active_state.__set__(self, active_state)
        re_init_idx = np.where(self._active_state == 2)[0]
        self._w[re_init_idx] = self.lr[0, re_init_idx]
        self.prob = self.w / np.sum(self.w)

    @property
    def w(self):
        return self._w[self._active_index]

    @w.setter
    def w(self, w):
        self._w[self._active_index] = w


class AdaNormalHedge(Meta):
    """Implementation of Achieving All with No Parameters: AdaNormalHedge.

    ``AdaNormalHedge`` is a powerful parameter-free PEA algorithm that attains
    many interesting data-dependent bounds without any prior information. Notice
    that when using it as the meta-algorithm to combine base-learners, the input
    loss should be non-negative.

    Args:
        N (int): Number of the total base-learners.

    References:
        http://proceedings.mlr.press/v40/Luo15.pdf
    """

    def __init__(self, N: int):
        super().__init__(np.ones(N), None)
        self._R = np.zeros(N)
        self._C = np.zeros(N)
        self._w = np.zeros(N)

    def _Phi(self, R, C):
        R_plus = np.maximum(0, R)
        return np.exp(np.square(R_plus) / (3 * C))

    def _w_func(self, R, C):
        return 0.5 * (self._Phi(R + 1, C + 1) - self._Phi(R - 1, C - 1))

    def opt_by_gradient(self, loss_bases, loss_meta):
        self.R += loss_meta - loss_bases
        self.C += np.abs(loss_meta - loss_bases)
        self.w = self._w_func(self.R, self.C)
        self.prob = self.init_prob * self.w
        self.t += 1

    @Meta.active_state.setter
    def active_state(self, active_state):
        super(AdaNormalHedge,
              AdaNormalHedge).active_state.__set__(self, active_state)
        re_init_idx = np.where(self._active_state == 2)[0]
        self._R[re_init_idx], self._C[re_init_idx] = 0, 0
        self._w[re_init_idx] = self._w_func(0, 0)
        self._prob[
            re_init_idx] = self._init_prob[re_init_idx] * self._w[re_init_idx]
        self.prob /= np.sum(self.prob)

    @property
    def w(self):
        return self._w[self._active_index]

    @w.setter
    def w(self, w):
        self._w[self._active_index] = w

    @property
    def R(self):
        return self._R[self._active_index]

    @R.setter
    def R(self, R):
        self._R[self._active_index] = R

    @property
    def C(self):
        return self._C[self._active_index]

    @C.setter
    def C(self, C):
        self._C[self._active_index] = C


class AFLHMeta(Meta):
    """Implementation of the meta-algorithm of Adaptive algorithms for online
    decision problems.

    ``AdaNormalHedge`` is a powerful parameter-free PEA algorithm that attains
    many interesting data-dependent bounds without any prior information. Notice
    that when using it as the meta-algorithm to combine base-learners, the input
    loss should be non-negative.

    Args:
        N (int): Number of the total base-learners.
        lr (float, numpy.ndarray): Learning rate for meta-algorithm.

    References:
        https://dominoweb.draco.res.ibm.com/reports/rj10418.pdf
    """

    def __init__(self, N: int, lr: Union[float, np.ndarray]):
        super().__init__(np.ones(N), lr)
        self._prob = np.zeros(N)

    def opt_by_gradient(self, loss_bases, loss_meta):
        lr = self.get_lr()
        exp_loss = np.exp(-lr * loss_bases)
        self.prob = self.prob * exp_loss / np.dot(self.prob, exp_loss)
        self.t += 1

    @Meta.active_state.setter
    def active_state(self, active_state):
        super(AFLHMeta, AFLHMeta).active_state.__set__(self, active_state)
        re_init_idx = np.where(self._active_state == 2)[0]
        self._prob[re_init_idx] = 1 / (self.t + 1)
        self.prob /= np.sum(self.prob)
