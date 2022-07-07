from abc import ABC, abstractmethod

import numpy as np
from pynol.learner.meta import Hedge


class OptimismMeta(ABC):
    """The abstract class defines the optimism for meta-algorithm.

    Attributes:
        is_external (bool): Indicates the optimism of meta-algorithm depends on
            the optimism given by the environment or computed by the algorithm
            itself. The default is True.
    """

    def __init__(self, is_external: bool = True):
        self.is_external = is_external

    @abstractmethod
    def compute_optimism_meta(self):
        """Compute the optimism for meta-algorithm."""
        raise NotImplementedError()


class InnerSwitchingOptimismMeta(OptimismMeta):
    """The abstract class defines the inner function with switching cost to compute the
    optimism for meta-algorithm.

    Args:
        penalty (float): Penalty coefficient of switching cost term.
        norm (int): Order of norm :math:`p`.
        order (int): Order of switching cost :math:`q`.
    """

    def __init__(self, penalty: float, norm: int = 2, order: int = 2):
        super().__init__(is_external=True)
        self.penalty = penalty
        self.norm = norm
        self.order = order

    def compute_optimism_meta(self, variables):
        """Set the surrogate loss of meta-algorithm as

        .. math::

            M_t(x)=\langle m_t, x_{t,i} \\rangle + \lVert
            x_{t,i} - x_{t-1, i} \\rVert_p^q,

        where :math:`x_t` is the submitted decision and :math:`x_{t, i}` is the
        decision of base-learner i at round :math:`t`.
        """
        new_x_bases = variables['schedule'].x_active_bases
        x_bases = variables['x_bases'] if 'x_bases' in variables else None
        optimism = variables['schedule'].optimism
        optimism_meta = self.inner_switching(new_x_bases, optimism,
                                             self.penalty, x_bases, self.norm,
                                             self.order)
        return optimism_meta

    @staticmethod
    def inner_switching(x, gradient, penalty, x_last, norm=2, order=2):
        if x_last is None:
            x_last = x
        return (x * gradient).sum(axis=1) + penalty * np.linalg.norm(
                x - x_last, ord=norm, axis=1)**order


class InnerOptimismMeta(InnerSwitchingOptimismMeta):
    """The abstract class defines the inner function to compute the optimism for meta-algorithm.

    .. Note::

        ``InnerOptimismMeta`` is a special case of ``InnerSwitchingOptimism``
        with ``penalty = 0``.
    """

    def __init__(self):
        return super().__init__(penalty=0.)


class SwordVariationOptimismMeta(OptimismMeta):
    """The optimism of meta-algorithm used in SwordVariation."""

    def __init__(self):
        super().__init__(is_external=False)

    def compute_optimism_meta(self, variables):
        """set the optimism for meta-algorithm as :math:`M_{t,i} = \langle
        \\nabla f_{t-1}(\\bar{x}_t), x_{t,i} \\rangle` with :math:`\\bar{x}_t =
        \sum_{i=1}^N p_{t-1,i}x_{t,i}`, where :math:`x_{t,i}` is the decision of
        the base-learner :math:`i` at round :math:`t`, and :math:`p_{t-1}` is
        the decision of meta-algorithm at round :math:`t-1`.
        """
        x_bases = variables['schedule'].x_active_bases
        x_combined = np.dot(variables['meta'].prob, x_bases)
        optimism = variables['env'].get_grad(x_combined)
        optimism_meta = np.dot(x_bases, optimism)
        return optimism_meta


class SwordBestOptimismMeta(OptimismMeta):
    """The optimism of meta-algorithm used in SwordBest, who learn two optimism
    sequence via another expert-tracking algorithm."""

    def __init__(self) -> None:
        super().__init__(is_external=False)
        self._meta = Hedge(prob=np.ones(2)/2, lr=1)
        self._variation_optimism_last = None

    def compute_optimism_meta(self, variables):
        """To achieve the best-of-both-worlds results, this method will learn the
        best optimism from two optimism sequence :math:`M_t^v= \\nabla
        f_{t-1}(\\bar{x}_t), \\bar{x}_t = \sum_{i=1}^{N_1}
        p_{t-1,i}x_{t,i}^v + \sum_{i=N_1+1}^{N_2} p_{t-1,i}x_{t,i}^s` and
        :math:`M_t^s = 0`. The final optimism is set as :math:`\langle M_t^b,
        x_{t,i} \\rangle` with :math:`M_t^b = \\beta_t M_t^v +
        (1-\\beta_t)M_t^s`, where :math:`\\beta_t` is updated by

        .. math::

            \\beta_t = \\frac{\exp(-2\sum_{s=1}^{t-1}\lVert\\nabla
            f_s(x_s)-M_t^v\\rVert_2^2)}{\exp(-2\sum_{s=1}^{t-1}\lVert\\nabla
            f_s(x_s)-M_t^v\\rVert_2^2) + \exp(-2\sum_{s=1}^{t-1}\lVert\\nabla
            f_s(x_s)\\rVert_2^2)}.

        """
        loss = np.zeros(2)
        if 'grad' not in variables:
            return None
        if self._variation_optimism_last is None:
            self._variation_optimism_last = np.zeros_like(variables['grad'])
        loss[0] = np.linalg.norm(variables['grad'] -
                                 self._variation_optimism_last)**2
        loss[1] = np.linalg.norm(variables['grad'])**2
        prob_optimism = self._meta.opt(loss)
        x_bases = variables['schedule'].x_active_bases
        x_combined = np.dot(variables['meta'].prob, x_bases)
        surrogate_grad = variables['env'].get_grad(x_combined)
        optimism = prob_optimism[0] * surrogate_grad
        optimism_meta = np.dot(x_bases, optimism)
        self._variation_optimism_last = surrogate_grad
        return optimism_meta
