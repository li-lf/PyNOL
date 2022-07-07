from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SurrogateMeta(ABC):
    """The abstract class defines the surrogate loss that passed to meta-algorithm."""

    def __init__(self):
        pass

    @abstractmethod
    def compute_surrogate_meta(self):
        """Compute the surrogate loss that passed to meta-algorithm."""
        pass


class SurrogateMetaFromBase(SurrogateMeta):
    """The class will set the surrogate loss of meta-algorithm as the
    surrogate loss of base-learners."""

    def __init__(self):
        pass

    def compute_surrogate_meta(self, variables):
        """Set the surrogate loss of meta-algorithm as the surrogate loss of base-learners."""
        return variables['surrogate_loss_bases']


class InnerSurrogateMeta(SurrogateMeta):
    """The class defines the inner surrogate loss for meta-algorithm."""

    def __init__(self):
        pass

    def compute_surrogate_meta(self, variables):
        """Set the surrogate loss of meta-algorithm as

        .. math::

            \ell'_t(x)=\langle \\nabla f_t(x_t), x_{t,i} \\rangle,

        where :math:`x_t` is the submitted decision and :math:`x_{t, i}` is the
        decision of base-learner i at round :math:`t`.
        """
        return variables['x_bases'] @ variables['grad']


class InnerSwitchingSurrogateMeta(SurrogateMeta):
    """The class defines the inner surrogate loss with switching cost for
    meta-algorithm.

    Args:
        penalty (float): Penalty coefficient of switching cost term.
        norm (int): Order of norm :math:`p`.
        order (int): Order of switching cost :math:`q`.

    """

    def __init__(self, penalty: float, norm: int = 2, order: int = 2):
        self.penalty = penalty
        self.norm = norm
        self.order = order
        self.x_last = None

    def compute_surrogate_meta(self, variables):
        """Set the surrogate loss of meta-algorithm as

        .. math::

            \ell'_t(x)=\langle \\nabla f_t(x_t), x_{t,i} \\rangle + \lVert
            x_{t,i} - x_{t-1, i} \\rVert_p^q,

        where :math:`x_t` is the submitted decision and :math:`x_{t, i}` is the
        decision of base-learner i at round :math:`t`.
        """
        loss = self.inner_switching(variables['x_bases'], variables['grad'],
                                    self.penalty, self.x_last, self.norm,
                                    self.order)
        self.x_last = variables['x_bases']
        return loss

    @staticmethod
    def inner_switching(x: np.ndarray,
                        gradient: np.ndarray,
                        penalty: float,
                        x_last: Optional[np.ndarray],
                        norm: int = 2,
                        order: int = 2):
        if x_last is None:
            x_last = x
        return np.dot(x, gradient) + penalty * np.linalg.norm(
            x - x_last, ord=norm, axis=1)**order
