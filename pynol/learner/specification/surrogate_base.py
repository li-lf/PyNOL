from abc import ABC, abstractmethod

import numpy as np


class SurrogateBase(ABC):
    """The abstract class defines the surrogate loss functions and surrogate
    gradient (if possible) for base-learners."""

    def __init__(self):
        pass

    @abstractmethod
    def compute_surrogate_base(self, variables):
        """Compute the surrogate loss functions and surrogate
        gradient (if possible) for base-learners."""
        raise NotImplementedError()


class LinearSurrogateBase(SurrogateBase):
    """The class defines the linear surrogate loss function for base-learners."""

    def __init__(self):
        pass

    def compute_surrogate_base(self, variables):
        """Compute the surrogate loss functions and surrogate
        gradient (if possible) for base-learners.

        Replace original convex function :math:`f_t(x)` with

        .. math::

            f'_t(x)=\langle \\nabla f_t(x_t),x - x_t \\rangle,

        for all base-learners, where :math:`x_t` is the submitted decision at
        round :math:`t`. Since the gradient of any decision for the linear
        function is :math:`\\nabla f_t(x_t)`, this method will return it also to
        reduce the gradient query complexity for base-learners.

        Args:
            variables (dict): intermediate variables of the learning process at
                current round.

        Returns:
            tuple: tuple contains:
                surrogate_func (Callable): Surrogate function for base-learners. \n
                surrogate_grad (numpy.ndarray): Surrogate gradient for base-learners.
        """
        return lambda x: np.dot(x - variables['x'], variables['grad']), variables['grad']


class InnerSurrogateBase(SurrogateBase):
    """The class defines the inner surrogate loss function for base-learners."""

    def __init__(self):
        pass

    def compute_surrogate_base(self, variables):
        """Compute the surrogate loss functions and surrogate
        gradient (if possible) for base-learners.

        Replace original convex function :math:`f_t(x)` with

        .. math::

            f'_t(x)=\langle \\nabla f_t(x_t), x \\rangle,

        for all base-learners, where :math:`x_t` is the submitted decision at
        round :math:`t`. Since the gradient of any decision for the inner
        function is :math:`\\nabla f_t(x_t)`, this method will return it also to
        reduce the gradient query complexity for base-learners.

        Args:
            variables (dict): intermediate variables of the learning process at
                current round.
        Returns:
            tuple: tuple contains:
                surrogate_func (Callable): Surrogate function for base-learners. \n
                surrogate_grad (numpy.ndarray): Surrogate gradient for base-learners.
        """
        return lambda x: np.dot(x, variables['grad']), variables['grad']
