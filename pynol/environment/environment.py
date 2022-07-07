from typing import Callable, Iterable, Optional
from autograd import grad as grad_solver
import numpy as np


class Environment:
    """Class for the environment, including loss function, optimism, and so on.

    At each round, the environment can set two loss functions, one is the origin
    loss function ``func`` :math:`f_t` and the other is the surrogate loss
    function ``surrogate_func`` :math:`f'_t` (if any). The gradient of which function is
    given to the learner is determined by ``use_surrogate_grad``.

    Args:
        func_sequence (Iterable, optional): Loss function sequence for whole
            time horizon.
        optimism (numpy.ndarray): Optimism at the beginning of current round.
        func (Callable, optional): Origin loss function at current round.
        grad (numpy.ndarray, optional): Gradient of all decisions for origin
            loss function, only used when the gradient of all decisions are the
            same, namely, the liner function.
        grad_func (Callable, optional): Gradient function for origin loss
            function. It can be given by the environment to accelerate the running
            time. If it is not given, the origin gradient function will be computed by
            ``autograd``.
        surrogate_func (Callable, optional): Surrogate loss function at current round.
        surrogate_grad (numpy.ndarray, optional): Gradient of all decisions for surrogate
            loss function, only used when the gradient of all decisions are the
            same, namely, the liner function.
        surrogate_grad_func (Callable, optional): Gradient function for surrogate loss
            function. It can be given by the environment to accelerate the running
            time. If it is not given, the surrogate gradient function will be computed by
            ``autograd``.
        use_surrogate_grad (bool): Gradient of which function is returned by the
            environment.
        full_info (bool): Specify the type of feedback: full-information or
            bandit feedback.

    """

    def __init__(self,
                 func_sequence: Optional[Iterable] = None,
                 optimism: Optional[np.ndarray] = None,
                 func: Optional[Callable[[np.ndarray], float]] = None,
                 grad: Optional[np.ndarray] = None,
                 grad_func: Optional[Callable[[np.ndarray], float]] = None,
                 surrogate_func: Optional[Callable[[np.ndarray], float]] = None,
                 surrogate_grad: Optional[np.ndarray] = None,
                 surrogate_grad_func: Optional[Callable[[np.ndarray], float]] = None,
                 use_surrogate_grad: bool = True,
                 full_info: bool = True) -> None:

        self.func_sequence = func_sequence
        self.optimism = optimism
        self.func = func
        self.grad = grad
        self.grad_func = grad_func
        self.surrogate_func = surrogate_func
        self.surrogate_grad = surrogate_grad
        self.surrogate_grad_func = surrogate_grad_func
        self.use_surrogate_grad = use_surrogate_grad
        self.full_info = full_info

    def __getitem__(self, t):
        self.func = self.func_sequence[t]
        self.grad = None
        self.grad_func = None
        self.surrogate_func = None
        self.surrogate_grad = None
        self.surrogate_grad_func = None
        return self

    def get_loss(self, x: np.ndarray):
        """Get the loss value of the decision :math:`x`.

        Args:
            x (numpy.ndarray): Decision of the learner.

        Returns:
            tuple: tuple contains:
                loss (float): Origin loss value.\n
                surrogate_loss (float): Surrogate loss value
        """
        loss = self.func(x)
        surrogate_loss = self.surrogate_func(
            x) if self.surrogate_func else None
        return loss, surrogate_loss

    def get_grad(self, x: np.ndarray):
        """Get the gradient of the decision :math:`x`.

        Args:
            x (numpy.ndarray): Decision of the learner.

        Returns:
            numpy.ndarray: Gradient of the decision :math:`x`.
        """
        if self.use_surrogate_grad:
            if self.surrogate_grad is not None:
                return self.surrogate_grad
            elif self.surrogate_grad_func is not None:
                return self.surrogate_grad_func(x)
            elif self.surrogate_func is not None:
                self.surrogate_grad_func = grad_solver(self.surrogate_func)
                return self.surrogate_grad_func(x)
            else:
                pass
        if self.grad is not None:
            return self.grad
        elif self.grad_func is not None:
            return self.grad_func(x)
        else:
            self.grad_func = grad_solver(self.func)
            return self.grad_func(x)
